//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    const int64_t stride_m = 2048;

    // Matrix dimensions
    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
        const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
        char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
            char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

            // Cache-line-owned output stores.
            //
            // atomic_store_f32 uses a global read-modify-write for every single
            // output element. That atomic is only needed because dst is
            // m-contiguous and the default striping (m = hart_id, hart_id +
            // stride_m, ...) scatters 16 adjacent m's -- one 64B/16-float cache
            // line, on this non-coherent-L1D target -- across 16 DIFFERENT
            // harts, so without an atomic two harts could race the same line.
            //
            // When M is a multiple of 16 and dst is 64B-aligned (true for every
            // Q8_0 weight matmul in this model family: hidden_size, ffn
            // intermediate_size, and vocab_size are all multiples of 16), we can
            // instead give each hart a WHOLE 64B output line (16 consecutive m,
            // for one n) to itself. Nothing else ever touches that hart's line,
            // so it can use a plain store -- no amoswapg.w, no global-bypass
            // cost. Falls back to the exact original per-element atomic path
            // whenever the alignment doesn't hold; correctness never depends on
            // which branch runs.
            const int64_t M_lines = M >> 4;  // 16 f32 = 64B cache line
            const bool aligned =
                (M >= 16) && ((M & 15) == 0) &&
                (((uintptr_t)dst_ptr2 & 63) == 0) && ((nbd1 & 63) == 0);

            if (aligned) {
                for (int64_t n = 0; n < N; n++) {
                    // src1 is F32, so column pointer moves by nb11
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);
                    float* dst_col = (float*)(dst_ptr2 + n * nbd1);

                    for (int64_t line = (int64_t)hart_id; line < M_lines; line += stride_m) {
                        const int64_t m0 = line << 4;  // first m in this line

                        // Set the vector mask ONCE for this whole 16-row line
                        // instead of paying a save/set/restore CSR sequence per
                        // 32-element block call -- see
                        // compute_block_dot_product_q8_0_masked in block_ops.h.
                        unsigned long saved_mask;
                        __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ volatile("mov.m.x m0, x0, 0xFF");

                        for (int j = 0; j < 16; j++) {
                            // src0 is Q8_0 blocks, row pointer moves by nb01
                            const block_q8_0* q_row =
                                (const block_q8_0*)(src0_ptr2 + (m0 + j) * nb01);
                            float sum = 0.0f;
                            for (int64_t kb = 0; kb < K_blocks; kb++) {
                                sum += compute_block_dot_product_q8_0_masked(
                                    q_row + kb, b_col_base + (kb << 5));
                            }
                            // This hart owns the whole 64B line exclusively --
                            // plain store, no atomic needed.
                            dst_col[m0 + j] = sum;
                        }

                        __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
                    }
                }
            } else {
                // Unaligned fallback: original per-element striped/atomic path.
                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                    for (int64_t m = (int64_t)hart_id; m < M; m += stride_m) {
                        const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                        float sum = 0.0f;

                        unsigned long saved_mask;
                        __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ volatile("mov.m.x m0, x0, 0xFF");

                        for (int64_t kb = 0; kb < K_blocks; kb++) {
                            sum += compute_block_dot_product_q8_0_masked(
                                q_row + kb, b_col_base + (kb << 5));
                        }

                        __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));

                        float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                        atomic_store_f32((volatile float*)dst_entry, sum);
                    }
                }
            }
        }
    }
    return 0;
}
