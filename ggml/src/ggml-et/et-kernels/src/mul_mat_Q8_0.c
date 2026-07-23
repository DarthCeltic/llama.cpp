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

// Row dot product with vectorized fp16 block-scale unpack.
//
// compute_block_dot_product_q8_0_masked() converts each block's fp16 delta to
// f32 with fp16_to_fp32() -- a scalar, branch-laden bit-twiddle (exponent
// zero/subnormal/inf/nan cases, a data-dependent normalize loop for
// subnormals). Q8_0 block deltas are always *normal* fp16 (d = max|q|/127,
// computed at quantization time), so none of those branches ever actually
// fire for this field -- the hardware vector fp16->f32 convert (fcvt.ps.f16,
// already used elsewhere in this file for f16 activation dots) is bit-exact
// for normal inputs and can replace it.
//
// block_q8_0 is 34 bytes (2-byte delta + 32 int8 quants), so 8 consecutive
// block deltas are NOT contiguous in memory -- pack them into a contiguous
// scratch buffer first (8 plain halfword loads/stores, no branch), then do
// ONE vector gather+convert for the group instead of 8 separate scalar
// converts. Falls back to fp16_to_fp32() for the K_blocks % 8 tail.
static inline float compute_row_dot_product_q8_0_scaleunpack(
    const block_q8_0* q_row, const float* b_col_base, int64_t K_blocks) {
    static const int32_t gather_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static const int32_t f16_pattern[8]  = {0, 2, 4, 6, 8, 10, 12, 14};

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    float row_sum = 0.0f;
    const int64_t groups = K_blocks / 8;
    const int64_t tail_start = groups * 8;

    for (int64_t g = 0; g < groups; g++) {
        // Pack 8 block deltas into a contiguous scratch buffer (plain
        // halfword copies, no branch), then convert all 8 to f32 in one
        // vector instruction.
        uint16_t deltas[8] __attribute__((aligned(16)));
        for (int i = 0; i < 8; i++) {
            deltas[i] = q_row[g * 8 + i].d;
        }
        float scales[8] __attribute__((aligned(32)));
        __asm__ volatile(
            "flw.ps    f21, %[pat]\n"
            "fgh.ps    f20, f21(%[d])\n"
            "fcvt.ps.f16 f20, f20\n"
            "fsw.ps    f20, %[out]\n"
            : [out] "=m"(*(float(*)[8])scales)
            : [d] "r"(deltas),
              [pat] "m"(*(const int32_t(*)[8])f16_pattern)
            : "f20", "f21"
        );

        for (int i = 0; i < 8; i++) {
            const block_q8_0* blk = q_row + g * 8 + i;
            const float* b = b_col_base + ((int64_t)(g * 8 + i) << 5);

            float acc;
            __asm__ volatile(
                "flw.ps   f31, %[gb]\n"
                "fbci.pi  f10, 0\n"
                "flw.ps   f12, 0(%[b])\n"
                "fgb.ps   f11, f31(%[a])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps f10, f11, f12, f10\n"
                "flw.ps   f12, 32(%[b])\n"
                "fgb.ps   f11, f31(%[a1])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps f10, f11, f12, f10\n"
                "flw.ps   f12, 64(%[b])\n"
                "fgb.ps   f11, f31(%[a2])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps f10, f11, f12, f10\n"
                "flw.ps   f12, 96(%[b])\n"
                "fgb.ps   f11, f31(%[a3])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps f10, f11, f12, f10\n"
                "fswizz.ps f1, f10, 0xB1\n"
                "fadd.ps   f2, f10, f1, rne\n"
                "fswizz.ps f3, f2, 0x4E\n"
                "fadd.ps   f4, f2, f3, rne\n"
                "fmvz.x.ps t0, f4, 4\n"
                "fbcx.ps   f5, t0\n"
                "fadd.ps   %[out], f4, f5, rne\n"
                : [out] "=f"(acc)
                : [gb] "m"(*(const int32_t(*)[8])gather_bytes),
                  [a] "r"(blk->qs), [a1] "r"(blk->qs + 8),
                  [a2] "r"(blk->qs + 16), [a3] "r"(blk->qs + 24),
                  [b] "r"(b)
                : "memory", "t0", "f1", "f2", "f3", "f4", "f5",
                  "f10", "f11", "f12", "f31"
            );
            row_sum += acc * scales[i];
        }
    }

    for (int64_t kb = tail_start; kb < K_blocks; kb++) {
        row_sum += compute_block_dot_product_q8_0_masked(q_row + kb, b_col_base + (kb << 5));
    }

    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
    return row_sum;
}

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

            // Cache-line-owned output stores. atomic_store_f32 issues a
            // global read-modify-write per output element; that's only
            // needed because the default hart striping scatters 16 adjacent
            // m-outputs (one 64B cache line on this non-coherent-L1D target)
            // across 16 different harts. When M is a multiple of 16 and dst
            // is 64B-aligned (true for every Q8_0 weight matmul in this
            // model family), give each hart a whole output cache line to
            // itself so it can use a plain store instead. Falls back to the
            // original per-element atomic path whenever alignment doesn't
            // hold; correctness never depends on which branch runs.
            const int64_t M_lines = M >> 4;  // 16 f32 = 64B cache line
            const bool aligned =
                (M >= 16) && ((M & 15) == 0) &&
                (((uintptr_t)dst_ptr2 & 63) == 0) && ((nbd1 & 63) == 0);

            if (aligned) {
                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);
                    float* dst_col = (float*)(dst_ptr2 + n * nbd1);

                    for (int64_t line = (int64_t)hart_id; line < M_lines; line += stride_m) {
                        const int64_t m0 = line << 4;
                        for (int j = 0; j < 16; j++) {
                            const block_q8_0* q_row =
                                (const block_q8_0*)(src0_ptr2 + (m0 + j) * nb01);
                            // This hart owns the whole 64B line exclusively --
                            // plain store, no atomic needed.
                            dst_col[m0 + j] =
                                compute_row_dot_product_q8_0_scaleunpack(q_row, b_col_base, K_blocks);
                        }
                    }
                }
            } else {
                // Unaligned fallback: original per-element striped/atomic path,
                // still using the scale-unpack row dot product.
                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                    for (int64_t m = (int64_t)hart_id; m < M; m += stride_m) {
                        const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                        float sum = compute_row_dot_product_q8_0_scaleunpack(q_row, b_col_base, K_blocks);

                        float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                        atomic_store_f32((volatile float*)dst_entry, sum);
                    }
                }
            }
        }
    }
    return 0;
}
