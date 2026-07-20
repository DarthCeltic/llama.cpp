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

// Using the block prefetch logic
static inline void prefetch_weight_row(const void* start_ptr, int64_t num_blocks, uint32_t worker_id) {
    const uint64_t cache_line_size = 64;
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (num_blocks * sizeof(block_q8_0));

    // 1. Align to cache lines and calculate range
    uint64_t startCL = (self_ptr + 63) >> 6;
    uint64_t endCL = (self_ptr_end + 63) >> 6;
    if (endCL >= startCL) {
        uint64_t total_lines = endCL - startCL + 1;
        // 2. Load balance across the minions in the Shire (assuming 8 per group for this logic)
        // Adjust worker_id if using global_id
        uint32_t local_worker_id = worker_id % 8;
        uint64_t lines_per_minion = total_lines >> 3;
        uint64_t extra = total_lines & 7;

        uint64_t offset = local_worker_id * lines_per_minion;
        if (local_worker_id < extra) {
            offset += local_worker_id;
            lines_per_minion++;
        } else {
            offset += extra;
        }
        self_ptr = (startCL + offset) << 6;
        int pending_lines = lines_per_minion;

        // 3. Hardware Prefetch Loop (16 lines at a time)
        for (; pending_lines > 0; pending_lines -= 16) {
            uint64_t current_batch = (pending_lines > 16 ? 16 : pending_lines) - 1;
            uint64_t self_size = current_batch; // bits 3:0

            __asm__ __volatile__ (
                "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
                "addi  x31, zero, 64\n"  // Stride = 64 bytes
                "or    x3, x1, %[ptr]\n"  // Combine Dest + VA
                "or    x3, x3, %[sz]\n"  // Combine with NumLines
                "csrw  0x81f, x3\n"  // prefetch_va
                :
                : [ptr] "r" (self_ptr),
                  [sz] "r" (self_size)
                : "x1", "x3", "x31", "memory"
            );
            self_ptr += (16 * 64);
        }
    }
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

    // Largest multiples of 8 and 4 <= N. mul_mat_Q8_0 is the dominant cost
    // for Q8_0-quantized models on this target and is memory-bound: the
    // per-column loop re-reads and re-converts every weight block once per
    // output column. When N >= 8 (SmolVLM2's vision-encoder GEMMs process
    // many patches together, so this is the common case there) this
    // processes 8 columns at a time so each weight block's gather+convert
    // happens once and is reused across all 8 -- see
    // compute_row_dot_product_q8_0_masked_x8 in block_ops.h. The N4/N1
    // tiles below only ever handle the remainder (N % 8), same as before.
    const int64_t N8 = N & ~(int64_t)7;
    const int64_t N4 = N & ~(int64_t)3;

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

            for (int64_t n = 0; n < N8; n += 8) {
                // src1 is F32, so column pointers move by nb11
                const float* b0_col_base = (const float*)(src1_ptr2 + (n + 0) * nb11);
                const float* b1_col_base = (const float*)(src1_ptr2 + (n + 1) * nb11);
                const float* b2_col_base = (const float*)(src1_ptr2 + (n + 2) * nb11);
                const float* b3_col_base = (const float*)(src1_ptr2 + (n + 3) * nb11);
                const float* b4_col_base = (const float*)(src1_ptr2 + (n + 4) * nb11);
                const float* b5_col_base = (const float*)(src1_ptr2 + (n + 5) * nb11);
                const float* b6_col_base = (const float*)(src1_ptr2 + (n + 6) * nb11);
                const float* b7_col_base = (const float*)(src1_ptr2 + (n + 7) * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

                    unsigned long saved_mask;
                    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ volatile("mov.m.x m0, x0, 0xFF");

                    compute_row_dot_product_q8_0_masked_x8(
                        q_row, K_blocks,
                        b0_col_base, b1_col_base, b2_col_base, b3_col_base,
                        b4_col_base, b5_col_base, b6_col_base, b7_col_base,
                        &sum0, &sum1, &sum2, &sum3, &sum4, &sum5, &sum6, &sum7);

                    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));

                    float* d0 = (float*)(dst_ptr2 + (n + 0) * nbd1 + m * sizeof(float));
                    float* d1 = (float*)(dst_ptr2 + (n + 1) * nbd1 + m * sizeof(float));
                    float* d2 = (float*)(dst_ptr2 + (n + 2) * nbd1 + m * sizeof(float));
                    float* d3 = (float*)(dst_ptr2 + (n + 3) * nbd1 + m * sizeof(float));
                    float* d4 = (float*)(dst_ptr2 + (n + 4) * nbd1 + m * sizeof(float));
                    float* d5 = (float*)(dst_ptr2 + (n + 5) * nbd1 + m * sizeof(float));
                    float* d6 = (float*)(dst_ptr2 + (n + 6) * nbd1 + m * sizeof(float));
                    float* d7 = (float*)(dst_ptr2 + (n + 7) * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)d0, sum0);
                    atomic_store_f32((volatile float*)d1, sum1);
                    atomic_store_f32((volatile float*)d2, sum2);
                    atomic_store_f32((volatile float*)d3, sum3);
                    atomic_store_f32((volatile float*)d4, sum4);
                    atomic_store_f32((volatile float*)d5, sum5);
                    atomic_store_f32((volatile float*)d6, sum6);
                    atomic_store_f32((volatile float*)d7, sum7);
                }
            }

            // Remainder columns (N % 8, down to the next multiple of 4) --
            // unchanged N=4 path, same as before, just starting from N8.
            for (int64_t n = N8; n < N4; n += 4) {
                // src1 is F32, so column pointers move by nb11
                const float* b0_col_base = (const float*)(src1_ptr2 + (n + 0) * nb11);
                const float* b1_col_base = (const float*)(src1_ptr2 + (n + 1) * nb11);
                const float* b2_col_base = (const float*)(src1_ptr2 + (n + 2) * nb11);
                const float* b3_col_base = (const float*)(src1_ptr2 + (n + 3) * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum0, sum1, sum2, sum3;

                    // Set the vector mask ONCE for this row's whole K_blocks
                    // loop instead of paying a save/set/restore CSR sequence
                    // on every 32-element block call -- see
                    // compute_row_dot_product_q8_0_masked_x4 in block_ops.h.
                    // Mask CSR state is untouched by anything else between
                    // these two asm blocks, so this is value-identical to
                    // the per-call save/set/restore it replaces.
                    unsigned long saved_mask;
                    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ volatile("mov.m.x m0, x0, 0xFF");

                    // Whole-row call: was one compute_block_dot_product_q8_0_
                    // masked_x4 call per K-block (each paying its own 4-column
                    // horizontal reduce), now one call that reduces each
                    // column exactly once for the entire row -- see the
                    // comment on compute_row_dot_product_q8_0_masked_x4.
                    compute_row_dot_product_q8_0_masked_x4(
                        q_row, K_blocks,
                        b0_col_base, b1_col_base, b2_col_base, b3_col_base,
                        &sum0, &sum1, &sum2, &sum3);

                    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));

                    // Store results in dst[m, n..n+3, i2, i3]
                    float* d0 = (float*)(dst_ptr2 + (n + 0) * nbd1 + m * sizeof(float));
                    float* d1 = (float*)(dst_ptr2 + (n + 1) * nbd1 + m * sizeof(float));
                    float* d2 = (float*)(dst_ptr2 + (n + 2) * nbd1 + m * sizeof(float));
                    float* d3 = (float*)(dst_ptr2 + (n + 3) * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)d0, sum0);
                    atomic_store_f32((volatile float*)d1, sum1);
                    atomic_store_f32((volatile float*)d2, sum2);
                    atomic_store_f32((volatile float*)d3, sum3);
                }
            }

            // Remainder columns (N % 4), and the whole loop when N < 4 (e.g.
            // single-token decode) -- original per-column path, unchanged.
            for (int64_t n = N4; n < N; n++) {
                // src1 is F32, so column pointer moves by nb11
                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = 0.0f;

                    // Set the vector mask ONCE for this row's whole K_blocks
                    // loop instead of paying a save/set/restore CSR sequence
                    // on every 32-element block call -- see
                    // compute_block_dot_product_q8_0_masked in block_ops.h.
                    // Mask CSR state is untouched by anything else between
                    // these two asm blocks, so this is value-identical to
                    // the per-call save/set/restore it replaces.
                    unsigned long saved_mask;
                    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ volatile("mov.m.x m0, x0, 0xFF");

                    for (int64_t kb = 0; kb < K_blocks; kb++) {
                        // q_row is a pointer to blocks, so + kb moves by sizeof(block_q8_0)
                        // b_col is float*, so we move 32 elements (kb << 5)
                        sum += compute_block_dot_product_q8_0_masked(q_row + kb, b_col_base + (kb << 5));
                    }

                    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));

                    // Store result in dst[m, n, i2, i3]
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
        }
    }
    return 0;
}
