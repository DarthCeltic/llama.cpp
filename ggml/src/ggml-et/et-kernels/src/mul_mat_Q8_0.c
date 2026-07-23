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

// Lightweight next-row prefetch.
//
// A prior attempt (PR#168's first version) reused the existing
// prefetch_weight_row() helper, calling it 8 times per row (once per
// worker_id 0..7) to cover the whole row -- its 8-way partitioning
// (division/modulo math to split a range across 8 COOPERATING workers) was
// designed for multiple harts prefetching a SHARED range together, not this
// loop's access pattern where exactly ONE hart owns its row exclusively. A
// real board run proved that 8x-per-row overhead is a net regression
// (-3.3%, tight CV both sides -- see corpus entry
// mask_hoist_plus_weight_row_prefetch_commit_11b05e9c). This is a from-
// scratch, independent reimplementation of the SAME underlying hardware
// primitive (the prefetch_va CSR, 0x81f) WITHOUT the unneeded worker-
// partitioning math: one direct loop over the row's own cache lines, issued
// ONCE per row instead of 8 times. The CSR's NumLines field is 4 bits (max
// 16 lines per csrw), so a K=2048 row (~34 lines) still needs ~3 CSR writes
// -- but with none of the division/modulo setup repeated 8x for no reason.
static inline void prefetch_row_light(const void* start_ptr, int64_t num_blocks) {
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (uintptr_t)(num_blocks * (int64_t)sizeof(block_q8_0));
    uint64_t startCL = ((uint64_t)self_ptr + 63) >> 6;
    uint64_t endCL = ((uint64_t)self_ptr_end + 63) >> 6;
    if (endCL < startCL) {
        return;
    }
    int64_t pending_lines = (int64_t)(endCL - startCL + 1);
    self_ptr = (uintptr_t)(startCL << 6);

    for (; pending_lines > 0; pending_lines -= 16) {
        uint64_t current_batch = (uint64_t)((pending_lines > 16 ? 16 : pending_lines) - 1);
        __asm__ __volatile__(
            "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
            "or    x3, x1, %[ptr]\n"           // Combine Dest + VA
            "or    x3, x3, %[sz]\n"            // Combine with NumLines
            "csrw  0x81f, x3\n"                // prefetch_va
            :
            : [ptr] "r"(self_ptr), [sz] "r"(current_batch)
            : "x1", "x3", "memory"
        );
        self_ptr += (16 * 64);
    }
}

// Row dot product with vectorized fp16 block-scale unpack (unchanged from
// the already-verified scale-unpack lever).
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

            // Original per-element hart striping (preserves full 2048-way
            // parallelism for this model's M -- see corpus entry
            // ..._PARALLELISM_BUG_20260723 for why restructuring this to
            // stripe over cache lines/tiles instead collapses parallelism).
            for (int64_t n = 0; n < N; n++) {
                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                for (int64_t m = (int64_t)hart_id; m < M; m += stride_m) {
                    // Prefetch this hart's OWN next row (m + stride_m) into
                    // L2 before computing the current one, so its memory
                    // latency overlaps this row's compute instead of
                    // stalling the next iteration. One lightweight call per
                    // row (see prefetch_row_light above), not 8.
                    const int64_t m_next = m + stride_m;
                    if (m_next < M) {
                        const void* next_row_ptr = (const void*)(src0_ptr2 + m_next * nb01);
                        prefetch_row_light(next_row_ptr, K_blocks);
                    }

                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = compute_row_dot_product_q8_0_scaleunpack(q_row, b_col_base, K_blocks);

                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
        }
    }
    return 0;
}
