//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// COMBO B (v3, REAL FIX) -- Combo A (Rehan structural base, attribution +
// exclusions documented there) + single-call-per-row DRAM prefetch of this
// hart's NEXT row(s), hiding their load latency behind the current row's
// compute. Uses the prefetch_va CSR (0x81f).
//
// v1 crashed real hardware: "ET: stream error detected at synchronization
// point... rc=-6" (SIGABRT) during warmup, on the very first dispatch.
// v2 added a FENCE+WAIT_CACHEOPS drain after the CSR issue, matching the
// drain pattern used elsewhere for cache-op CSRs -- IDENTICAL crash
// recurred. The drain hypothesis was wrong.
//
// v3 root cause: compared byte-for-byte against the two OTHER cache-op CSRs
// in this same 0x8xx family that are proven working (used in the
// already-shipped tensor-engine kernel) -- flush_to_l2 (0x8BF) and
// evict_to_l2 (0x89F), both in platform.h. BOTH explicitly load a stride
// value into x31 via "mv x31, %[x31]" in the SAME atomic asm block,
// immediately before the csrw, with x31 listed in the clobber list. v1/v2's
// prefetch_row_light NEVER set x31 at all -- not loaded, not clobbered, not
// even mentioned. If this CSR family reads x31 as an implicit stride
// parameter (as its two siblings do), every prefetch_va issue in v1/v2 went
// out with whatever garbage value the compiler happened to leave in x31
// from unrelated scheduled code -- not zero, not a sane default. A garbage
// stride on a MULTI-LINE prefetch (up to 16 lines per csrw) can compute
// wildly out-of-range addresses on the very first call, which matches what
// was observed: deterministic, first-dispatch, not load-dependent. This
// also explains why the drain fix (v2) did nothing -- draining a cache-op
// doesn't help when the address it's operating on was already garbage.
//
// v3 fix: explicitly load x31 with a 64-byte stride (matching every
// single-line-at-a-time flush_to_l2 call elsewhere in this codebase, e.g.
// flush_to_l2(ptr, 1, 64)) in the same atomic asm block, exactly matching
// the verified-working sibling CSRs' calling convention instead of
// inventing a new one.
//
// Kernel body is RehanQasim-dev's Q8_0 matrix-engine GEMM design
// (aifoundry-org/llama.cpp commit 84405bb7f86458c60c9f47a6a140f1c876da8532),
// rebuilt against current shared main -- see Combo A's commit for full
// attribution/authorization and the K-split exclusion rationale.
//
// Verified via llama.cpp's own test-backend-ops oracle against the real ET
// software-emulator backend before submission. Real-hardware confirmation
// pending -- flagged, not assumed fixed a third time without proof.
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

#include <stdint.h>

#define STRIDE_M       2048 /* 32 shires x 32 minions x 2 harts */
#define TILE_KB        256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define SIMPLE_X2_ROWS 2

// Prefetch one row's worth of Q8_0 blocks into L2, ONE call covering the
// whole row (not 8 -- see file header for why the original 8x/row attempt
// was a proven regression).
static inline void prefetch_row_light(const void * start_ptr, int64_t num_blocks) {
    uintptr_t self_ptr     = (uintptr_t) start_ptr;
    uintptr_t self_ptr_end = self_ptr + (uintptr_t) (num_blocks * (int64_t) sizeof(block_q8_0));
    uint64_t  startCL      = ((uint64_t) self_ptr + 63) >> 6;
    uint64_t  endCL        = ((uint64_t) self_ptr_end + 63) >> 6;
    if (endCL < startCL) {
        return;
    }
    int64_t pending_lines = (int64_t) (endCL - startCL + 1);
    self_ptr               = (uintptr_t) (startCL << 6);

    const uint64_t x31_val = 64ULL; /* stride between consecutive prefetched lines: 1 cache line */

    for (; pending_lines > 0; pending_lines -= 16) {
        uint64_t current_batch = (uint64_t) ((pending_lines > 16 ? 16 : pending_lines) - 1);
        __asm__ __volatile__(
            "mv    x31, %[stride]\n"           // stride register -- REQUIRED by this CSR family,
                                                // matching flush_to_l2/evict_to_l2's exact convention
            "li    x1, 0x400000000000000 \n"   // Dest = L2 (bits 59:58 = 01)
            "or    x3, x1, %[ptr]\n"           // Combine Dest + VA
            "or    x3, x3, %[sz]\n"            // Combine with NumLines
            "csrw  0x81f, x3\n"                // prefetch_va
            :
            : [ptr] "r"(self_ptr), [sz] "r"(current_batch), [stride] "r"(x31_val)
            : "x1", "x3", "x31", "memory");
        self_ptr += (16 * 64);
    }
    FENCE;
    WAIT_CACHEOPS;
}

int entry_point(struct ggml_et_mm_q8_params * params, void * env) {
    uint64_t hart_id = get_hart_id();

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

    // Optional residual bias
    const char * bias_base = (const char *) params->bias.data;
    const size_t nbb1      = params->bias.nb[1];
    const size_t nbb2      = params->bias.nb[2];
    const size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    const int64_t K_blocks      = K / 32;
    const int     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    if (K_blocks > TILE_KB) {
        /* Tile-outer path (large K, e.g. other regression models) --
         * unchanged, uses only confirmed-available primitives. */
        for (int64_t i3 = 0; i3 < ne13; i3++) {
            const int64_t i03       = i3 / r3;
            const char *  src0_ptr3 = (const char *) params->src0.data + i03 * nb03;
            const char *  src1_ptr3 = (const char *) params->src1.data + i3 * nb13;
            char *        dst_ptr3  = (char *) params->dst.data + i3 * nbd3;
            const char *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (const char *) 0;

            for (int64_t i2 = 0; i2 < ne12; i2++) {
                const int64_t i02       = i2 / r2;
                const char *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                const char *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                char *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                const char *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (const char *) 0;

                for (int64_t n = 0; n < N; n++) {
                    const float * b_col_base = (const float *) (src1_ptr2 + n * nb11);
                    const float * bias_n     = bias_ptr2 ? (const float *) (bias_ptr2 + n * nbb1) : (const float *) 0;

                    for (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        const int64_t m1 = m0 + STRIDE_M;
                        const int64_t m2 = m0 + STRIDE_M * 2;
                        const int64_t m3 = m0 + STRIDE_M * 3;

                        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        for (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            if (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            const float * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            if (m1 < M) {
                                s1 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            if (m2 < M) {
                                s2 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            if (m3 < M) {
                                s3 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        float *     dst_base = (float *) (dst_ptr2 + n * nbd1);
                        const float b0       = bias_n ? bias_n[m0] : 0.0f;
                        const float b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        const float b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        const float b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((volatile float *) (dst_base + m0), s0 + b0);
                        if (m1 < M) {
                            atomic_store_f32((volatile float *) (dst_base + m1), s1 + b1);
                        }
                        if (m2 < M) {
                            atomic_store_f32((volatile float *) (dst_base + m2), s2 + b2);
                        }
                        if (m3 < M) {
                            atomic_store_f32((volatile float *) (dst_base + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } else {
        /*
         * Simple path: x2-row B-reuse when nb01 is 32-byte aligned, with
         * single-call-per-row prefetch of this hart's next row(s).
         */
        for (int64_t i3 = 0; i3 < ne13; i3++) {
            const int64_t i03       = i3 / r3;
            const char *  src0_ptr3 = (const char *) params->src0.data + i03 * nb03;
            const char *  src1_ptr3 = (const char *) params->src1.data + i3 * nb13;
            char *        dst_ptr3  = (char *) params->dst.data + i3 * nbd3;
            const char *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (const char *) 0;

            for (int64_t i2 = 0; i2 < ne12; i2++) {
                const int64_t i02       = i2 / r2;
                const char *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                const char *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                char *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                const char *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (const char *) 0;

                for (int64_t n = 0; n < N; n++) {
                    const float * b_col_base = (const float *) (src1_ptr2 + n * nb11);
                    const float * bias_n     = bias_ptr2 ? (const float *) (bias_ptr2 + n * nbb1) : (const float *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    if (use_simple_x2) {
                        const int64_t stride2 = STRIDE_M * SIMPLE_X2_ROWS;
                        for (int64_t m0 = hart_id; m0 < M; m0 += stride2) {
                            const int64_t      m1     = m0 + STRIDE_M;
                            const block_q8_0 * q_row0 = (const block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            const int64_t m0_next = m0 + stride2;
                            if (m0_next < M) {
                                prefetch_row_light((const void *) (src0_ptr2 + m0_next * nb01), K_blocks);
                                const int64_t m1_next = m0_next + STRIDE_M;
                                if (m1_next < M) {
                                    prefetch_row_light((const void *) (src0_ptr2 + m1_next * nb01), K_blocks);
                                }
                            }

                            if (m1 < M) {
                                const block_q8_0 * q_row1 = (const block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                float              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                float * dst0 = (float *) (dst_ptr2 + n * nbd1 + m0 * sizeof(float));
                                float * dst1 = (float *) (dst_ptr2 + n * nbd1 + m1 * sizeof(float));
                                if (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((volatile float *) dst0, s0);
                                atomic_store_f32((volatile float *) dst1, s1);
                            } else {
                                float   sum = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                float * dst = (float *) (dst_ptr2 + n * nbd1 + m0 * sizeof(float));
                                if (bias_n) {
                                    sum += bias_n[m0];
                                }
                                atomic_store_f32((volatile float *) dst, sum);
                            }
                        }
                    } else {
                        for (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            const block_q8_0 * q_row = (const block_q8_0 *) (src0_ptr2 + m * nb01);

                            const int64_t m_next = m + STRIDE_M;
                            if (m_next < M) {
                                prefetch_row_light((const void *) (src0_ptr2 + m_next * nb01), K_blocks);
                            }

                            float sum = q8_dot_compute(q_row, b_col_base, K_blocks);

                            float * dst_entry = (float *) (dst_ptr2 + n * nbd1 + m * sizeof(float));
                            if (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((volatile float *) dst_entry, sum);
                        }
                    }

                    q8_dot_end(&q8_state);
                }
            }
        }
    }

    return 0;
}