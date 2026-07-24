//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// COMBO F (v2, REAL FIX) -- Combo D/F's K-split (small-rows + primary +
// grouped tiers), fixing a genuine deadlock found after a real-hardware
// hang.
//
// v1 reset ready_ctr/consumed_ctr and re-barriered INSIDE the n-loop --
// once per N-tile within a single kernel dispatch, not once per dispatch.
// Hart1 (producer)'s transition from iteration n to n+1 was:
// scp_signal(ready, wid_final) [last row of n] -> scp_signal(ready, 0)
// [reset for n+1] -> scp_signal(consumed, 0) -> et_barrier() -- with
// NOTHING forcing hart1 to wait for hart0 to actually consume n's last
// row first. If hart1 reached the reset before hart0's
// scp_wait(ready_ctr, wid_final) for n's last row observed the real
// value, ready_ctr was overwritten to 0 out from under that wait -- a
// value hart0 was waiting for that would never reappear. Hart1 then
// blocked at ITS OWN et_barrier() for n+1, which hart0 could never reach
// because it was still stuck earlier in the code waiting on n's stale
// ready value. Both harts deadlocked, permanently. This only fires when
// N>1 within a single dispatch (decode's N=1 gives no "next iteration"
// to race into) -- exactly why every n=1 oracle test passed while the
// real board's PP256 (N=256) benchmark phase hung for 48+ minutes before
// being force-cancelled.
//
// v2 fix: reset ready_ctr/consumed_ctr and barrier EXACTLY ONCE, before
// the entire i3/i2/n loop nest, with a single monotonically-increasing
// wid spanning the WHOLE dispatch's units of work -- matching the
// already-shipped, already-correct mul_mat_Q8_0_matrix_engine.c (the
// tensor-engine kernel Combo G dispatches to), which uses exactly this
// reset-once/monotonic-wid pattern and was the reference this should
// have matched from the start instead of re-deriving the handshake
// per-iteration.
//
// Verified via llama.cpp's own test-backend-ops oracle against the real
// ET software-emulator backend, INCLUDING a new n>1 case specifically
// added to exercise the cross-iteration boundary that v1's n=1-only
// tests could never reach.
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

#include <stdint.h>

#define STRIDE_M            2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT     1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS 256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64 /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS     8    /* max rows per minion for K-split */
#define KSPLIT_GROUP_ROWS   4
#define TILE_KB             256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define SIMPLE_X2_ROWS      2

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

    // K-split decision (all three tiers)
    const int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    const int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    const int     is_hart1              = hart_id & 1;
    const int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    const int64_t k_half                = K_blocks / 2;
    const int     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
    const int     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    const int     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    if (use_ksplit) {
        /*
         * Each hart of the minion processes half the K dimension for the
         * SAME row, combined via a single-buffered L2-SCP producer/
         * consumer exchange. Reset ONCE for the whole dispatch, wid
         * monotonically increasing across every (i3,i2,n,m) unit of
         * work -- see file header for why a per-n reset deadlocked.
         */
        const int64_t k_start = is_hart1 ? k_half : 0;
        const int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        const uint64_t      scp_base    = (uint64_t) local_minion * 192;
        volatile float *    l2scp_data  = (volatile float *) et_shire_l2scp_local(scp_base);
        volatile uint32_t * ready_ctr    = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 64);
        volatile uint32_t * consumed_ctr = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 128);

        uint32_t wid = 0;

        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);
        }
        et_barrier(ET_BARRIER_MINION);
        if (!is_hart1) {
            evict_to_l2((const void *) ready_ctr, 1, 64);
            WAIT_CACHEOPS;
            evict_to_l2((const void *) consumed_ctr, 1, 64);
            WAIT_CACHEOPS;
        }

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

                    if (is_hart1) {
                        for (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                            const block_q8_0 * q_row = (const block_q8_0 *) (src0_ptr2 + m * nb01);
                            float partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                            if (wid >= 1) {
                                scp_wait(consumed_ctr, wid);
                            }
                            *l2scp_data = partial;
                            FENCE;
                            flush_to_l2((const void *) l2scp_data, 1, 64);
                            WAIT_CACHEOPS;
                            wid++;
                            scp_signal(ready_ctr, wid);
                        }
                    } else {
                        for (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                            const block_q8_0 * q_row = (const block_q8_0 *) (src0_ptr2 + m * nb01);
                            float partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                            wid++;
                            scp_wait(ready_ctr, wid);
                            evict_to_l2((const void *) l2scp_data, 1, 64);
                            WAIT_CACHEOPS;
                            FENCE;
                            float other = *l2scp_data;
                            scp_signal(consumed_ctr, wid);

                            float * dst_entry = (float *) (dst_ptr2 + n * nbd1 + m * sizeof(float));
                            float   sum       = partial + other;
                            if (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((volatile float *) dst_entry, sum);
                        }
                    }
                }
            }
        }
    } else if (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime. Same reset-
         * once/monotonic-wid fix as use_ksplit above.
         */
        const int64_t        k_start    = is_hart1 ? k_half : 0;
        const int64_t        k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        const uint64_t        scp_base    = (uint64_t) local_minion * 192;
        volatile float *      l2scp_data  = (volatile float *) et_shire_l2scp_local(scp_base); /* 4 floats */
        volatile uint32_t *   ready_ctr    = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 64);
        volatile uint32_t *   consumed_ctr = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 128);

        uint32_t wid = 0;

        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);
        }
        et_barrier(ET_BARRIER_MINION);
        if (!is_hart1) {
            evict_to_l2((const void *) ready_ctr, 1, 64);
            WAIT_CACHEOPS;
            evict_to_l2((const void *) consumed_ctr, 1, 64);
            WAIT_CACHEOPS;
        }

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

                    if (is_hart1) {
                        for (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                            const int64_t m0 = m_base;
                            const int64_t m1 = m0 + STRIDE_M_KSPLIT;
                            const int64_t m2 = m1 + STRIDE_M_KSPLIT;
                            const int64_t m3 = m2 + STRIDE_M_KSPLIT;

                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            for (int64_t kb = 0; kb < k_len; kb += TILE_KB) {
                                int64_t tile_len = k_len - kb;
                                if (tile_len > TILE_KB) {
                                    tile_len = TILE_KB;
                                }
                                const float * b_tile = b_col_base + (k_start + kb) * 32;
                                const int64_t row_kb = k_start + kb;

                                if (m0 < M) {
                                    s0 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m1 < M) {
                                    s1 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m2 < M) {
                                    s2 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m3 < M) {
                                    s3 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb, b_tile, tile_len);
                                }
                            }

                            if (wid >= 1) {
                                scp_wait(consumed_ctr, wid);
                            }
                            l2scp_data[0] = s0;
                            l2scp_data[1] = s1;
                            l2scp_data[2] = s2;
                            l2scp_data[3] = s3;
                            FENCE;
                            flush_to_l2((const void *) l2scp_data, 1, 64);
                            WAIT_CACHEOPS;
                            wid++;
                            scp_signal(ready_ctr, wid);
                        }
                    } else {
                        for (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                            const int64_t m0 = m_base;
                            const int64_t m1 = m0 + STRIDE_M_KSPLIT;
                            const int64_t m2 = m1 + STRIDE_M_KSPLIT;
                            const int64_t m3 = m2 + STRIDE_M_KSPLIT;

                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            for (int64_t kb = 0; kb < k_len; kb += TILE_KB) {
                                int64_t tile_len = k_len - kb;
                                if (tile_len > TILE_KB) {
                                    tile_len = TILE_KB;
                                }
                                const float * b_tile = b_col_base + (k_start + kb) * 32;
                                const int64_t row_kb = k_start + kb;

                                if (m0 < M) {
                                    s0 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m1 < M) {
                                    s1 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m2 < M) {
                                    s2 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb, b_tile, tile_len);
                                }
                                if (m3 < M) {
                                    s3 += compute_row_dot_q8_0((const block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb, b_tile, tile_len);
                                }
                            }

                            wid++;
                            scp_wait(ready_ctr, wid);
                            evict_to_l2((const void *) l2scp_data, 1, 64);
                            WAIT_CACHEOPS;
                            FENCE;
                            const float p0 = l2scp_data[0];
                            const float p1 = l2scp_data[1];
                            const float p2 = l2scp_data[2];
                            const float p3 = l2scp_data[3];
                            scp_signal(consumed_ctr, wid);

                            float *     c_base = (float *) (dst_ptr2 + n * nbd1);
                            const float b0     = bias_n ? bias_n[m0] : 0.0f;
                            const float b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            const float b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            const float b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            if (m0 < M) {
                                atomic_store_f32((volatile float *) (c_base + m0), s0 + p0 + b0);
                            }
                            if (m1 < M) {
                                atomic_store_f32((volatile float *) (c_base + m1), s1 + p1 + b1);
                            }
                            if (m2 < M) {
                                atomic_store_f32((volatile float *) (c_base + m2), s2 + p2 + b2);
                            }
                            if (m3 < M) {
                                atomic_store_f32((volatile float *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } else if (K_blocks > TILE_KB) {
        /* Tile-outer path, unchanged from Combo A/D. */
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
        /* Simple path (mask-hoist + x2-row-batch), unchanged from Combo A/D. */
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
                        for (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            const int64_t      m1     = m0 + STRIDE_M;
                            const block_q8_0 * q_row0 = (const block_q8_0 *) (src0_ptr2 + m0 * nb01);

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