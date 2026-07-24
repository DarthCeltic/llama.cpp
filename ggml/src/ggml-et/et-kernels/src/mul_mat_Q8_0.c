//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// COMBO D -- Combo A (see that file for full attribution/authorization)
// PLUS Rehan's K-split "small rows" carve-out, reimplemented from scratch
// using REAL, VERIFIED synchronization primitives instead of his fork-only
// et_sem_post/et_sem_wait (which don't exist on current shared main -- see
// Combo A's header for the full ABI investigation).
//
// The real primitives (et_barrier, scp_signal, scp_wait) were found already
// confirmed present in current main's platform.h AND already used, in
// exactly this producer/consumer L2-SCP-exchange role, inside the
// already-shipped mul_mat_Q8_0_matrix_engine.c (see its hart1-producer /
// hart0-consumer block, ~line 260-330: scp_signal(ready_ctr,0) +
// et_barrier() to rendezvous on a cold dispatch, then a wid-indexed
// scp_signal/scp_wait handshake per unit of work, flush_to_l2 on the
// writer side and evict_to_l2 on the reader side around every access). This
// K-split reimplementation follows that exact idiom (single-buffered,
// since K-split's per-row exchange is one value, not a multi-stage
// pipeline) rather than inventing a new combination of primitives -- unlike
// the light-prefetch lever (which invented a new use of an undocumented
// CSR with no working precedent anywhere in this codebase, and crashed
// real hardware twice despite passing the oracle both times), every piece
// used here has a demonstrated-working precedent in a real, shipped kernel
// for this exact purpose.
//
// use_ksplit_small_rows fires when rows_per_minion<=2 AND K_blocks>=64 --
// checked against llama32_1b's actual shapes: K/V-proj (M=512, K=2048,
// GQA num_kv_heads=8*head_dim=64) hits this exactly, where plain per-hart
// striping only engages 512 of 2048 harts (25% utilization). Q/O-proj
// (M=2048) and gate/up/down-proj/lm_head don't trigger it (M too large or
// K_blocks too small) and fall through unchanged to Combo A's tile-outer/
// simple-path branches below.
//
// Given the real-hardware lesson from the prefetch lever, this was ALSO
// given oracle coverage at the actual trigger shape before any board
// submission: the default test-backend-ops suite already includes
// test_mul_mat(Q8_0, F32, m=6, n=4096, k=5120) (tests/test-backend-ops.cpp
// ~line 8107), which satisfies rows_per_minion=1<=2 and K_blocks=160>=64 --
// this exercises use_ksplit_small_rows for real, not just the trivial
// m=16 k=256 case that never triggers it.
//
// Verified via llama.cpp's own test-backend-ops oracle against the real ET
// software-emulator backend before submission.
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

#include <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define SIMPLE_X2_ROWS             2

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

    // K-split "small rows" decision (see file header for the shape math)
    const int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    const int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    const int     is_hart1              = hart_id & 1;
    const int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    const int64_t k_half                = K_blocks / 2;
    const int     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);

    if (use_ksplit_small_rows) {
        /*
         * Each hart of the minion processes half the K dimension for the
         * SAME row, then the two halves are combined via a single-buffered
         * L2-SCP producer/consumer exchange (hart1 = producer, hart0 =
         * consumer), modeled on mul_mat_Q8_0_matrix_engine.c's real,
         * shipped scp_signal/scp_wait idiom.
         */
        const int64_t k_start = is_hart1 ? k_half : 0;
        const int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        const uint64_t    scp_base    = (uint64_t) local_minion * 192; /* 3 cache lines/minion: data, ready, consumed */
        volatile float *  l2scp_data  = (volatile float *) et_shire_l2scp_local(scp_base);
        volatile uint32_t * ready_ctr    = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 64);
        volatile uint32_t * consumed_ctr = (volatile uint32_t *) et_shire_l2scp_local(scp_base + 128);

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
                        // Producer: reset counters, rendezvous so the consumer
                        // sees the reset before its first scp_wait (avoids a
                        // stale-counter race on a cold dispatch).
                        scp_signal(ready_ctr, 0);
                        scp_signal(consumed_ctr, 0);
                        et_barrier(ET_BARRIER_MINION);

                        uint32_t wid = 0;
                        for (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                            const block_q8_0 * q_row = (const block_q8_0 *) (src0_ptr2 + m * nb01);
                            float partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                            // Wait for the consumer to finish reading the
                            // previous row before overwriting the shared slot.
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
                        // Consumer: rendezvous, then read the producer's
                        // counter resets before the first scp_wait.
                        et_barrier(ET_BARRIER_MINION);
                        evict_to_l2((const void *) ready_ctr, 1, 64);
                        WAIT_CACHEOPS;
                        evict_to_l2((const void *) consumed_ctr, 1, 64);
                        WAIT_CACHEOPS;

                        uint32_t wid = 0;
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
    } else if (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles - avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
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
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
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