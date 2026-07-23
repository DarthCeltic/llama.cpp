//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// COMBO C (v2, prefetch removed) -- a deliberately DIFFERENT lever
// combination from Combo A/B, not a strict superset. Composes two
// independently-proven techniques:
//
//   1. Mask-hoist (q8_dot_begin/end, shared with Combo A/B / Rehan's design)
//   2. Batched-8 fp16 scale-convert (THIS session's own standalone lever,
//      already oracle-verified earlier and staged as PR#168's prior content
//      -- packs 8 consecutive blocks' deltas into a scratch buffer and does
//      ONE fgh.ps+fcvt.ps.f16 for all 8, instead of 8 separate
//      fbcx.ps+fcvt.ps.f16 calls)
//
// v1 of this combo also included the light-prefetch lever, but that lever
// was found to crash on a real full-model board run (undrained prefetch_va
// CSR -- see Combo B v2's header for the full diagnosis). Rather than stack
// an unconfirmed prefetch fix INTO this combo too, prefetch is isolated to
// its own single-variable test on PR#169 (Combo B v2) and dropped entirely
// here -- this combo now tests ONLY the batched-8 scale-convert lever in
// isolation, on top of a base already proven clean on real hardware
// (Combo A, no prefetch, no batched scale-convert).
//
// NOT included here: Rehan's x2-row-batching and deferred-whole-row vector
// accumulation (f20 persists across all blocks, reduced once at the end).
// Investigated porting batched-8 scale-convert INTO that deferred-
// accumulation design and found it requires extracting an arbitrary lane
// (0..7) from a converted-scales vector register into a scalar for a
// per-lane broadcast -- every lane-extract primitive actually observed in
// this codebase (fmvz.x.ps) only pulls a FIXED lane (lane 4, after a
// pairwise reduction has consolidated the vector), not an arbitrary index.
// Inventing an 8-way arbitrary-lane extract with no reference to verify
// against is exactly the kind of guess that produced this session's earlier
// board-proven regression (cache-line-owned stores); not worth the risk.
// Batched-8 scale-convert composes cleanly instead with a per-block scalar
// reduction (this file), trading away x2-row B-load reuse for the batched
// scale win -- a genuinely different, non-overlapping bet from Combo A/B,
// so real board data (not prediction) shows which strategy is worth more
// for THIS lever combination. See corpus entry
// hackathon_llama32_1b_ABI_COMPAT_and_THREE_COMBO_PLAN_20260723.
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

#define STRIDE_M 2048 /* 32 shires x 32 minions x 2 harts */
#define TILE_KB  256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */

// Row dot product with batched-8 fp16 scale-unpack. Mask is assumed already
// set to 0xFF by the caller's q8_dot_begin (mask-hoist happens ONCE per n,
// not once per row/block) -- this function does not touch the mask register.
static inline float compute_row_dot_q8_0_scaleunpack_nomask(const block_q8_0 * q_row,
                                                             const float *      b_col_base,
                                                             int64_t            K_blocks) {
    static const int32_t gather_bytes[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    static const int32_t f16_pattern[8]  = { 0, 2, 4, 6, 8, 10, 12, 14 };

    float         row_sum    = 0.0f;
    const int64_t groups     = K_blocks / 8;
    const int64_t tail_start = groups * 8;

    for (int64_t g = 0; g < groups; g++) {
        uint16_t deltas[8] __attribute__((aligned(16)));
        for (int i = 0; i < 8; i++) {
            deltas[i] = q_row[g * 8 + i].d;
        }
        float scales[8] __attribute__((aligned(32)));
        __asm__ volatile(
            "flw.ps      f21, %[pat]\n"
            "fgh.ps      f20, f21(%[d])\n"
            "fcvt.ps.f16 f20, f20\n"
            "fsw.ps      f20, %[out]\n"
            : [out] "=m"(*(float (*)[8]) scales)
            : [d] "r"(deltas), [pat] "m"(*(const int32_t (*)[8]) f16_pattern)
            : "f20", "f21");

        for (int i = 0; i < 8; i++) {
            const block_q8_0 * blk = q_row + g * 8 + i;
            const float *      b   = b_col_base + ((int64_t) (g * 8 + i) << 5);

            float acc;
            __asm__ volatile(
                "flw.ps     f31, %[gb]\n"
                "fbci.pi    f10, 0\n"
                "flw.ps     f12, 0(%[b])\n"
                "fgb.ps     f11, f31(%[a])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps   f10, f11, f12, f10\n"
                "flw.ps     f12, 32(%[b])\n"
                "fgb.ps     f11, f31(%[a1])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps   f10, f11, f12, f10\n"
                "flw.ps     f12, 64(%[b])\n"
                "fgb.ps     f11, f31(%[a2])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps   f10, f11, f12, f10\n"
                "flw.ps     f12, 96(%[b])\n"
                "fgb.ps     f11, f31(%[a3])\n"
                "fcvt.ps.pw f11, f11\n"
                "fmadd.ps   f10, f11, f12, f10\n"
                "fswizz.ps  f1, f10, 0xB1\n"
                "fadd.ps    f2, f10, f1, rne\n"
                "fswizz.ps  f3, f2, 0x4E\n"
                "fadd.ps    f4, f2, f3, rne\n"
                "fmvz.x.ps  t0, f4, 4\n"
                "fbcx.ps    f5, t0\n"
                "fadd.ps    %[out], f4, f5, rne\n"
                : [out] "=f"(acc)
                : [gb] "m"(*(const int32_t (*)[8]) gather_bytes), [a] "r"(blk->qs), [a1] "r"(blk->qs + 8),
                  [a2] "r"(blk->qs + 16), [a3] "r"(blk->qs + 24), [b] "r"(b)
                : "memory", "t0", "f1", "f2", "f3", "f4", "f5", "f10", "f11", "f12", "f31");
            row_sum += acc * scales[i];
        }
    }

    for (int64_t kb = tail_start; kb < K_blocks; kb++) {
        row_sum += compute_row_dot_q8_0(q_row + kb, b_col_base + (kb << 5), 1);
    }

    return row_sum;
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
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    if (K_blocks > TILE_KB) {
        /* Tile-outer path (large K, e.g. other regression models) --
         * unchanged from Combo A/B, uses only confirmed-available
         * primitives, no scale-convert or prefetch changes here. */
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
         * Simple path: mask-hoisted (once per n, via q8_dot_begin/end),
         * per-row prefetch of the next row, batched-8 scale-convert per
         * row. Uniform per-hart striping (no x2-row pairing -- traded
         * away in this combo for the batched scale-convert, see header).
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

                    for (int64_t m = (int64_t) hart_id; m < M; m += STRIDE_M) {
                        const block_q8_0 * q_row = (const block_q8_0 *) (src0_ptr2 + m * nb01);
                        float sum = compute_row_dot_q8_0_scaleunpack_nomask(q_row, b_col_base, K_blocks);

                        float * dst_entry = (float *) (dst_ptr2 + n * nbd1 + m * sizeof(float));
                        if (bias_n) {
                            sum += bias_n[m];
                        }
                        atomic_store_f32((volatile float *) dst_entry, sum);
                    }

                    q8_dot_end(&q8_state);
                }
            }
        }
    }

    return 0;
}