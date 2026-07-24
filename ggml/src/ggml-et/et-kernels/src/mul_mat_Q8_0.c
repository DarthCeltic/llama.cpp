//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// COMBO E -- x2-row B-load reuse (Rehan's win) COMBINED with batched-8
// fp16 scale-convert (this session's own lever), via a different
// composition than Combo C's trade-off.
//
// Combo C traded away x2-row-batching to get batched-8 scale-convert
// because Rehan's design defers the horizontal reduction to once-per-row
// (persistent f20/f21 vector accumulators across all blocks), which needs
// an arbitrary-lane (0..7) vector-to-scalar extract to apply a batch of 8
// DIFFERENT per-block scales to that deferred accumulator -- no verified
// ISA primitive for that exists in this codebase (only a FIXED lane-4
// extract, used after a pairwise reduction, is ever seen).
//
// This combo sidesteps that blocker entirely: instead of extracting from
// an ISA vector register, the batched-8 scale-convert already writes its
// result to a PLAIN C FLOAT ARRAY in memory (scales[8], via fsw.ps) -- a
// completely ordinary compiler-visible float array, not a vector lane.
// Reading scales[i] is then just a normal C array index, no ISA extract
// needed at all. This means batched-8 scale-convert composes cleanly with
// x2-row B-reuse PROVIDED the horizontal reduction happens per-block
// (immediately, for each row) rather than being deferred to end-of-row --
// i.e. this combo pays Rehan's "reduce once at the end" efficiency to gain
// the batched scale-convert efficiency, the mirror-image trade of Combo C
// (which paid away x2's B-reuse to gain the same thing). Whether this
// nets out positive, given each has different real costs, is exactly the
// kind of question real board data (not prediction) should answer.
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

#define STRIDE_M       2048 /* 32 shires x 32 minions x 2 harts */
#define TILE_KB        256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define SIMPLE_X2_ROWS 2

// Compute two rows' raw (UNSCALED) dot products for one Q8_0 block, reusing
// the same loaded B-values across both rows (Rehan's x2 win). Returns the
// two 8-wide partial sums horizontally reduced to scalars immediately (not
// deferred) so a plain C float scale can be applied by the caller.
static inline void q8_dot_block_x2_raw(const block_q8_0 * blk0, const block_q8_0 * blk1, const float * b_ptr,
                                       float * out0, float * out1) {
    __asm__ volatile(
        "fbci.pi     f10, 0\n"
        "fbci.pi     f11, 0\n"

        "flw.ps      f12, %[bv0]\n"
        "fgb.ps      f16, f31(%[r0ap0])\n"
        "fcvt.ps.pw  f16, f16\n"
        "fmadd.ps    f10, f16, f12, f10\n"
        "fgb.ps      f17, f31(%[r1ap0])\n"
        "fcvt.ps.pw  f17, f17\n"
        "fmadd.ps    f11, f17, f12, f11\n"

        "flw.ps      f13, %[bv1]\n"
        "fgb.ps      f16, f31(%[r0ap1])\n"
        "fcvt.ps.pw  f16, f16\n"
        "fmadd.ps    f10, f16, f13, f10\n"
        "fgb.ps      f17, f31(%[r1ap1])\n"
        "fcvt.ps.pw  f17, f17\n"
        "fmadd.ps    f11, f17, f13, f11\n"

        "flw.ps      f14, %[bv2]\n"
        "fgb.ps      f16, f31(%[r0ap2])\n"
        "fcvt.ps.pw  f16, f16\n"
        "fmadd.ps    f10, f16, f14, f10\n"
        "fgb.ps      f17, f31(%[r1ap2])\n"
        "fcvt.ps.pw  f17, f17\n"
        "fmadd.ps    f11, f17, f14, f11\n"

        "flw.ps      f15, %[bv3]\n"
        "fgb.ps      f16, f31(%[r0ap3])\n"
        "fcvt.ps.pw  f16, f16\n"
        "fmadd.ps    f10, f16, f15, f10\n"
        "fgb.ps      f17, f31(%[r1ap3])\n"
        "fcvt.ps.pw  f17, f17\n"
        "fmadd.ps    f11, f17, f15, f11\n"

        "fswizz.ps   f1, f10, 0xB1\n"
        "fadd.ps     f2, f10, f1, rne\n"
        "fswizz.ps   f3, f2, 0x4E\n"
        "fadd.ps     f4, f2, f3, rne\n"
        "fmvz.x.ps   t0, f4, 4\n"
        "fbcx.ps     f5, t0\n"
        "fadd.ps     %[o0], f4, f5, rne\n"

        "fswizz.ps   f1, f11, 0xB1\n"
        "fadd.ps     f2, f11, f1, rne\n"
        "fswizz.ps   f3, f2, 0x4E\n"
        "fadd.ps     f4, f2, f3, rne\n"
        "fmvz.x.ps   t0, f4, 4\n"
        "fbcx.ps     f5, t0\n"
        "fadd.ps     %[o1], f4, f5, rne\n"
        : [o0] "=f"(*out0), [o1] "=f"(*out1)
        : [r0ap0] "r"(&blk0->qs[0]), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]), [r0ap3] "r"(&blk0->qs[24]),
          [r1ap0] "r"(&blk1->qs[0]), [r1ap1] "r"(&blk1->qs[8]), [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]),
          [bv0] "m"(*(const float (*)[8]) & b_ptr[0]), [bv1] "m"(*(const float (*)[8]) & b_ptr[8]),
          [bv2] "m"(*(const float (*)[8]) & b_ptr[16]), [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
        : "t0", "f1", "f2", "f3", "f4", "f5", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
}

// x2-row dot with batched-8 scale-convert. Mask assumed already 0xFF
// (caller wraps with q8_dot_begin/end).
static inline void q8_dot_compute_x2_scaleunpack(const block_q8_0 * q_row0, const block_q8_0 * q_row1,
                                                  const float * b_col_base, int64_t K_blocks, float * out0,
                                                  float * out1) {
    static const int32_t gather_bytes[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    static const int32_t f16_pattern[8]  = { 0, 2, 4, 6, 8, 10, 12, 14 };

    __asm__ volatile("flw.ps f31, %[g]\n" : : [g] "m"(*(const int32_t (*)[8]) gather_bytes) : "f31");

    float         row_sum0   = 0.0f;
    float         row_sum1   = 0.0f;
    const int64_t groups     = K_blocks / 8;
    const int64_t tail_start = groups * 8;

    for (int64_t g = 0; g < groups; g++) {
        // Batch-convert 8 blocks' deltas for row0 and row1 into plain C
        // float arrays -- ordinary memory, no ISA lane extract needed.
        uint16_t deltas0[8] __attribute__((aligned(16)));
        uint16_t deltas1[8] __attribute__((aligned(16)));
        for (int i = 0; i < 8; i++) {
            deltas0[i] = q_row0[g * 8 + i].d;
            deltas1[i] = q_row1[g * 8 + i].d;
        }
        float scales0[8] __attribute__((aligned(32)));
        float scales1[8] __attribute__((aligned(32)));
        __asm__ volatile(
            "flw.ps      f21, %[pat]\n"
            "fgh.ps      f20, f21(%[d0])\n"
            "fcvt.ps.f16 f20, f20\n"
            "fsw.ps      f20, %[out0]\n"
            "fgh.ps      f20, f21(%[d1])\n"
            "fcvt.ps.f16 f20, f20\n"
            "fsw.ps      f20, %[out1]\n"
            : [out0] "=m"(*(float (*)[8]) scales0), [out1] "=m"(*(float (*)[8]) scales1)
            : [d0] "r"(deltas0), [d1] "r"(deltas1), [pat] "m"(*(const int32_t (*)[8]) f16_pattern)
            : "f20", "f21");

        for (int i = 0; i < 8; i++) {
            const block_q8_0 * blk0  = q_row0 + g * 8 + i;
            const block_q8_0 * blk1  = q_row1 + g * 8 + i;
            const float *      b_ptr = b_col_base + ((int64_t) (g * 8 + i) << 5);

            float raw0, raw1;
            q8_dot_block_x2_raw(blk0, blk1, b_ptr, &raw0, &raw1);
            row_sum0 += raw0 * scales0[i];
            row_sum1 += raw1 * scales1[i];
        }
    }

    for (int64_t kb = tail_start; kb < K_blocks; kb++) {
        row_sum0 += compute_row_dot_q8_0(q_row0 + kb, b_col_base + (kb << 5), 1);
        row_sum1 += compute_row_dot_q8_0(q_row1 + kb, b_col_base + (kb << 5), 1);
    }

    *out0 = row_sum0;
    *out1 = row_sum1;
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
         * Simple path: x2-row B-reuse when nb01 is 32-byte aligned, WITH
         * batched-8 scale-convert (see file header). Falls back to a
         * single-row batched-8 path for the odd leftover row / non-aligned
         * case.
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
                                q8_dot_compute_x2_scaleunpack(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

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