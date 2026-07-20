#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// Q8_0 x F32 -> F32 MUL_MAT on the tensor engine: hart 1 dequantizes weights into L2 SCP, hart 0 runs TensorFMA32 (REUSE path reuses K-windows across N-tiles; ORIGINAL path is the no-reuse GEMV/single-tile case).

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define BLOCK_K QK8_0   // 32 elements per Q8_0 block
#define FMA_K   16      // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

// REUSE_MAX caps the L2-SCP C-scratch footprint (reuse limit); KWIN is the dequant-cache depth (K-blocks/window).
#ifndef REUSE_MAX
#define REUSE_MAX 32
#endif
#ifndef KWIN
#define KWIN    8       // K-blocks per dequant window (cache depth)
#endif

#define MACHINE_SLOTS (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE)  // 1024

#define CACHEOP_MAX 0
#define REP_RATE    0

// L1-SCP (Hart 0 only) layout: lines 0-15 A buffer 0, 16-31 B panel, 32-47 A buffer 1 (double-buffered activation).
#define A_L1_START 0    // L1 SCP lines  0..15 for A (activations), buffer 0
#define B_L1_START 16   // L1 SCP lines 16..31 for B (dequantized weights)
#define A_L1_ALT   32   // L1 SCP lines 32..47 for A (activations), buffer 1

// Single dequant panel: BLOCK_K k-lines x TILE_M m (FP32), [k][m] order: panel[k*TILE_M + m].
#define SCP_PANEL_SIZE   (BLOCK_K * TILE_M * (uint64_t)sizeof(float))  // 2048

// L2 SCP per-minion layout: REUSE path's double-buffered cache + C-scratch + sync counters; ORIGINAL path reuses the same region as two plain panels.
#define RU_BUF_BYTES     (KWIN * SCP_PANEL_SIZE)
#define RU_CACHE_BYTES   (2 * RU_BUF_BYTES)
#define RU_CSCRATCH_BYTES (REUSE_MAX * 16 * 64ULL)
#define SCP_READY_OFF    (RU_CACHE_BYTES + RU_CSCRATCH_BYTES)
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)

// Dequantizes one Q8_0 block (32 elems) of TILE_M weight rows into the TenB panel: value = d * qs[k].
static inline void __attribute__((always_inline))
dequant_q8_0_panel(float *panel, const char *src0_batch,
                   int64_t mb, int64_t kb_block, int64_t nb1_0) {
    static const int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448
    };
    static const int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 1, 2, 3, 4, 5, 6, 7
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, (%[sidx])    \n\t"
        "flw.ps    f2, (%[gidx])    \n\t"
        : [ms] "=&r"(old_mask)
        : [sidx] "r"(scatter_idx), [gidx] "r"(gather_idx)
        : "f1", "f2"
    );

    char *pbase = (char *) panel;
    for (int j = 0; j < TILE_M; ++j) {
        const block_q8_0 *blk =
            (const block_q8_0 *)(src0_batch + (mb + j) * nb1_0) + kb_block;
        uint32_t scale_raw = (uint32_t) blk->d;
        const int8_t *qs = blk->qs;
        char *col = pbase + j * 4;

        __asm__ volatile(
            "fbcx.ps     f3, %[sb]      \n\t"
            "fcvt.ps.f16 f3, f3         \n\t"

            "fgb.ps      f4, f2(%[qs0]) \n\t"
            "fcvt.ps.pw  f5, f4, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c0])  \n\t"

            "fgb.ps      f4, f2(%[qs8]) \n\t"
            "fcvt.ps.pw  f5, f4, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c8])  \n\t"

            "fgb.ps      f4, f2(%[qs16]) \n\t"
            "fcvt.ps.pw  f5, f4, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c16]) \n\t"

            "fgb.ps      f4, f2(%[qs24]) \n\t"
            "fcvt.ps.pw  f5, f4, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c24]) \n\t"
            :
            : [sb] "r"(scale_raw),
              [qs0] "r"(qs), [qs8] "r"(qs + 8),
              [qs16] "r"(qs + 16), [qs24] "r"(qs + 24),
              [c0] "r"(col), [c8] "r"(col + 8 * 64),
              [c16] "r"(col + 16 * 64), [c24] "r"(col + 24 * 64)
            : "f3", "f4", "f5", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

// Preloads one C row (f0..f31: row R -> f(2R)/f(2R+1)) with the bias, so every FMA at first_pass=0 yields C = bias + sum(A*B).
#define ET_PRELOAD_BIAS_ROW(R, FLO, FHI)                                       \
    do {                                                                       \
        const char *bp = bias_row_base + (int64_t)(R) * nbb1;                  \
        __asm__ volatile(                                                      \
            "flw.ps " FLO ", 0(%[p])   \n\t"                                   \
            "flw.ps " FHI ", 32(%[p])  \n\t"                                   \
            : : [p] "r"(bp) : FLO, FHI, "memory");                             \
    } while (0)

// Spill/seed the 16x16 FP32 C accumulator (row n -> f2n/f2n+1) to/from L2-SCP scratch (row n at byte offset n*64).
#define C_ROW_PAIR_ST(n0, n1, base)                                              \
    __asm__ volatile("fsw.ps f" #n0 ", (%0)\n\t fsw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "memory")
#define C_ROW_PAIR_LD(n0, n1, base)                                              \
    __asm__ volatile("flw.ps f" #n0 ", (%0)\n\t flw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "f" #n0, "f" #n1)

static inline void __attribute__((always_inline))
c_spill(char *s) {
    C_ROW_PAIR_ST(0,  1,  s + 0  * 64); C_ROW_PAIR_ST(2,  3,  s + 1  * 64);
    C_ROW_PAIR_ST(4,  5,  s + 2  * 64); C_ROW_PAIR_ST(6,  7,  s + 3  * 64);
    C_ROW_PAIR_ST(8,  9,  s + 4  * 64); C_ROW_PAIR_ST(10, 11, s + 5  * 64);
    C_ROW_PAIR_ST(12, 13, s + 6  * 64); C_ROW_PAIR_ST(14, 15, s + 7  * 64);
    C_ROW_PAIR_ST(16, 17, s + 8  * 64); C_ROW_PAIR_ST(18, 19, s + 9  * 64);
    C_ROW_PAIR_ST(20, 21, s + 10 * 64); C_ROW_PAIR_ST(22, 23, s + 11 * 64);
    C_ROW_PAIR_ST(24, 25, s + 12 * 64); C_ROW_PAIR_ST(26, 27, s + 13 * 64);
    C_ROW_PAIR_ST(28, 29, s + 14 * 64); C_ROW_PAIR_ST(30, 31, s + 15 * 64);
}

static inline void __attribute__((always_inline))
c_seed(char *s) {
    C_ROW_PAIR_LD(0,  1,  s + 0  * 64); C_ROW_PAIR_LD(2,  3,  s + 1  * 64);
    C_ROW_PAIR_LD(4,  5,  s + 2  * 64); C_ROW_PAIR_LD(6,  7,  s + 3  * 64);
    C_ROW_PAIR_LD(8,  9,  s + 4  * 64); C_ROW_PAIR_LD(10, 11, s + 5  * 64);
    C_ROW_PAIR_LD(12, 13, s + 6  * 64); C_ROW_PAIR_LD(14, 15, s + 7  * 64);
    C_ROW_PAIR_LD(16, 17, s + 8  * 64); C_ROW_PAIR_LD(18, 19, s + 9  * 64);
    C_ROW_PAIR_LD(20, 21, s + 10 * 64); C_ROW_PAIR_LD(22, 23, s + 11 * 64);
    C_ROW_PAIR_LD(24, 25, s + 12 * 64); C_ROW_PAIR_LD(26, 27, s + 13 * 64);
    C_ROW_PAIR_LD(28, 29, s + 14 * 64); C_ROW_PAIR_LD(30, 31, s + 15 * 64);
}

int entry_point(struct ggml_et_mm_q8_params *params, void *env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
    uint64_t local_minion = (hart_id >> 1) & 0x1F;

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    if ((M % TILE_M) != 0)  return 0;
    if ((K % BLOCK_K) != 0) return 0;

    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    const char *src0_base = (const char *) params->src0.data;
    const char *src1_base = (const char *) params->src1.data;
    char       *dst_base  = (char *) params->dst.data;

    // Fused residual bias (MUL_MAT+ADD); bias.data == NULL means unfused.
    const char   *bias_base = (const char *) params->bias.data;
    const int64_t nbb1 = params->bias.nb[1];
    const int64_t nbb2 = params->bias.nb[2];
    const int64_t nbb3 = params->bias.nb[3];

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = ne2_1 * ne3_1;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / BLOCK_K;        // number of Q8_0 blocks

    const uint64_t scp_base = local_minion * SCP_PER_MINION;
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // Reuse one prepared w_tile set across as many N-tiles as REUSE_MAX allows, spread evenly across n_groups.
    int64_t n_groups = (n_tiles + REUSE_MAX - 1) / REUSE_MAX;
    int64_t kwin_use = KWIN;
    if (n_tiles == 64) {
        // N=1024 (second most common shape): reduced reuse factor and K-window depth.
        n_groups = 16;  // ru_n = 4
        kwin_use = 2;
    }
    if (n_groups < 1) n_groups = 1;
    int64_t ru_n = (n_tiles + n_groups - 1) / n_groups;
    if (ru_n < 1) ru_n = 1;

    const int64_t units_pb   = m_tiles * n_groups;
    const int64_t base_units = units_pb * batch_count;

    // If work units don't fill the ~1024 core slots, split K across an intra-shire minion group and ring-reduce the partial C.
    int64_t k_splits = 1;
    {
        int64_t ks = 1;
        while (ks * 2 <= MINIONS_PER_SHIRE &&
               base_units * ks * 2 <= MACHINE_SLOTS &&
               (k_steps % (ks * 2)) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    const int64_t k_split         = local_minion % k_splits;
    const int64_t local_tile_idx  = local_minion / k_splits;
    const int64_t tiles_stride    = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;
    const int64_t my_start        = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;

    const int64_t k_steps_per_split = k_steps / k_splits;
    const int64_t k_start_block     = k_split * k_steps_per_split;

    // Reuse pays only when it groups >=2 N-tiles; otherwise fall back to the ORIGINAL path.
    const int reuse_ok = (ru_n >= 2);

    if (reuse_ok) {
        char *cache_buf[2] = {
            (char *) et_shire_l2scp_local(scp_base),
            (char *) et_shire_l2scp_local(scp_base + RU_BUF_BYTES),
        };
        char *cscratch = (char *) et_shire_l2scp_local(scp_base + RU_CACHE_BYTES);

        const int64_t k_end_block = k_start_block + k_steps_per_split;
        const int64_t n_windows   = (k_steps_per_split + kwin_use - 1) / kwin_use;

        // Hart 1: producer.
        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);
            // Barrier so the consumer sees this counter reset before its first scp_wait (avoids a stale-counter race on cold dispatch).
            et_barrier(ET_BARRIER_MINION);
            uint32_t wid = 0;

            for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
                const int64_t batch_idx    = unit / units_pb;
                const int64_t unit_in_b    = unit % units_pb;
                const int64_t mb_idx       = unit_in_b % m_tiles;

                const int64_t i3   = batch_idx / ne2_1;
                const int64_t i2   = batch_idx % ne2_1;
                const int64_t i2_0 = i2 / r2;
                const int64_t i3_0 = i3 / r3;

                const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
                const int64_t mb = mb_idx * TILE_M;

                for (int64_t kw = 0; kw < n_windows; ++kw) {
                    const int buf = wid & 1;
                    if (wid >= 2) scp_wait(consumed_ctr, wid - 1);

                    const int64_t kb0 = k_start_block + kw * kwin_use;
                    const int64_t kbn = (kb0 + kwin_use <= k_end_block) ? kwin_use : (k_end_block - kb0);

                    float *cf = (float *) cache_buf[buf];
                    for (int64_t i = 0; i < kbn; ++i) {
                        dequant_q8_0_panel(cf + i * (SCP_PANEL_SIZE / 4),
                                           src0_batch, mb, kb0 + i, nb1_0);
                    }
                    FENCE;
                    flush_to_l2_multi(cache_buf[buf], kbn * BLOCK_K, 64);
                    WAIT_CACHEOPS;

                    wid++;
                    scp_signal(ready_ctr, wid);
                }
            }
            FENCE;
            return 0;
        }

        // Hart 0: consumer.
        setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
        ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
        CLEAR_TENSOR_ERROR;
        // Rendezvous with the producer so its counter reset is visible before we read.
        et_barrier(ET_BARRIER_MINION);
        evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
        evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

        // First global minion id of this K-split group (ring-reduce chain base).
        const uint64_t group_base_global = get_minion_id() - (uint64_t) k_split;

        uint32_t wid = 0;
        for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
            const int64_t batch_idx = unit / units_pb;
            const int64_t unit_in_b = unit % units_pb;
            const int64_t g_idx     = unit_in_b / m_tiles;
            const int64_t mb_idx    = unit_in_b % m_tiles;

            const int64_t i3 = batch_idx / ne2_1;
            const int64_t i2 = batch_idx % ne2_1;

            const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
            char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;
            const char *bias_batch = bias_base ? bias_base + i3 * nbb3 + i2 * nbb2
                                               : (const char *) 0;

            const int64_t mb        = mb_idx * TILE_M;
            const int64_t nb_base_t = g_idx * ru_n;
            int64_t r_count = n_tiles - nb_base_t;
            if (r_count > ru_n) r_count = ru_n;

            for (int64_t kw = 0; kw < n_windows; ++kw) {
                const int buf = wid & 1;
                wid++;
                scp_wait(ready_ctr, wid);

                const int64_t kb0 = k_start_block + kw * kwin_use;
                const int64_t kbn = (kb0 + kwin_use <= k_end_block) ? kwin_use : (k_end_block - kb0);
                const int is_last = (kw == n_windows - 1);
                float *cf = (float *) cache_buf[buf];

                for (int64_t r = 0; r < r_count; ++r) {
                    const int64_t nb = (nb_base_t + r) * TILE_N;
                    const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
                    const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

                    char *cs = cscratch + r * (16 * 64);

                    // Fused bias is seeded ONLY by k_split 0, so the ring-reduce sums it exactly once.
                    int first;
                    if (kw == 0) {
                        if (bias_batch && k_split == 0) {
                            const char *bias_row_base =
                                bias_batch + nb * nbb1 + mb * (int64_t) sizeof(float);
                            unsigned long _bmask;
                            __asm__ volatile("mova.x.m %[ms]\n\t mov.m.x m0, x0, 0xFF\n\t"
                                             : [ms] "=&r"(_bmask) : :);
                            if (n_cur >  0) ET_PRELOAD_BIAS_ROW(0,  "f0",  "f1");
                            if (n_cur >  1) ET_PRELOAD_BIAS_ROW(1,  "f2",  "f3");
                            if (n_cur >  2) ET_PRELOAD_BIAS_ROW(2,  "f4",  "f5");
                            if (n_cur >  3) ET_PRELOAD_BIAS_ROW(3,  "f6",  "f7");
                            if (n_cur >  4) ET_PRELOAD_BIAS_ROW(4,  "f8",  "f9");
                            if (n_cur >  5) ET_PRELOAD_BIAS_ROW(5,  "f10", "f11");
                            if (n_cur >  6) ET_PRELOAD_BIAS_ROW(6,  "f12", "f13");
                            if (n_cur >  7) ET_PRELOAD_BIAS_ROW(7,  "f14", "f15");
                            if (n_cur >  8) ET_PRELOAD_BIAS_ROW(8,  "f16", "f17");
                            if (n_cur >  9) ET_PRELOAD_BIAS_ROW(9,  "f18", "f19");
                            if (n_cur > 10) ET_PRELOAD_BIAS_ROW(10, "f20", "f21");
                            if (n_cur > 11) ET_PRELOAD_BIAS_ROW(11, "f22", "f23");
                            if (n_cur > 12) ET_PRELOAD_BIAS_ROW(12, "f24", "f25");
                            if (n_cur > 13) ET_PRELOAD_BIAS_ROW(13, "f26", "f27");
                            if (n_cur > 14) ET_PRELOAD_BIAS_ROW(14, "f28", "f29");
                            if (n_cur > 15) ET_PRELOAD_BIAS_ROW(15, "f30", "f31");
                            // n_cur==4 runs AROWS=4 (5 rows); the padded 5th C row (f8,f9) must be zero under first_pass=0.
                            if (n_cur == 4) {
                                static const float __attribute__((aligned(64))) zline[8] = {0};
                                __asm__ volatile("flw.ps f8, 0(%[z])\n\t flw.ps f9, 0(%[z])\n\t"
                                                 : : [z] "r"(zline) : "f8", "f9", "memory");
                            }
                            __asm__ volatile("mova.m.x %[ms]\n\t" : : [ms] "r"(_bmask) :);
                            first = 0;
                        } else {
                            first = 1;   // zero-init (unfused, or non-zero k_split)
                        }
                    } else {
                        c_seed(cs);      // carry partial from prior windows
                        first = 0;
                    }

                    if (n_cur == 4) {
                        // Errata-D: present AROWS=4 by zero-padding row 4 in both double-buffered A regions.
                        static const float __attribute__((aligned(64))) zero_line[16] = {0};
                        tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t) zero_line, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                        tensor_load(false, false, A_L1_ALT + 4, TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t) zero_line, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                    }

                    // Software-pipelined: each step is one A-tile+FMA, with the next A-tile prefetched into the alternate L1 buffer.
                    const int64_t nsteps = kbn * 2;
#define A_TILE_ADDR(st)                                                                  \
    ((uint64_t)(src1_batch + nb * nb1_1 +                                                \
        (((kb0 + (st) / 2) * BLOCK_K) + ((st) % 2) * FMA_K) * (int64_t) sizeof(float)))
                    tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                                A_TILE_ADDR(0), 0, n_cur - 1, (uint64_t) nb1_1, 0);
                    for (int64_t s = 0; s < nsteps; ++s) {
                        const uint64_t a_cur = (s & 1) ? A_L1_ALT : A_L1_START;

                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        if (s + 1 < nsteps) {
                            const uint64_t a_nxt = ((s + 1) & 1) ? A_L1_ALT : A_L1_START;
                            tensor_load(false, false, a_nxt, TENSOR_LOAD_PLAIN, 0,
                                        A_TILE_ADDR(s + 1), 0, n_cur - 1, (uint64_t) nb1_1, 0);
                        }

                        const int64_t i = s / 2, half = s % 2;
                        tensor_load_setup_b(
                            false,
                            (uint64_t)(cf + i * (SCP_PANEL_SIZE / 4) + half * FMA_K * TILE_M),
                            FMA_K - 1, 64, 1);

                        tensor_fma(
                            false, 3, arows_fma, FMA_K - 1, 0,
                            false, false, false, true,
                            B_L1_START, a_cur, TENSOR_FMA_OP_FP32, first);
                        tensor_wait(TENSOR_FMA_WAIT);
                        first = 0;
                    }
#undef A_TILE_ADDR

                    if (is_last) {
                        // Ring-reduce the K-split partials: chain g0->g1->...->g(last), each recv+FADDs then forwards; last minion stores.
                        if (k_splits > 1) {
                            const uint64_t num_regs = (uint64_t) n_cur * 2;
                            if (k_split > 0) {
                                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, num_regs,
                                                   group_base_global + (uint64_t)(k_split - 1));
                                tensor_wait(TENSOR_REDUCE_WAIT);
                            }
                            if (k_split < k_splits - 1) {
                                tensor_reduce_send(0, num_regs,
                                                   group_base_global + (uint64_t)(k_split + 1));
                                tensor_wait(TENSOR_REDUCE_WAIT);
                            }
                        }
                        if (k_split == k_splits - 1) {
                            tensor_store(
                                0, 0, 3, n_cur - 1,
                                (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                                0, (uint64_t) nb1_d);
                            tensor_wait(TENSOR_STORE_WAIT);
                        }
                    } else {
                        c_spill(cs);
                    }
                }
                scp_signal(consumed_ctr, wid);
            }
        }
        FENCE;
        return 0;
    }

    // ORIGINAL path: one output tile at a time (N % TILE_N != 0), no reuse.
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;
    float *scp_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base),
        (float *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };

    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);
        // See REUSE path: barrier so the consumer observes the reset before it reads.
        et_barrier(ET_BARRIER_MINION);
        uint32_t chunk_id = 0;

        for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
            const int64_t tiles_per_batch = m_tiles * n_tiles;
            const int64_t batch_idx       = tile / tiles_per_batch;
            const int64_t tile_in_batch   = tile % tiles_per_batch;
            const int64_t mb_idx          = tile_in_batch % m_tiles;

            const int64_t i3   = batch_idx / ne2_1;
            const int64_t i2   = batch_idx % ne2_1;
            const int64_t i2_0 = i2 / r2;
            const int64_t i3_0 = i3 / r3;

            const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            const int64_t mb = mb_idx * TILE_M;

            for (int64_t kb = 0; kb < k_steps; ++kb) {
                int buf = chunk_id & 1;
                if (chunk_id >= 2) scp_wait(consumed_ctr, chunk_id - 1);

                dequant_q8_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2_multi(scp_panel[buf], BLOCK_K, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }
        FENCE;
        return 0;
    }

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;
    et_barrier(ET_BARRIER_MINION);
    evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

    uint32_t chunk_id = 0;
    for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx       = tile / tiles_per_batch;
        const int64_t tile_in_batch   = tile % tiles_per_batch;
        const int64_t nb_idx          = tile_in_batch / m_tiles;
        const int64_t mb_idx          = tile_in_batch % m_tiles;

        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;

        const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
        const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        if (n_cur == 4) {
            static const float __attribute__((aligned(64))) zero_line[16] = {0};
            tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) zero_line, 0, 0, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
        }

        int first = 1;
        for (int64_t kb = 0; kb < k_steps; ++kb) {
            int buf = chunk_id & 1;
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            for (int half = 0; half < 2; ++half) {
                const int64_t k_elem = kb * BLOCK_K + half * FMA_K;
                tensor_load(
                    false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                    (uint64_t)(src1_batch + nb * nb1_1 + k_elem * (int64_t) sizeof(float)),
                    0, n_cur - 1, (uint64_t) nb1_1, 0);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                tensor_load_setup_b(
                    false,
                    (uint64_t)(scp_panel[buf] + (int64_t) half * FMA_K * TILE_M),
                    FMA_K - 1, 64, 1);

                tensor_fma(
                    false, 3, arows_fma, FMA_K - 1, 0,
                    false, false, false, true,
                    B_L1_START, A_L1_START, TENSOR_FMA_OP_FP32, first);
                tensor_wait(TENSOR_FMA_WAIT);
                first = 0;
            }

            scp_signal(consumed_ctr, chunk_id);
        }

        tensor_store(
            0, 0, 3, n_cur - 1,
            (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
            0, (uint64_t) nb1_d);
        tensor_wait(TENSOR_STORE_WAIT);
    }

    FENCE;
    return 0;
}
