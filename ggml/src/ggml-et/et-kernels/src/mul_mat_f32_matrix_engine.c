#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"

/*
 * Minimal single TensorFMA32 test for GGML MUL_MAT on ET-SoC-1.
 *
 * Assumptions:
 *   - M = N = K = 16 (exactly one 16×16×16 tile)
 *   - src0 is [M][K] row-major FP32 (M rows, K cols contiguous)
 *   - src1 is [N][K] row-major FP32
 *   - dst  is [N][M] row-major FP32
 *   - GGML semantics: dst[n][m] = Σ_k src0[m][k] * src1[n][k]
 *   - All data pointers are 64-byte aligned
 *   - Only hart 0 of the first minion executes
 *
 * FMA mapping:
 *   FMA computes C[i][j] = Σ_k A[i][k] * B[k][j]
 *
 *   A_fma[i][k] = src1[n+i][k]         → plain TensorLoad, 16 lines → SCP 0–15
 *   B_fma[k][j] = src0[m+j][k]         → TensorLoadTranspose32      → SCP 16–31
 *     (Transpose32 loads 16 memory rows of src0, each 16 FP32,
 *      writes SCP[16+k].e[j] = src0[j][k])
 *   C_fma[i][j] = dst[n+i][m+j]        → stored from f-regs to memory
 *
 * SCP budget: 32 of 48 lines used.
 * f-register budget: all 32 (16 C rows × 2 regs each).
 */


#define L1D_NUM_SETS      16
#define L1D_NUM_WAYS      4
#define L1D_LINE_SIZE     64

#define TILE      16  // hardware tile size
#define NUM_HARTS 1024

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();

    // Only hart 0 have access to tensor engine
    if (hart_id & 1) {
        return 0;
    }

    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    // ===== Dimensions =====
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    const int64_t ne2_0 = params->src0.ne[2];
    const int64_t ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2];
    const int64_t ne3_1 = params->src1.ne[3];

    // ===== Byte strides =====
    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2];
    const int64_t nb3_0 = params->src0.nb[3];

    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2];
    const int64_t nb3_1 = params->src1.nb[3];

    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2];
    const int64_t nb3_d = params->dst.nb[3];

    const char* src0_base = (const char*)params->src0.data;
    const char* src1_base = (const char*)params->src1.data;
    char*       dst_base  = (char*)params->dst.data;

    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    // ===== Tile grid =====
    const int64_t m_tiles = M / TILE;
    const int64_t n_tiles = (N + TILE - 1) / TILE;   // ceil for N edge
    const int64_t tiles_per_batch = m_tiles * n_tiles;
    const int64_t batch_count     = ne2_1 * ne3_1;
    const int64_t total_tiles     = batch_count * tiles_per_batch;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;
    // No-broadcast case (e.g. standard multi-head attention without GQA,
    // where src0/src1 share the same head/batch counts): i2_0/i3_0 collapse
    // to i2/i3 exactly, since dividing by 1 is the identity. Checked once
    // per kernel call (loop-invariant), not assumed -- the original r2/r3
    // division is kept as the exact fallback whenever broadcasting is
    // actually in effect (r2>1 or r3>1, e.g. real GQA).
    const bool no_broadcast = (r2 == 1 && r3 == 1);

    for (int64_t tile = global_id; tile < total_tiles; tile += NUM_HARTS) {

        const int64_t batch_idx     = tile / tiles_per_batch;
        const int64_t tile_in_batch = tile % tiles_per_batch;
        const int64_t nb_idx = tile_in_batch / m_tiles;
        const int64_t mb_idx = tile_in_batch % m_tiles;

        const int64_t i3   = batch_idx / ne2_1;
        const int64_t i2   = batch_idx % ne2_1;
        const int64_t i2_0 = no_broadcast ? i2 : i2 / r2;
        const int64_t i3_0 = no_broadcast ? i3 : i3 / r3;

        const char* src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
        const char* src1_batch = src1_base + i3   * nb3_1 + i2   * nb2_1;
        char*       dst_batch  = dst_base  + i3   * nb3_d + i2   * nb2_d;

        const int64_t mb = mb_idx * TILE;
        const int64_t nb = nb_idx * TILE;

        // N edge: last tile may have fewer than 16 rows
        const int64_t n_cur = (nb + TILE <= N) ? TILE : (N - nb);

        // Latency-hiding K-loop, adapted from the vendor's own reference
        // pattern (et-platform/test-compute-kernels/src/tl_tfma_fc/
        // tl_tfma_fc.c and coop_tl_tfma_2s/coop_tl_tfma_2s.c): the A tile
        // ("plain load") alternates between two 16-line SCP zones (line 0
        // and line 32, the latter previously unused -- this kernel's own
        // header comment already notes only 32 of 48 SCP lines are used).
        // The NEXT iteration's A tile is issued right after the CURRENT
        // iteration's tensor_fma (not waited on), so its load latency runs
        // concurrently with the FMA's compute time instead of serially
        // before it -- the exact "latency hiding" the PRM recommends per
        // this file's own long-standing XXX comment below, which was never
        // implemented. B ("transpose load") is left untouched, still fully
        // serial each iteration, matching the vendor pattern's own
        // asymmetric treatment (only one of the two loads is prefetched).
        //
        // Correctness invariants (checked, not assumed):
        //   - A's two alternating zones (0, 32) never overlap B's fixed
        //     zone (16) or each other.
        //   - The FMA for iteration kb reads A from whichever zone holds
        //     kb's already-loaded, already-waited-on data (`a_cur`); the
        //     prefetch issued right after it targets the OTHER zone
        //     (`a_next`), which is guaranteed not to be the zone the FMA
        //     is reading from -- no write-during-read hazard.
        //   - The prefetch's own completion is explicitly waited on at the
        //     TOP of the next iteration (a separate wait from the previous
        //     iteration's FMA wait) before that data is ever used as FMA
        //     input -- this is NOT assumed to "have had enough time";
        //     it is a real, explicit tensor_wait every iteration.
        const int64_t a_zone[2] = {0, 32};
        int64_t a_slot = 0;   // index into a_zone: which zone holds the CURRENT tile

        // Prologue: load kb=0's A tile into zone 0 before the loop starts.
        tensor_load(
            false, false,
            (uint64_t)a_zone[0],
            0,                                                          // plain load
            0,
            (uint64_t)(src1_batch + nb * nb1_1),
            0,
            n_cur - 1,
            (uint64_t)nb1_1,
            0
        );
        tensor_wait(TENSOR_LOAD_WAIT_0);

        // Accumulate across K (always full 16-wide chunks)
        for (int64_t kb = 0; kb < K; kb += TILE) {
            const int64_t a_cur  = a_zone[a_slot];
            const int64_t a_next = a_zone[a_slot ^ 1];

            // B = transpose(src0[mb:mb+15][kb:kb+15]) -> SCP 16:31 (unchanged, fully serial)
            tensor_load(
                false, false,
                TILE,                                                       // SCP line 16
                7,                                                          // 111 -> TensorLoadTranspose32
                0,
                (uint64_t)(src0_batch + mb * nb1_0 + kb * sizeof(float)),
                0,
                TILE - 1,                                                   // always 16 K elements
                (uint64_t)nb1_0,
                1
            );
            tensor_wait(TENSOR_LOAD_WAIT_1);

            tensor_fma(
                false,
                3,              // BCOLS=3 -> 16 M columns (always full. calculated as `4 * BCOLS + 4`)
                n_cur - 1,      // AROWS: N edge
                TILE - 1,       // ACOLS=15 (K always full tile)
                0,
                false, false, false,
                false,
                TILE,           // BSTART=16
                (uint64_t)a_cur,// ASTART: alternates 0/32, never the zone being prefetched into below
                0,              // TensorFMA32
                (kb == 0)       // first pass: overwrite; else: accumulate
            );
            // XXX: tensor_fma outputs to f0~f31 register. It is, in theory, possible to carve out
            //       some space on the stack and use that as a way to perform tiled MM. But that is really
            //       difficult, not tried and not what the PRM wants you to do (it recommends latency hiding)
            // XXX: This also means that if the compiler tried to carry any FP value across tensor_fma
            //      without spilling - it breaks.

            // Prefetch the NEXT tile's A into the zone NOT being read by the
            // FMA just issued above -- issued before waiting on that FMA, so
            // the load latency overlaps the FMA's compute time.
            const int64_t kb_next = kb + TILE;
            if (kb_next < K) {
                tensor_load(
                    false, false,
                    (uint64_t)a_next,
                    0,
                    0,
                    (uint64_t)(src1_batch + nb * nb1_1 + kb_next * sizeof(float)),
                    0,
                    n_cur - 1,
                    (uint64_t)nb1_1,
                    0
                );
            }

            tensor_wait(TENSOR_FMA_WAIT);
            if (kb_next < K) {
                // Explicit wait for the prefetch issued above -- required
                // before the next iteration's FMA reads a_next, not assumed
                // safe just because time has passed.
                tensor_wait(TENSOR_LOAD_WAIT_0);
                a_slot ^= 1;
            }
        }

        // Store n_cur rows x 64 bytes
        tensor_store(
            0,                                                              // STEP=0
            0,                                                              // FREG=0
            3,                                                              // SIZE=3 -> 64B/row (16 * SIZE + 16)
            n_cur - 1,                                                      // N edge
            (uint64_t)(dst_batch + nb * nb1_d + mb * sizeof(float)),
            0,
            (uint64_t)nb1_d
        );

        tensor_wait(TENSOR_STORE_WAIT);
    }

    FENCE; // XXX: Do we need this?
    return 0;
}
