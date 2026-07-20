//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#define BLOCK_OPS_H

#include <stdint.h>
#include "math_fp.h"
#include "quants.h"

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
inline void __attribute__((always_inline)) excl_mode(uint64_t val)
{
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
static inline float compute_block_dot_product_q8_0(const block_q8_0* a_block, const float* b_col_start) {

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3; // chunk * 8

        __asm__ volatile(
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12"
        );
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__ (
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"             // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"              // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (final_sum)
        :: "t0", "f10", "f2", "f3", "f4", "f5"
    );

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

// Same as compute_block_dot_product_q8_0, but skips the per-call vector-mask
// save/set/restore. The mask CSR is hardware-persistent state untouched by
// compiler-generated code between explicit asm blocks (unlike an ordinary
// vector register the allocator could reuse) -- so a caller that invokes
// this in a tight loop (e.g. across all K_blocks of one row) can set the
// mask to 0xFF ONCE before the loop and restore it ONCE after, instead of
// paying the save/set/restore CSR sequence on every 32-element block. The
// externally observable mask state is identical either way: whatever it
// was before the caller's own save+set, 0xFF for the loop's duration,
// restored after -- this only removes (K_blocks-1) redundant CSR
// read-modify-write pairs per row, it does not change any computed value.
// Caller is responsible for the surrounding mask save/set(0xFF)/restore.
static inline float compute_block_dot_product_q8_0_masked(const block_q8_0* a_block, const float* b_col_start) {
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3; // chunk * 8

        __asm__ volatile(
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12"
        );
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__ (
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"             // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"              // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (final_sum)
        :: "t0", "f10", "f2", "f3", "f4", "f5"
    );

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

// Same contract as compute_block_dot_product_q8_0_masked (caller has already
// set the vector mask to 0xFF), but computes FOUR output columns' partial
// dot products from a SINGLE gather+int8->f32 convert of this block's weight
// bytes, instead of repeating that gather+convert once per column.
//
// mul_mat_Q8_0 is the dominant cost for Q8_0-quantized models on this target
// and is memory-bound, not compute-bound: the naive per-column loop re-reads
// and re-converts every weight block once per output column (N times for an
// N-wide activation batch), even though the weight bytes never change across
// columns. This reuses ONE fgb.ps/fcvt.ps.pw per chunk across 4 independent
// FMA accumulators (one per column), cutting weight-side memory/conversion
// traffic ~4x whenever 4 output columns are computed together (e.g. prefill/
// vision-patch batches; single-token decode has nothing to batch and uses
// the original per-column helper unchanged -- see the N%4 remainder path in
// mul_mat_Q8_0.c).
//
// Value-identical to calling compute_block_dot_product_q8_0_masked once per
// column: same per-chunk fgb.ps/fcvt.ps.pw/fmadd.ps sequence and the same
// final *scale, just fanned out over four independent accumulators (f10/
// f13/f16/f19, chosen to avoid the f11/f12/f31 temporaries already live in
// the shared gather/convert/load step) instead of one.
static inline void compute_block_dot_product_q8_0_masked_x4(
        const block_q8_0* a_block,
        const float* b0_col_start, const float* b1_col_start,
        const float* b2_col_start, const float* b3_col_start,
        float* out0, float* out1, float* out2, float* out3) {
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");
    __asm__ volatile("fbci.pi f13, 0" ::: "f13");
    __asm__ volatile("fbci.pi f16, 0" ::: "f16");
    __asm__ volatile("fbci.pi f19, 0" ::: "f19");

    static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3; // chunk * 8

        __asm__ volatile(
            "fgb.ps f11, f31(%[a_ptr])\n"             // Gather 8 int8 bytes from A ONCE
            "fcvt.ps.pw f11, f11\n"                   // Convert int8 vector to float vector ONCE
            "flw.ps f12, %[b0_vec]\n"
            "fmadd.ps f10, f11, f12, f10\n"           // col 0 acc += a_vec * b0_vec
            "flw.ps f12, %[b1_vec]\n"
            "fmadd.ps f13, f11, f12, f13\n"           // col 1 acc += a_vec * b1_vec
            "flw.ps f12, %[b2_vec]\n"
            "fmadd.ps f16, f11, f12, f16\n"           // col 2 acc += a_vec * b2_vec
            "flw.ps f12, %[b3_vec]\n"
            "fmadd.ps f19, f11, f12, f19\n"           // col 3 acc += a_vec * b3_vec
            :
            : [a_ptr] "r"(&a_block->qs[offset]),
              [b0_vec] "m"(*(const float(*)[8])&b0_col_start[offset]),
              [b1_vec] "m"(*(const float(*)[8])&b1_col_start[offset]),
              [b2_vec] "m"(*(const float(*)[8])&b2_col_start[offset]),
              [b3_vec] "m"(*(const float(*)[8])&b3_col_start[offset])
            : "f10", "f11", "f12", "f13", "f16", "f19"
        );
    }

    // Horizontal-reduce each of the 4 accumulators with the exact same
    // reduction sequence the single-column helper uses, run once per
    // accumulator (sequentially, so reusing the f1-f5/t0 scratch registers
    // across the four reductions is safe -- no two reductions overlap).
    float s0, s1, s2, s3;
    __asm__ __volatile__ (
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (s0)
        :: "t0", "f10", "f1", "f2", "f3", "f4", "f5"
    );
    __asm__ __volatile__ (
        "fswizz.ps f1, f13, 0xB1 \n\t"
        "fadd.ps   f2, f13, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (s1)
        :: "t0", "f13", "f1", "f2", "f3", "f4", "f5"
    );
    __asm__ __volatile__ (
        "fswizz.ps f1, f16, 0xB1 \n\t"
        "fadd.ps   f2, f16, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (s2)
        :: "t0", "f16", "f1", "f2", "f3", "f4", "f5"
    );
    __asm__ __volatile__ (
        "fswizz.ps f1, f19, 0xB1 \n\t"
        "fadd.ps   f2, f19, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (s3)
        :: "t0", "f19", "f1", "f2", "f3", "f4", "f5"
    );

    const float scale = fp16_to_fp32(a_block->d);
    *out0 = s0 * scale;
    *out1 = s1 * scale;
    *out2 = s2 * scale;
    *out3 = s3 * scale;
}

// Whole-ROW counterpart to compute_block_dot_product_q8_0_masked_x4: computes
// all K_blocks of a row's dot product against 4 activation columns with a
// SINGLE horizontal reduce per column at the end, instead of one reduce per
// block. Same caller contract (mask already set to 0xFF).
//
// Why this matters: compute_block_dot_product_q8_0_masked_x4 pays a full
// swizzle/add/swizzle/add/move/broadcast/add reduce sequence (7 instructions)
// FOUR TIMES -- once per column -- on every single 32-element block, even
// though the row-level caller immediately throws that reduce away into a
// plain scalar accumulator (see mul_mat_Q8_0.c's `sum0 += p0` loop). For a
// K_blocks-block row that is 28*K_blocks reduce instructions where 28 would
// do: the accumulate-then-reduce-once identity holds because addition
// commutes/associates across blocks, so
//   sum_over_blocks( reduce8(scale_b * chunk_b) )
//     == reduce8( sum_over_blocks(scale_b * chunk_b) )
// -- summing 4 independent 8-lane vector accumulators across all blocks and
// reducing each ONCE at the end is value-identical to reducing every block
// and summing the scalars, just without repeating the reduce K_blocks-1
// extra times per column. Per-block per-lane accumulation order is
// unchanged, so this differs from the block-reduce path only in f32
// rounding order across the *reduce* step, not in which terms are summed.
//
// The whole K loop lives in ONE asm block (same reason as the per-block
// helper above: separate asm statements would let the compiler treat the
// accumulators as clobbered between them). Per-block fp16 scales are
// pre-converted into a stack array ahead of the loop -- the ISA has no way
// to call fp16_to_fp32 (ordinary C) from inside a live asm block without
// spilling the persistent accumulators, so the conversion has to happen
// before entry, exactly like it already does once per call in the per-block
// helper, just amortized over the whole row instead of repeated per call.
#define Q8_ROW_MAX_KBLOCKS 128  // K <= 4096; SmolVLM2's largest GEMM K is well under this
static inline void compute_row_dot_product_q8_0_masked_x4(
        const block_q8_0* q_row, int64_t K_blocks,
        const float* b0_col_base, const float* b1_col_base,
        const float* b2_col_base, const float* b3_col_base,
        float* out0, float* out1, float* out2, float* out3) {

    if (K_blocks > Q8_ROW_MAX_KBLOCKS) {
        // Never reached by any current SmolVLM2 GEMM shape; keeps this
        // function total instead of silently truncating a future larger K.
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int64_t kb = 0; kb < K_blocks; kb++) {
            float p0, p1, p2, p3;
            compute_block_dot_product_q8_0_masked_x4(
                q_row + kb,
                b0_col_base + (kb << 5), b1_col_base + (kb << 5),
                b2_col_base + (kb << 5), b3_col_base + (kb << 5),
                &p0, &p1, &p2, &p3);
            s0 += p0; s1 += p1; s2 += p2; s3 += p3;
        }
        *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
        return;
    }

    float scales[Q8_ROW_MAX_KBLOCKS];
    for (int64_t kb = 0; kb < K_blocks; kb++) {
        scales[kb] = fp16_to_fp32(q_row[kb].d);
    }

    static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    float s0, s1, s2, s3;
    __asm__ __volatile__(
        "flw.ps  f31, %[gather]\n\t"        // gather byte offsets {0..7}, hoisted
        "fbci.pi f10, 0\n\t"                // persistent accumulator, column 0
        "fbci.pi f13, 0\n\t"                // persistent accumulator, column 1
        "fbci.pi f16, 0\n\t"                // persistent accumulator, column 2
        "fbci.pi f19, 0\n\t"                // persistent accumulator, column 3
        "mv   t0, %[qs]\n\t"                // t0 = &q_row[0].qs[0], stride 34B/block
        "mv   t1, %[b0]\n\t"
        "mv   t2, %[b1]\n\t"
        "mv   t3, %[b2]\n\t"
        "mv   t4, %[b3]\n\t"
        "mv   t5, %[sc]\n\t"                // t5 = scales[0], stride 4B/block
        "mv   t6, %[kb]\n\t"
        "beqz t6, 2f\n\t"
        "1:\n\t"
        "lw       a0, 0(t5)\n\t"            // this block's pre-converted f32 scale
        "fbcx.ps  f20, a0\n\t"              // broadcast it to all 8 lanes
        "fbci.pi  f21, 0\n\t"               // per-block temp, column 0
        "fbci.pi  f22, 0\n\t"               // per-block temp, column 1
        "fbci.pi  f23, 0\n\t"               // per-block temp, column 2
        "fbci.pi  f24, 0\n\t"               // per-block temp, column 3
        // chunk 0 (weight bytes 0-7)
        "fgb.ps   f11, f31(t0)\n\t"
        "fcvt.ps.pw f11, f11\n\t"
        "flw.ps   f12, 0(t1)\n\t"
        "fmadd.ps f21, f11, f12, f21\n\t"
        "flw.ps   f12, 0(t2)\n\t"
        "fmadd.ps f22, f11, f12, f22\n\t"
        "flw.ps   f12, 0(t3)\n\t"
        "fmadd.ps f23, f11, f12, f23\n\t"
        "flw.ps   f12, 0(t4)\n\t"
        "fmadd.ps f24, f11, f12, f24\n\t"
        // chunk 1 (weight bytes 8-15)
        "addi     a1, t0, 8\n\t"
        "fgb.ps   f11, f31(a1)\n\t"
        "fcvt.ps.pw f11, f11\n\t"
        "flw.ps   f12, 32(t1)\n\t"
        "fmadd.ps f21, f11, f12, f21\n\t"
        "flw.ps   f12, 32(t2)\n\t"
        "fmadd.ps f22, f11, f12, f22\n\t"
        "flw.ps   f12, 32(t3)\n\t"
        "fmadd.ps f23, f11, f12, f23\n\t"
        "flw.ps   f12, 32(t4)\n\t"
        "fmadd.ps f24, f11, f12, f24\n\t"
        // chunk 2 (weight bytes 16-23)
        "addi     a1, t0, 16\n\t"
        "fgb.ps   f11, f31(a1)\n\t"
        "fcvt.ps.pw f11, f11\n\t"
        "flw.ps   f12, 64(t1)\n\t"
        "fmadd.ps f21, f11, f12, f21\n\t"
        "flw.ps   f12, 64(t2)\n\t"
        "fmadd.ps f22, f11, f12, f22\n\t"
        "flw.ps   f12, 64(t3)\n\t"
        "fmadd.ps f23, f11, f12, f23\n\t"
        "flw.ps   f12, 64(t4)\n\t"
        "fmadd.ps f24, f11, f12, f24\n\t"
        // chunk 3 (weight bytes 24-31)
        "addi     a1, t0, 24\n\t"
        "fgb.ps   f11, f31(a1)\n\t"
        "fcvt.ps.pw f11, f11\n\t"
        "flw.ps   f12, 96(t1)\n\t"
        "fmadd.ps f21, f11, f12, f21\n\t"
        "flw.ps   f12, 96(t2)\n\t"
        "fmadd.ps f22, f11, f12, f22\n\t"
        "flw.ps   f12, 96(t3)\n\t"
        "fmadd.ps f23, f11, f12, f23\n\t"
        "flw.ps   f12, 96(t4)\n\t"
        "fmadd.ps f24, f11, f12, f24\n\t"
        // fold this block's scale into the four persistent accumulators
        "fmadd.ps f10, f21, f20, f10\n\t"
        "fmadd.ps f13, f22, f20, f13\n\t"
        "fmadd.ps f16, f23, f20, f16\n\t"
        "fmadd.ps f19, f24, f20, f19\n\t"
        "addi t0, t0, 34\n\t"                // next block: sizeof(block_q8_0)
        "addi t1, t1, 128\n\t"               // next 32 floats
        "addi t2, t2, 128\n\t"
        "addi t3, t3, 128\n\t"
        "addi t4, t4, 128\n\t"
        "addi t5, t5, 4\n\t"                 // next scale
        "addi t6, t6, -1\n\t"
        "bnez t6, 1b\n\t"
        "2:\n\t"
        // ONE horizontal reduce per column (hoisted out of the block loop)
        "fswizz.ps f1, f10, 0xB1\n\t"
        "fadd.ps   f2, f10, f1, rne\n\t"
        "fswizz.ps f3, f2, 0x4E\n\t"
        "fadd.ps   f4, f2, f3, rne\n\t"
        "fmvz.x.ps a0, f4, 4\n\t"
        "fbcx.ps   f5, a0\n\t"
        "fadd.ps   %[o0], f4, f5, rne\n\t"
        "fswizz.ps f1, f13, 0xB1\n\t"
        "fadd.ps   f2, f13, f1, rne\n\t"
        "fswizz.ps f3, f2, 0x4E\n\t"
        "fadd.ps   f4, f2, f3, rne\n\t"
        "fmvz.x.ps a0, f4, 4\n\t"
        "fbcx.ps   f5, a0\n\t"
        "fadd.ps   %[o1], f4, f5, rne\n\t"
        "fswizz.ps f1, f16, 0xB1\n\t"
        "fadd.ps   f2, f16, f1, rne\n\t"
        "fswizz.ps f3, f2, 0x4E\n\t"
        "fadd.ps   f4, f2, f3, rne\n\t"
        "fmvz.x.ps a0, f4, 4\n\t"
        "fbcx.ps   f5, a0\n\t"
        "fadd.ps   %[o2], f4, f5, rne\n\t"
        "fswizz.ps f1, f19, 0xB1\n\t"
        "fadd.ps   f2, f19, f1, rne\n\t"
        "fswizz.ps f3, f2, 0x4E\n\t"
        "fadd.ps   f4, f2, f3, rne\n\t"
        "fmvz.x.ps a0, f4, 4\n\t"
        "fbcx.ps   f5, a0\n\t"
        "fadd.ps   %[o3], f4, f5, rne\n\t"
        : [o0] "=f"(s0), [o1] "=f"(s1), [o2] "=f"(s2), [o3] "=f"(s3)
        : [gather] "m"(*(const int32_t(*)[8])gather_pattern),
          [qs] "r"(q_row->qs),
          [b0] "r"(b0_col_base), [b1] "r"(b1_col_base),
          [b2] "r"(b2_col_base), [b3] "r"(b3_col_base),
          [sc] "r"(scales), [kb] "r"(K_blocks)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "memory",
          "f1", "f2", "f3", "f4", "f5", "f10", "f11", "f12", "f13", "f16",
          "f19", "f20", "f21", "f22", "f23", "f24", "f31"
    );

    *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
}


// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16_naive(const uint16_t* a_block, const float* b_col_start) {
    float acc_vec[8] __attribute__ ((aligned (32))) = {0.0f};
    // Byte offsets for 16-bit (half-word) elements
    static const int32_t gather_pattern[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    unsigned long temp_mask;

    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    // Load the pattern once into f31 for the duration of all 4 chunks
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    for (int chunk = 0; chunk < 4; chunk++) {
        // Correct pointers:
        // a_block elements are 2 bytes, b_col elements are 4 bytes
        const uint16_t* a_ptr = &a_block[chunk << 3]; // chunk * 8
        const float* b_ptr = &b_col_start[chunk << 3]; // chunk * 8

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"
            "fgh.ps f11, f31(%[a_p])\n"      // Uses {0,2,4,6,8,10,12,14} byte offsets
            "fcvt.ps.f16 f11, f11\n"
            "flw.ps f12, (%[b_p])\n"         // Standard vector load (32-bit floats)
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_p] "r"(a_ptr),
              [b_p] "r"(b_ptr)
            : "f10", "f11", "f12"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
           acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];

}


// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f16_partial(const uint16_t* a_block, const float* b_col_start, int elements) {
    // This matches compute_block_dot_product_f16_naive behavior
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16(const uint16_t* a_block, const float* b_col_start) {
    return compute_block_dot_product_f16_partial(a_block, b_col_start, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f32_partial(const float* a_block, const float* b_col_start, int elements) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Calculate how many full 8-element chunks we can process
    int vec_end = (elements / 8) * 8;

    if (vec_end > 0) {
        // Set mask register to enable all 8 vector elements
        unsigned long temp_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
        __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

        // Process full 8-element chunks
        for (int i = 0; i < vec_end; i += 8) {
            // Vectorized f32 multiply-accumulate
            __asm__ volatile(
                "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"                 // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"                // Store back to accumulator

                : [result] "=m"(*(float(*)[8])acc_vec)
                : [acc] "m"(*(const float(*)[8])acc_vec),
                  [a_vec] "m"(*(const float(*)[8])(a_block + i)),
                  [b_vec] "m"(*(const float(*)[8])(b_col_start + i))
                : "f10", "f11", "f12"
            );
        }

        // Restore original mask
        __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
    }

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    // Handle remaining elements (< 8) with scalar operations
    for (int i = vec_end; i < elements; i++) {
        final_sum += a_block[i] * b_col_start[i];
    }

    return final_sum;
}







// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f32(const float* a_block, const float* b_col_start) {
    return compute_block_dot_product_f32_partial(a_block, b_col_start, QK_F32);

    // float acc_vec[8];
    // unsigned long old_mask;
    // __asm__ volatile(
    //     // Save current mask
    //     "mova.x.m %[old_mask]\n"
    //     // Enable all 8 lanes
    //     "mov.m.x m0, x0, 0xFF\n"

    //     "flw.ps  f11, %[a]\n"
    //     "flw.ps  f12, %[b]\n"
    //     "fmadd.ps f10, f11, f12, f10\n"
    //     "fsw.ps  f10, %[out]\n"
    //     "mova.m.x %[old_mask]\n"

    //     : [out] "=m" (*(float(*)[8])acc_vec),
    //       [old_mask] "=r"(old_mask)
    //     : [a] "m" (*(const float(*)[8])a_block),
    //       [b] "m" (*(const float(*)[8])b_col_start)
    //     : "f10", "f11", "f12"
    // );

    // // Horizontal reduction
    // return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
    //        acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];

}

#endif // BLOCK_OPS_H
