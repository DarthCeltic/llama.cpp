//******************************************************************************
// SSM_CONV F32 Kernel -- causal depthwise 1D convolution (short-conv / SSM prelude)
//
// Ports ggml_compute_forward_ssm_conv_f32 (ggml/src/ggml-cpu/ops.cpp) faithfully:
// for each (seq, token, row) triple, dst = dot(src0 sliding window, src1 kernel row).
// Previously unimplemented on this backend -- GGML_OP_SSM_CONV fell through to
// ggml_backend_et_device_supports_op's `default: supported = false`, forcing ggml's
// scheduler to round-trip every call through a CPU fallback backend. LFM2's
// shortconv layers call this every recurrent layer, every token.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_ssm_conv_params {
    struct ggml_tensor src0;     // conv_x: {d_conv-1+n_t, d_inner, n_seqs}, F32
    struct ggml_tensor src1;     // conv1d.weight: {d_conv, d_inner}, F32
    struct ggml_tensor dst;      // {d_inner, n_t, n_seqs}, F32
};

int entry_point(struct ggml_et_ssm_conv_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return -1;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor* src0 = &params->src0; // conv_x
    struct ggml_tensor* src1 = &params->src1; // conv1d.weight
    struct ggml_tensor* dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float* src0_data = (const float*)src0->data;
    const float* src1_data = (const float*)src1->data;
    float*       dst_data  = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t nc  = src1->ne[0]; // d_conv
    const int64_t ncs = src0->ne[0]; // d_conv - 1 + n_t
    const int64_t nr  = src0->ne[1]; // d_inner
    const int64_t n_t = dst->ne[1];  // tokens per sequence
    const int64_t n_s = dst->ne[2];  // number of sequences in the batch

    if (dst->ne[0] != nr) {
        return -1;
    }

    // Reference semantics assume src0 is row-contiguous (nb[1] == ne[0]*sizeof(float))
    // so that a base pointer can be indexed with plain element offsets below --
    // matches the CPU reference's own GGML_ASSERT and is enforced by supports_op.
    if (src0->nb[0] != (uint64_t)sizeof(float) || src1->nb[0] != (uint64_t)sizeof(float) ||
        src0->nb[1] != (uint64_t)(src0->ne[0] * (int64_t)sizeof(float))) {
        return -1;
    }

    // rows per thread -- same distribution strategy as the CPU reference
    const int64_t dr  = (nr + num_threads - 1) / num_threads;
    const int64_t ir0 = dr * thread_id;
    int64_t       ir1 = ir0 + dr;
    if (ir1 > nr) {
        ir1 = nr;
    }
    if (ir0 >= nr) {
        return 0;
    }
    const int64_t ir = ir1 - ir0;

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs} sliding window
            const float* s = (const float*)((const char*)src0_data + ir0 * src0->nb[1] + i2 * src0->nb[0] + i3 * src0->nb[2]);
            const float* c = (const float*)((const char*)src1_data + ir0 * src1->nb[1]);
            float*       x = (float*)((char*)dst_data + ir0 * dst->nb[0] + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t i1 = 0; i1 < ir; ++i1) {
                float sumf = 0.0f;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1 * ncs] * c[i0 + i1 * nc];
                }
                x[i1] = sumf;
            }
        }
    }

    return 0;
}
