#pragma once

#include <cstring>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool supports_op_unary(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
    if (ggml_nelements(op) != ggml_nelements(op->src[0])) return false;

    switch (op->op) {
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

inline bool dispatch_unary(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_unary(node)) return false;

    const ggml_tensor * src = node->src[0];

    const char * shader = nullptr;
    UINT param0 = 0;
    switch (node->op) {
        case GGML_OP_SQR:  shader = "sqr_f32"; break;
        case GGML_OP_SQRT: shader = "sqrt_f32"; break;
        case GGML_OP_LOG:  shader = "log_f32"; break;
        case GGML_OP_SIN:  shader = "sin_f32"; break;
        case GGML_OP_COS:  shader = "cos_f32"; break;
        case GGML_OP_LEAKY_RELU: {
            shader = "leaky_relu_f32";
            float slope = 0.0f;
            std::memcpy(&slope, node->op_params, sizeof(float));
            std::memcpy(&param0, &slope, sizeof(uint32_t));
            break;
        }
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:         shader = "abs_f32"; break;
                case GGML_UNARY_OP_SGN:         shader = "sgn_f32"; break;
                case GGML_UNARY_OP_NEG:         shader = "neg_f32"; break;
                case GGML_UNARY_OP_STEP:        shader = "step_f32"; break;
                case GGML_UNARY_OP_TANH:        shader = "tanh_f32"; break;
                case GGML_UNARY_OP_ELU:         shader = "elu_f32"; break;
                case GGML_UNARY_OP_RELU:        shader = "relu_f32"; break;
                case GGML_UNARY_OP_SIGMOID:     shader = "sigmoid_f32"; break;
                case GGML_UNARY_OP_GELU:        shader = "gelu_f32"; break;
                case GGML_UNARY_OP_GELU_QUICK:  shader = "gelu_quick_f32"; break;
                case GGML_UNARY_OP_GELU_ERF:    shader = "gelu_erf_f32"; break;
                case GGML_UNARY_OP_SILU:        shader = "silu_f32"; break;
                case GGML_UNARY_OP_HARDSWISH:   shader = "hardswish_f32"; break;
                case GGML_UNARY_OP_HARDSIGMOID: shader = "hardsigmoid_f32"; break;
                case GGML_UNARY_OP_EXP:         shader = "exp_f32"; break;
                default: return true;
            }
            break;
        default: return true;
    }

    const size_t bytes = ggml_nbytes(node);
    if (bytes == 0) return true;
    const size_t count = ggml_nelements(node);
    if (count > 0xFFFFFFFFull) return true;

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    uint32_t off[2] = { 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 2, &uav_table, off)) return true;
    if ((off[0] % 4) != 0 || (off[1] % 4) != 0) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 2, 4);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[4] = {
        static_cast<UINT>(count),
        off[0],
        off[1],
        param0,
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 2, consts, 4)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12