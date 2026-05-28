#pragma once

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool reductions_wave_enabled(dispatch_ctx & ctx) {
    if (std::getenv("GGML_D3D12_DISABLE_WAVE") != nullptr || ctx.device == nullptr) return false;

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts = {};
    if (FAILED(ctx.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts, sizeof(opts)))) {
        return false;
    }
    return opts.WaveOps != FALSE;
}

inline bool reductions_same_shape_f32(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 && ggml_are_same_shape(op, op->src[0]);
}

inline bool supports_op_reductions(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;

    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32) return false;
    if (src->ne[0] <= 0) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;

    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];

    switch (op->op) {
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return reductions_same_shape_f32(op);
        case GGML_OP_L2_NORM: {
            float eps = 0.0f;
            std::memcpy(&eps, op->op_params, sizeof(float));
            return reductions_same_shape_f32(op) && eps <= 1e-4f;
        }
        case GGML_OP_GROUP_NORM: {
            const int n_groups = op->op_params[0];
            return reductions_same_shape_f32(op) && n_groups > 0 && n_groups <= src->ne[2];
        }
        case GGML_OP_SUM:
            return op->type == GGML_TYPE_F32 && ggml_nelements(op) == 1;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->type == GGML_TYPE_F32 && op->ne[0] == 1 && op->ne[1] == src->ne[1] &&
                   op->ne[2] == src->ne[2] && op->ne[3] == src->ne[3];
        case GGML_OP_ARGMAX:
            return op->type == GGML_TYPE_I32 && src->ne[0] <= 2147483647ll && ggml_nelements(op) == rows;
        default:
            return false;
    }
}

inline bool reductions_fits_u32(uint64_t value) {
    return value <= 0xFFFFFFFFull;
}

inline bool dispatch_reductions(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_reductions(node)) return false;

    const ggml_tensor * src = node->src[0];
    const uint64_t total = static_cast<uint64_t>(ggml_nelements(src));
    if (total == 0) return true;

    const bool use_wave = reductions_wave_enabled(ctx);
    const char * shader = nullptr;
    UINT param0 = 0;
    uint64_t param1_u64 = 0;
    uint64_t reduction_width = static_cast<uint64_t>(src->ne[0]);
    uint64_t dispatch_x = total / reduction_width;
    uint64_t dispatch_y = 1;

    switch (node->op) {
        case GGML_OP_NORM:
            shader = use_wave ? "norm_f32_wave" : "norm_f32";
            std::memcpy(&param0, node->op_params, sizeof(float));
            break;
        case GGML_OP_RMS_NORM:
            shader = use_wave ? "rms_norm_f32_wave" : "rms_norm_f32";
            std::memcpy(&param0, node->op_params, sizeof(float));
            break;
        case GGML_OP_L2_NORM:
            shader = use_wave ? "l2_norm_f32_wave" : "l2_norm_f32";
            std::memcpy(&param0, node->op_params, sizeof(float));
            break;
        case GGML_OP_GROUP_NORM: {
            shader = use_wave ? "group_norm_f32_wave" : "group_norm_f32";
            const int n_groups = node->op_params[0];
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params + 1, sizeof(float));
            std::memcpy(&param0, &eps, sizeof(float));

            const uint64_t plane = static_cast<uint64_t>(src->ne[0]) * static_cast<uint64_t>(src->ne[1]);
            const uint64_t channels = static_cast<uint64_t>(src->ne[2]);
            const uint64_t channels_per_group = (channels + static_cast<uint64_t>(n_groups) - 1ull) / static_cast<uint64_t>(n_groups);
            reduction_width = plane * channels_per_group;
            param1_u64 = plane * channels;
            dispatch_x = static_cast<uint64_t>(n_groups);
            dispatch_y = static_cast<uint64_t>(src->ne[3]);
            break;
        }
        case GGML_OP_SUM:
            shader = use_wave ? "sum_f32_wave" : "sum_f32";
            reduction_width = total;
            dispatch_x = 1;
            break;
        case GGML_OP_SUM_ROWS:
            shader = use_wave ? "sum_rows_f32_wave" : "sum_rows_f32";
            break;
        case GGML_OP_MEAN:
            shader = use_wave ? "mean_f32_wave" : "mean_f32";
            break;
        case GGML_OP_ARGMAX:
            shader = use_wave ? "argmax_f32_wave" : "argmax_f32";
            break;
        default:
            return true;
    }

    if (!reductions_fits_u32(reduction_width) || !reductions_fits_u32(dispatch_x) ||
        !reductions_fits_u32(dispatch_y) || !reductions_fits_u32(param1_u64)) {
        return true;
    }
    if (ggml_nbytes(src) > 0xFFFFFFFFull || ggml_nbytes(node) > 0xFFFFFFFFull) {
        return true;
    }

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    uint32_t off[2] = { 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 2, &uav_table, off)) return true;
    if ((off[0] % 4) != 0 || (off[1] % 4) != 0) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 5);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (pso == nullptr) return true;

    const UINT consts[5] = {
        static_cast<UINT>(reduction_width),
        off[0],
        off[1],
        param0,
        static_cast<UINT>(param1_u64),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 5)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(dispatch_x), static_cast<UINT>(dispatch_y), 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
