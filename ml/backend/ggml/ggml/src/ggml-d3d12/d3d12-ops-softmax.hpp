#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool softmax_wave_enabled(dispatch_ctx & ctx) {
    if (std::getenv("GGML_D3D12_DISABLE_WAVE") != nullptr || ctx.device == nullptr) return false;

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts = {};
    if (FAILED(ctx.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts, sizeof(opts)))) {
        return false;
    }
    return opts.WaveOps != FALSE;
}

inline bool supports_op_softmax(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;
    if (op->op != GGML_OP_SOFT_MAX) return false;
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
    if (!ggml_are_same_shape(op, op->src[0])) return false;

    float max_bias = 0.0f;
    std::memcpy(&max_bias, reinterpret_cast<const char *>(op->op_params) + sizeof(float), sizeof(float));
    if (max_bias != 0.0f) return false;
    if (op->src[2] != nullptr) return false;

    const ggml_tensor * mask = op->src[1];
    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F32) return false;
        if (!ggml_is_contiguous(mask)) return false;
        // mask must match src in dims 0/1; dims 2/3 may broadcast (typically
        // mask_ne2 == 1 to share an attn mask across all heads).
        const ggml_tensor * src = op->src[0];
        if (mask->ne[0] != src->ne[0]) return false;
        if (mask->ne[1] != src->ne[1]) return false;
        if (mask->ne[2] == 0 || mask->ne[3] == 0) return false;
        if (src->ne[2] % mask->ne[2] != 0) return false;
        if (src->ne[3] % mask->ne[3] != 0) return false;
    }
    return true;
}

inline bool dispatch_softmax(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_softmax(node)) return false;

    const ggml_tensor * src  = node->src[0];
    const ggml_tensor * mask = node->src[1];

    float scale = 1.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));

    const int64_t ne0_i = src->ne[0];
    const int64_t n_rows_i = src->ne[1] * src->ne[2] * src->ne[3];
    if (ne0_i <= 0 || n_rows_i <= 0) return true;

    const uint64_t ne0 = static_cast<uint64_t>(ne0_i);
    const uint64_t n_rows = static_cast<uint64_t>(n_rows_i);
    const uint64_t u32_max = std::numeric_limits<UINT>::max();
    if (n_rows > u32_max || ne0 > u32_max / sizeof(float)) return true;

    const uint64_t row_bytes = ne0 * sizeof(float);
    if (row_bytes == 0 || row_bytes > u32_max) return true;

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (mask != nullptr && !ctx_transition(ctx, mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    // Sliding UAVs so the dispatcher works for tensors past the 4 GB shader
    // address boundary in a > 4 GB parent buffer.
    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * mask_for_bind = mask != nullptr ? mask : src;
    const ggml_tensor * uavs[3] = { src, mask_for_bind, node };
    uint32_t off[3] = { 0, 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 3, &uav_table, off)) return true;
    if ((off[0] % sizeof(float)) != 0 || (off[2] % sizeof(float)) != 0) return true;
    if (mask != nullptr && (off[1] % sizeof(float)) != 0) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 12);
    if (root_sig == nullptr) return true;

    const char * shader = softmax_wave_enabled(ctx) ? "soft_max_f32_wave" : "soft_max_f32";
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    UINT scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    const ggml_tensor * mask_for_dims = mask != nullptr ? mask : src;
    const UINT consts[12] = {
        static_cast<UINT>(ne0),
        static_cast<UINT>(src->ne[1]),
        static_cast<UINT>(src->ne[2]),
        static_cast<UINT>(mask_for_dims->ne[2]),
        static_cast<UINT>(mask_for_dims->ne[3]),
        off[0],
        off[2],
        mask != nullptr ? off[1] : 0u,
        static_cast<UINT>(mask != nullptr ? row_bytes : 0),
        scale_bits,
        mask != nullptr ? 1u : 0u,
        0,
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 12)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(n_rows), 1, 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
