#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline float rope_param_f32(const ggml_tensor * op, int index) {
    float value = 0.0f;
    std::memcpy(&value, op->op_params + index, sizeof(value));
    return value;
}

inline UINT rope_param_u32_bits(const ggml_tensor * op, int index) {
    UINT value = 0;
    std::memcpy(&value, op->op_params + index, sizeof(value));
    return value;
}

inline bool rope_u32_range_fits(size_t offset, uint64_t bytes) {
    const uint64_t u32_max = static_cast<uint64_t>(std::numeric_limits<UINT>::max());
    const uint64_t u32_range = u32_max + 1ull;
    return static_cast<uint64_t>(offset) <= u32_max && bytes <= u32_range && bytes <= u32_range - static_cast<uint64_t>(offset);
}

inline bool supports_op_rope(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ROPE) return false;
    if (op->src[0] == nullptr || op->src[1] == nullptr) return false;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];

    if (op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) return false;
    if (!ggml_are_same_shape(op, src0)) return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    if (!ggml_is_vector(src1)) return false;

    const int n_dims = op->op_params[1];
    const int mode   = op->op_params[2];
    if (n_dims <= 0 || (n_dims & 1) != 0 || n_dims > src0->ne[0]) return false;
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) return false;

    for (int i = 0; i < GGML_MROPE_SECTIONS; ++i) {
        if (op->op_params[11 + i] != 0) return false;
    }

    if (src1->ne[0] < src0->ne[2]) return false;

    if (src2 != nullptr) {
        if (src2->type != GGML_TYPE_F32) return false;
        if (!ggml_is_contiguous(src2)) return false;
        if (src2->ne[0] < n_dims / 2) return false;
    }

    const float ext_factor  = rope_param_f32(op, 7);
    const float attn_factor = rope_param_f32(op, 8);
    if (ext_factor != 0.0f || attn_factor != 1.0f) return false;

    return true;
}

inline bool dispatch_rope(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_rope(node)) return false;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * src2 = node->src[2];

    const tensor_resource sr = ctx_resolve_tensor(ctx, src0);
    const tensor_resource pr = ctx_resolve_tensor(ctx, src1);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!sr.valid || !pr.valid || !dr.valid) return true;

    tensor_resource fr{};
    if (src2 != nullptr) {
        fr = ctx_resolve_tensor(ctx, src2);
        if (!fr.valid) return true;
    }

    const uint64_t ne0 = static_cast<uint64_t>(src0->ne[0]);
    const uint64_t ne1 = static_cast<uint64_t>(src0->ne[1]);
    const uint64_t ne2 = static_cast<uint64_t>(src0->ne[2]);
    const uint64_t ne3 = static_cast<uint64_t>(src0->ne[3]);
    const uint64_t n_dims = static_cast<uint64_t>(node->op_params[1]);
    const uint64_t u32_max = static_cast<uint64_t>(std::numeric_limits<UINT>::max());

    if (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) return true;
    if (ne0 > u32_max || ne1 > u32_max || ne2 > u32_max || ne3 > u32_max || n_dims > u32_max) return true;
    if ((sr.offset_bytes % 4) != 0 || (pr.offset_bytes % 4) != 0 || (dr.offset_bytes % 4) != 0) return true;
    if (src2 != nullptr && (fr.offset_bytes % 4) != 0) return true;

    const uint64_t data_bytes = static_cast<uint64_t>(ggml_nbytes(src0));
    const uint64_t dst_bytes  = static_cast<uint64_t>(ggml_nbytes(node));
    const uint64_t pos_bytes  = ne2 * sizeof(int32_t);
    const uint64_t freq_bytes = (n_dims / 2ull) * sizeof(float);
    if (!rope_u32_range_fits(sr.offset_bytes, data_bytes) ||
        !rope_u32_range_fits(dr.offset_bytes, dst_bytes) ||
        !rope_u32_range_fits(pr.offset_bytes, pos_bytes)) {
        return true;
    }
    if (src2 != nullptr && !rope_u32_range_fits(fr.offset_bytes, freq_bytes)) return true;

    if (!ctx_transition(ctx, src0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, src1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (src2 != nullptr && !ctx_transition(ctx, src2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * freq_for_bind = src2 != nullptr ? src2 : src0;
    const ggml_tensor * uavs[4] = { src0, src1, freq_for_bind, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 4, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 4, 12);
    if (rs == nullptr) return true;

    const char * shader = node->op_params[2] == GGML_ROPE_TYPE_NEOX ? "rope_neox_f32" : "rope_norm_f32";
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (pso == nullptr) return true;

    const UINT consts[12] = {
        static_cast<UINT>(ne0),
        static_cast<UINT>(ne1),
        static_cast<UINT>(ne2),
        static_cast<UINT>(ne3),
        static_cast<UINT>(n_dims),
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        static_cast<UINT>(pr.offset_bytes),
        src2 != nullptr ? static_cast<UINT>(fr.offset_bytes) : 0u,
        rope_param_u32_bits(node, 5),
        rope_param_u32_bits(node, 6),
        src2 != nullptr ? 1u : 0u,
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 4, consts, 12)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(ne1), static_cast<UINT>(ne2), static_cast<UINT>(ne3));
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
