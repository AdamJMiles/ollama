#pragma once

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool glu_same_outer_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

inline bool supports_op_glu(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;
    if (op->op != GGML_OP_GLU) return false;
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;

    if (op->src[1] != nullptr) {
        if (op->src[1]->type != GGML_TYPE_F32) return false;
        if (!ggml_is_contiguous(op->src[1])) return false;
        if (!ggml_are_same_shape(op->src[0], op->src[1])) return false;
        if (!ggml_are_same_shape(op, op->src[0])) return false;
    } else {
        if (op->ne[0] <= 0 || op->src[0]->ne[0] <= 0 || (op->src[0]->ne[0] % 2) != 0 || op->src[0]->ne[0] / 2 != op->ne[0]) return false;
        if (!glu_same_outer_shape(op, op->src[0])) return false;
    }

    switch (ggml_get_glu_op(op)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            return true;
        default:
            return false;
    }
}

inline bool dispatch_glu(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_glu(node)) return false;

    const bool split = node->src[1] != nullptr;
    const ggml_glu_op gop = ggml_get_glu_op(node);
    const bool swapped = node->op_params[1] != 0;

    const char * shader = nullptr;
    switch (gop) {
        case GGML_GLU_OP_REGLU:       shader = split ? "reglu_split_f32"       : "reglu_f32"; break;
        case GGML_GLU_OP_GEGLU:       shader = split ? "geglu_split_f32"       : "geglu_f32"; break;
        case GGML_GLU_OP_SWIGLU:      shader = split ? "swiglu_split_f32"      : "swiglu_f32"; break;
        case GGML_GLU_OP_GEGLU_ERF:   shader = split ? "geglu_erf_split_f32"   : "geglu_erf_f32"; break;
        case GGML_GLU_OP_GEGLU_QUICK: shader = split ? "geglu_quick_split_f32" : "geglu_quick_f32"; break;
        default: return true;
    }

    const tensor_resource sr = ctx_resolve_tensor(ctx, node->src[0]);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!sr.valid || !dr.valid) return true;

    tensor_resource gr{};
    if (split) {
        gr = ctx_resolve_tensor(ctx, node->src[1]);
        if (!gr.valid) return true;
    }

    const int64_t count_out_i64 = ggml_nelements(node);
    const int64_t n_out_i64 = node->ne[0];
    if (count_out_i64 <= 0 || n_out_i64 <= 0) return true;

    constexpr uint64_t u32max = 0xFFFFFFFFull;
    if (static_cast<uint64_t>(count_out_i64) > u32max ||
        static_cast<uint64_t>(n_out_i64) > u32max ||
        (!split && static_cast<uint64_t>(n_out_i64) > (u32max / 2)) ||
        sr.offset_bytes > u32max ||
        dr.offset_bytes > u32max ||
        (split && gr.offset_bytes > u32max)) {
        return true;
    }
    if ((sr.offset_bytes % 4) != 0 ||
        (dr.offset_bytes % 4) != 0 ||
        (split && (gr.offset_bytes % 4) != 0)) {
        return true;
    }

    if (!ctx_transition(ctx, node->src[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (split && !ctx_transition(ctx, node->src[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    const UINT count_out = static_cast<UINT>(count_out_i64);
    const UINT n_out = static_cast<UINT>(n_out_i64);

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    if (split) {
        const ggml_tensor * uavs[3] = { node->src[0], node->src[1], node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;
        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 3, 4);
        if (rs == nullptr) return true;
        ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
        if (pso == nullptr) return true;
        const UINT consts[4] = { count_out, static_cast<UINT>(sr.offset_bytes), static_cast<UINT>(gr.offset_bytes), static_cast<UINT>(dr.offset_bytes) };
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 3, consts, 4)) return true;
    } else {
        const ggml_tensor * uavs[2] = { node->src[0], node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;
        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 5);
        if (rs == nullptr) return true;
        ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
        if (pso == nullptr) return true;
        const UINT consts[5] = { count_out, n_out, static_cast<UINT>(sr.offset_bytes), static_cast<UINT>(dr.offset_bytes), swapped ? 1u : 0u };
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 5)) return true;
    }

    ctx_dispatch_1d(ctx, count_out, 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12