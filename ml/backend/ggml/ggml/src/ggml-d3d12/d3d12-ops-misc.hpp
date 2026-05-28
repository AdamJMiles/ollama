#pragma once

#include <cstring>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool supports_op_misc(const ggml_tensor * op) {
    if (op == nullptr) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(op)) return false;

    switch (op->op) {
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
            if (op->src[0] == nullptr) return false;
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op->src[0])) return false;
            return true;
        case GGML_OP_ARANGE:
            return true;
        default:
            return false;
    }
}

inline bool dispatch_misc(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_misc(node)) return false;

    if (node->op == GGML_OP_ARANGE) {
        const tensor_resource dr = ctx_resolve_tensor(ctx, node);
        if (!dr.valid) return true;
        if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

        D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
        const ggml_tensor * uavs[1] = { node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 1, &uav_table)) return true;

        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 1, 4);
        if (!rs) return true;
        ID3D12PipelineState * pso = ctx.psos->get("arange_f32", rs, {});
        if (!pso) return true;

        float start = ((const float *) node->op_params)[0];
        float stop  = ((const float *) node->op_params)[1];
        float step  = ((const float *) node->op_params)[2];
        (void) stop;
        UINT count = (UINT) ggml_nelements(node);
        UINT consts[4] = { count, (UINT) dr.offset_bytes, 0, 0 };
        std::memcpy(&consts[2], &start, sizeof(UINT));
        std::memcpy(&consts[3], &step,  sizeof(UINT));
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 1, consts, 4)) return true;

        ctx_dispatch_1d(ctx, count, 256);
        ctx_uav_barrier(ctx, node);
        return true;
    }

    const ggml_tensor * src = node->src[0];
    const char * shader = (node->op == GGML_OP_SCALE) ? "scale_f32" : "clamp_f32";
    const tensor_resource sr = ctx_resolve_tensor(ctx, src);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!sr.valid || !dr.valid) return true;

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 5);
    if (!rs) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (!pso) return true;

    UINT count = (UINT) ggml_nelements(node);
    UINT p0 = 0;
    UINT p1 = 0;
    if (node->op == GGML_OP_SCALE) {
        float scale = ((const float *) node->op_params)[0];
        float bias  = ((const float *) node->op_params)[1];
        std::memcpy(&p0, &scale, sizeof(UINT));
        std::memcpy(&p1, &bias,  sizeof(UINT));
    } else {
        float mn = ((const float *) node->op_params)[0];
        float mx = ((const float *) node->op_params)[1];
        std::memcpy(&p0, &mn, sizeof(UINT));
        std::memcpy(&p1, &mx, sizeof(UINT));
    }
    UINT consts[5] = { count, (UINT) sr.offset_bytes, (UINT) dr.offset_bytes, p0, p1 };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 5)) return true;

    ctx_dispatch_1d(ctx, count, 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
