#pragma once

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool supports_op_binary(const ggml_tensor * op) {
    if (op == nullptr) return false;
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            break;
        default:
            return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) return false;
    if (!ggml_are_same_shape(op->src[0], op->src[1])) return false;
    if (!ggml_are_same_shape(op, op->src[0])) return false;
    return true;
}

inline bool dispatch_binary(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_binary(node)) return false;

    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];

    const char * shader = nullptr;
    switch (node->op) {
        case GGML_OP_ADD: shader = "add_f32"; break;
        case GGML_OP_SUB: shader = "sub_f32"; break;
        case GGML_OP_MUL: shader = "mul_f32"; break;
        case GGML_OP_DIV: shader = "div_f32"; break;
        default: return true;
    }

    const tensor_resource ar = ctx_resolve_tensor(ctx, a);
    const tensor_resource br = ctx_resolve_tensor(ctx, b);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!ar.valid || !br.valid || !dr.valid) return true;

    const size_t count = ggml_nelements(node);
    if (count == 0) return true;
    if ((ar.offset_bytes % 4) != 0 || (br.offset_bytes % 4) != 0 || (dr.offset_bytes % 4) != 0) return true;
    if (count > 0xFFFFFFFFull ||
        ar.offset_bytes > 0xFFFFFFFFull ||
        br.offset_bytes > 0xFFFFFFFFull ||
        dr.offset_bytes > 0xFFFFFFFFull) return true;

    if (!ctx_transition(ctx, a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { a, b, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 4);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[4] = {
        static_cast<UINT>(count),
        static_cast<UINT>(ar.offset_bytes),
        static_cast<UINT>(br.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 4)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
