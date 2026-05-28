#pragma once

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool binary_op_supported(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return true;
        default:
            return false;
    }
}

inline bool binary_is_broadcast_compatible(const ggml_tensor * a, const ggml_tensor * b) {
    // Each src1 dim must divide the corresponding src0 dim.
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (b->ne[i] <= 0 || a->ne[i] <= 0) return false;
        if ((a->ne[i] % b->ne[i]) != 0) return false;
    }
    return true;
}

inline bool supports_op_binary(const ggml_tensor * op) {
    if (op == nullptr) return false;
    if (!binary_op_supported(op->op)) return false;
    if (op->src[0] == nullptr || op->src[1] == nullptr) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) return false;

    if (!ggml_are_same_shape(op, op->src[0])) return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) return false;

    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];

    if (ggml_are_same_shape(a, b)) {
        // Fast path: same-shape, both contiguous.
        if (!ggml_is_contiguous(b)) return false;
        return true;
    }

    // Broadcast path: src1 dims must divide src0 dims (ggml broadcast rules).
    if (std::getenv("GGML_D3D12_DISABLE_BINARY_BCAST") != nullptr) return false;
    if (!binary_is_broadcast_compatible(a, b)) return false;
    // src1 must have a sensible element-wise stride (4 bytes per F32) on dim0
    // so our shader can index it via byte offsets.
    if (b->nb[0] != sizeof(float)) return false;
    return true;
}

inline bool dispatch_binary(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_binary(node)) return false;

    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    const bool same_shape = ggml_are_same_shape(a, b);

    const char * shader = nullptr;
    if (same_shape) {
        switch (node->op) {
            case GGML_OP_ADD: shader = "add_f32"; break;
            case GGML_OP_SUB: shader = "sub_f32"; break;
            case GGML_OP_MUL: shader = "mul_f32"; break;
            case GGML_OP_DIV: shader = "div_f32"; break;
            default: return true;
        }
    } else {
        switch (node->op) {
            case GGML_OP_ADD: shader = "add_f32_bcast"; break;
            case GGML_OP_SUB: shader = "sub_f32_bcast"; break;
            case GGML_OP_MUL: shader = "mul_f32_bcast"; break;
            case GGML_OP_DIV: shader = "div_f32_bcast"; break;
            default: return true;
        }
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

    if (same_shape) {
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

    // Broadcast path. Stride bounds check (each must fit u32).
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (static_cast<uint64_t>(b->nb[i]) > 0xFFFFFFFFull) return true;
        if (static_cast<uint64_t>(a->ne[i]) > 0xFFFFFFFFull) return true;
        if (static_cast<uint64_t>(b->ne[i]) > 0xFFFFFFFFull) return true;
    }


    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 16);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[16] = {
        static_cast<UINT>(count),
        static_cast<UINT>(a->ne[0]), static_cast<UINT>(a->ne[1]), static_cast<UINT>(a->ne[2]), static_cast<UINT>(a->ne[3]),
        static_cast<UINT>(b->ne[0]), static_cast<UINT>(b->ne[1]), static_cast<UINT>(b->ne[2]), static_cast<UINT>(b->ne[3]),
        static_cast<UINT>(b->nb[0]), static_cast<UINT>(b->nb[1]), static_cast<UINT>(b->nb[2]), static_cast<UINT>(b->nb[3]),
        static_cast<UINT>(ar.offset_bytes),
        static_cast<UINT>(br.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 16)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
