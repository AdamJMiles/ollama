#pragma once

// Memory ops: GGML_OP_CPY, GGML_OP_DUP, etc.
// Currently implements: contiguous F32 byte-copy via copy_f32 kernel.
//
// Future work (Phase 5 op group I "memops"): CONT/CONCAT/PAD/REPEAT/
// GET_ROWS/SET_ROWS — each as additional `if (...) { ... }` arms below.

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool supports_op_memops(const ggml_tensor * op) {
    if (op == nullptr) return false;
    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP: {
            const ggml_tensor * src = op->src[0];
            if (src == nullptr) return false;
            if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            if (ggml_nelements(op) != ggml_nelements(src)) return false;
            return true;
        }
        default:
            return false;
    }
}

inline bool dispatch_memops(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_memops(node)) return false;

    // (Currently only CPY/DUP F32 contiguous; future memops branch here.)
    const ggml_tensor * src = node->src[0];

    const tensor_resource src_r = ctx_resolve_tensor(ctx, src);
    const tensor_resource dst_r = ctx_resolve_tensor(ctx, node);
    if (!src_r.valid || !dst_r.valid) return true; // claimed but failed

    const size_t bytes = ggml_nbytes(src);
    if (bytes == 0) return true; // trivial no-op
    if ((bytes % 4) != 0) return true;
    if ((src_r.offset_bytes % 4) != 0 || (dst_r.offset_bytes % 4) != 0) return true;
    if (bytes > 0xFFFFFFFFull ||
        src_r.offset_bytes > 0xFFFFFFFFull ||
        dst_r.offset_bytes > 0xFFFFFFFFull) {
        return true;
    }

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, /*uav=*/2, /*dwords=*/4);
    if (root_sig == nullptr) return true;

    ID3D12PipelineState * pso = ctx.psos->get("copy_f32", root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[4] = {
        static_cast<UINT>(bytes),
        static_cast<UINT>(src_r.offset_bytes),
        static_cast<UINT>(dst_r.offset_bytes),
        0,
    };

    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, /*uav=*/2, consts, /*dwords=*/4)) {
        return true;
    }

    const UINT dwords = static_cast<UINT>(bytes / 4);
    ctx_dispatch_1d(ctx, dwords, /*threads_per_group=*/256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
