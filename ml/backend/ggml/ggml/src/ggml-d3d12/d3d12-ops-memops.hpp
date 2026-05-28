#pragma once

// Memory ops: GGML_OP_CPY, GGML_OP_DUP, GGML_OP_SET_ROWS.
// Currently implements:
//   - Contiguous F32 byte-copy via copy_f32 kernel (CPY/DUP).
//   - SET_ROWS for F32 source into F32 or F16 destination, with I32/I64 idx.
//     This is the unblocker for KV cache updates in the ollama new engine.
//
// Future work (Phase 5 op group I "memops"): CONT/CONCAT/PAD/REPEAT/
// GET_ROWS — each as additional arms below.

#include "d3d12-ops-common.hpp"
#include <cstdlib>

namespace ggml_d3d12 {

inline bool memops_env_disable_set_rows() {
    static const bool v = std::getenv("GGML_D3D12_DISABLE_SET_ROWS") != nullptr;
    return v;
}

inline bool memops_env_log_set_rows() {
    static const bool v = std::getenv("GGML_D3D12_LOG_SET_ROWS") != nullptr;
    return v;
}

inline bool supports_op_memops(const ggml_tensor * op) {
    if (op == nullptr) return false;
    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP: {
            const ggml_tensor * src = op->src[0];
            if (src == nullptr) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            if (ggml_nelements(op) != ggml_nelements(src)) return false;
            // Same-type byte copy (F32->F32 covers the original case but also
            // F16->F16, Q8_0->Q8_0, etc. since this is just a bulk byte copy
            // for contiguous tensors of identical layout).
            //
            // Quantized types must additionally have matching shape: for Q-types
            // the CPU reference walks blocks in shape order (i0,i1,i2,i3) and
            // assigns to dst's block grid, which only coincides with a flat
            // memcpy when shapes are identical. ne_src != ne_dst with
            // ggml_nbytes equal still produces a different element layout per
            // CPU semantics.
            //
            // copy_f32 issues one 32-bit Store per thread, so total bytes must
            // be a multiple of 4. Q4_0 (18 B/block) rows of odd block count are
            // not 4-aligned and would silently lose the trailing 2 bytes.
            if (src->type == op->type) {
                if (ggml_nbytes(src) != ggml_nbytes(op)) return false;
                if (ggml_is_quantized(src->type) && !ggml_are_same_shape(src, op)) return false;
                if ((ggml_nbytes(op) % 4u) != 0u) return false;
                return true;
            }
            // F32 -> F16 conversion (KV cache writes during decode).
            if (src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16) {
                return true;
            }
            return false;
        }
        case GGML_OP_SET_ROWS: {
            if (memops_env_disable_set_rows()) return false;
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            if (src0 == nullptr || src1 == nullptr) return false;
            if (src0->type != GGML_TYPE_F32) return false;
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) return false;
            if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) return false;
            // src row contents must be contiguous in column-dim so we can
            // stride by sizeof(f32). All real ggml call sites satisfy this.
            if (src0->nb[0] != ggml_type_size(src0->type)) return false;
            if (op->nb[0]   != ggml_type_size(op->type))   return false;
            // dst row stride must be 4-byte aligned for our raw-buffer stores.
            if ((op->nb[1] % 4) != 0) return false;
            return true;
        }
        default:
            return false;
    }
}

// Forward decls for the two memops kernels.
static bool dispatch_cpy_dup(dispatch_ctx & ctx, const ggml_tensor * node);
static bool dispatch_set_rows(dispatch_ctx & ctx, const ggml_tensor * node);

inline bool dispatch_memops(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_memops(node)) return false;
    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return dispatch_cpy_dup(ctx, node);
        case GGML_OP_SET_ROWS:
            return dispatch_set_rows(ctx, node);
        default:
            return false;
    }
}

static bool dispatch_cpy_dup(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];

    if (src->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F16) {
        // F32 -> F16 conversion. Each thread emits one 32-bit store covering
        // two F16 dst elements; a trailing odd element RMWs the half-uint.
        const uint64_t elem_count = static_cast<uint64_t>(ggml_nelements(node));
        if (elem_count == 0) return true;
        if (elem_count > 0xFFFFFFFFull) return true;

        const uint64_t src_bytes = elem_count * sizeof(float);
        const uint64_t dst_bytes = ((elem_count + 1ull) / 2ull) * 4ull;
        if (src_bytes > 0xFFFFFFFFull || dst_bytes > 0xFFFFFFFFull) return true;

        if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
        if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

        D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
        const ggml_tensor * uavs[2] = { src, node };
        uint32_t off[2] = { 0, 0 };
        if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 2, &uav_table, off)) return true;
        if ((off[0] % 4) != 0 || (off[1] % 4) != 0) return true;

        ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 2, 4);
        if (root_sig == nullptr) return true;

        ID3D12PipelineState * pso = ctx.psos->get("cpy_f32_to_f16_fp16", root_sig, {});
        if (pso == nullptr) return true;

        const UINT consts[4] = {
            static_cast<UINT>(elem_count),
            off[0],
            off[1],
            0,
        };
        if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 2, consts, 4)) return true;

        const UINT threads = static_cast<UINT>((elem_count + 1ull) / 2ull);
        ctx_dispatch_1d(ctx, threads, 256);
        ctx_uav_barrier(ctx, node);
        return true;
    }

    // Same-type byte copy.
    const size_t bytes = ggml_nbytes(src);
    if (bytes == 0) return true; // trivial no-op
    if ((bytes % 4) != 0) return true;
    if (bytes > 0xFFFFFFFFull) return true;

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table2 = {};
    const ggml_tensor * uavs2[2] = { src, node };
    uint32_t off2[2] = { 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs2, 2, &uav_table2, off2)) return true;
    if ((off2[0] % 4) != 0 || (off2[1] % 4) != 0) return true;

    ID3D12RootSignature * root_sig2 = ctx_get_root_sig(ctx, /*uav=*/2, /*dwords=*/4);
    if (root_sig2 == nullptr) return true;

    ID3D12PipelineState * pso2 = ctx.psos->get("copy_f32", root_sig2, {});
    if (pso2 == nullptr) return true;

    const UINT consts2[4] = {
        static_cast<UINT>(bytes),
        off2[0],
        off2[1],
        0,
    };

    if (!ctx_bind_compute(ctx, pso2, root_sig2, uav_table2, /*uav=*/2, consts2, /*dwords=*/4)) {
        return true;
    }

    const UINT dwords = static_cast<UINT>(bytes / 4);
    ctx_dispatch_1d(ctx, dwords, /*threads_per_group=*/256);
    ctx_uav_barrier(ctx, node);
    return true;
}

// GGML_OP_SET_ROWS — write F32 src rows into indexed rows of an F32/F16 dst.
// One thread group per (i, i02, i03) source row; group of 64 threads covers
// the row's columns. Used by the new-engine KV cache.
//
// Source tensors here may live on memory not directly bindable as a UAV by
// our shader: src0 commonly comes from CPU-backend buffers (e.g. when the
// preceding op is run on CPU due to a missing D3D12 kernel) and src1 (idx)
// lives in a D3D12 UPLOAD-heap host buffer (no UAV access). We use
// ctx_stage_tensor_uav to bring both into the per-graph DEFAULT-heap UAV
// scratch buffer transparently. dst (the KV cache) is always on a D3D12
// DEFAULT-heap buffer of this device, so staging is a pass-through.
static bool dispatch_set_rows(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const tensor_resource src_r = ctx_stage_tensor_uav(ctx, src0);
    const tensor_resource idx_r = ctx_stage_tensor_uav(ctx, src1);
    const tensor_resource dst_r = ctx_stage_tensor_uav(ctx, node);

    if (memops_env_log_set_rows()) {
        GGML_LOG_INFO("SET_ROWS valid=%d/%d/%d src0[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d  "
            "src1[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d  "
            "dst[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d  "
            "src_off=%zu idx_off=%zu dst_off=%zu\n",
            (int)src_r.valid, (int)idx_r.valid, (int)dst_r.valid,
            (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], (int)src0->type,
            (long long)src1->ne[0], (long long)src1->ne[1], (long long)src1->ne[2], (long long)src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], (int)src1->type,
            (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
            node->nb[0], node->nb[1], node->nb[2], node->nb[3], (int)node->type,
            (size_t)src_r.offset_bytes, (size_t)idx_r.offset_bytes, (size_t)dst_r.offset_bytes);
    }

    if (!src_r.valid || !idx_r.valid || !dst_r.valid) return true;

    const uint64_t nc   = static_cast<uint64_t>(src0->ne[0]);
    const uint64_t nr   = static_cast<uint64_t>(src0->ne[1]);
    const uint64_t ne02 = static_cast<uint64_t>(src0->ne[2]);
    const uint64_t ne03 = static_cast<uint64_t>(src0->ne[3]);
    const uint64_t ne11 = static_cast<uint64_t>(src1->ne[1]);
    const uint64_t ne12 = static_cast<uint64_t>(src1->ne[2]);

    if (nc == 0 || nr == 0 || ne02 == 0 || ne03 == 0) return true; // empty
    if (ne11 == 0 || ne12 == 0) return true;

    const uint64_t total_rows = nr * ne02 * ne03;
    if (total_rows > 0xFFFFFFFFull) return true;

    if (src_r.offset_bytes > 0xFFFFFFFFull ||
        idx_r.offset_bytes > 0xFFFFFFFFull ||
        dst_r.offset_bytes > 0xFFFFFFFFull) {
        return true;
    }
    if (src0->nb[1] > 0xFFFFFFFFull || src0->nb[2] > 0xFFFFFFFFull || src0->nb[3] > 0xFFFFFFFFull) return true;
    if (node->nb[1] > 0xFFFFFFFFull || node->nb[2] > 0xFFFFFFFFull || node->nb[3] > 0xFFFFFFFFull) return true;
    if (src1->nb[0] > 0xFFFFFFFFull || src1->nb[1] > 0xFFFFFFFFull || src1->nb[2] > 0xFFFFFFFFull) return true;
    if (nc > 0xFFFFFFFFull || nr > 0xFFFFFFFFull || ne02 > 0xFFFFFFFFull || ne03 > 0xFFFFFFFFull) return true;
    if (ne11 > 0xFFFFFFFFull || ne12 > 0xFFFFFFFFull) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const tensor_resource uavs[3] = { src_r, idx_r, dst_r };
    if (!ctx_bind_raw_uavs_resolved(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, /*uav=*/3, /*dwords=*/20);
    if (root_sig == nullptr) return true;

    ID3D12PipelineState * pso = ctx.psos->get("set_rows_f32_fp16", root_sig, {});
    if (pso == nullptr) return true;

    const bool dst_is_f16 = (node->type == GGML_TYPE_F16);
    const bool idx_is_i64 = (src1->type == GGML_TYPE_I64);
    const UINT flags = (dst_is_f16 ? 1u : 0u) | (idx_is_i64 ? 2u : 0u);

    const UINT consts[20] = {
        static_cast<UINT>(nc),
        static_cast<UINT>(nr),
        static_cast<UINT>(ne02),
        static_cast<UINT>(ne03),
        static_cast<UINT>(ne11),
        static_cast<UINT>(ne12),
        static_cast<UINT>(src_r.offset_bytes),
        static_cast<UINT>(dst_r.offset_bytes),
        static_cast<UINT>(idx_r.offset_bytes),
        static_cast<UINT>(src0->nb[1]),
        static_cast<UINT>(src0->nb[2]),
        static_cast<UINT>(src0->nb[3]),
        static_cast<UINT>(node->nb[1]),
        static_cast<UINT>(node->nb[2]),
        static_cast<UINT>(node->nb[3]),
        static_cast<UINT>(src1->nb[0]),
        static_cast<UINT>(src1->nb[1]),
        static_cast<UINT>(src1->nb[2]),
        flags,
        0,
    };

    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, /*uav=*/3, consts, /*dwords=*/20)) {
        return true;
    }

    ctx_dispatch_groups(ctx, static_cast<UINT>(total_rows), 1, 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
