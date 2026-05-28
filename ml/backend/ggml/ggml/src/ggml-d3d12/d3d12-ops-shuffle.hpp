#pragma once

#include <cstddef>
#include <cstdint>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool shuffle_u64_fits_u32(uint64_t v) {
    return v <= 0xFFFFFFFFull;
}

inline bool shuffle_i64_dim_fits_u32(int64_t v) {
    return v > 0 && static_cast<uint64_t>(v) <= 0xFFFFFFFFull;
}

inline bool shuffle_size_fits_u32(size_t v) {
    return static_cast<uint64_t>(v) <= 0xFFFFFFFFull;
}

inline bool shuffle_dims_fit_u32(const ggml_tensor * t) {
    return t != nullptr &&
        shuffle_i64_dim_fits_u32(t->ne[0]) &&
        shuffle_i64_dim_fits_u32(t->ne[1]) &&
        shuffle_i64_dim_fits_u32(t->ne[2]) &&
        shuffle_i64_dim_fits_u32(t->ne[3]);
}

inline bool shuffle_strides_fit_u32_aligned(const ggml_tensor * t) {
    if (t == nullptr) return false;
    for (int i = 0; i < 4; ++i) {
        if (!shuffle_size_fits_u32(t->nb[i]) || (t->nb[i] % 4) != 0) return false;
    }
    return true;
}

inline bool shuffle_resource_offset_ok(const tensor_resource & r) {
    return r.valid && shuffle_size_fits_u32(r.offset_bytes) && (r.offset_bytes % 4) == 0;
}

inline bool shuffle_tensor_span_fits_u32(size_t base, const ggml_tensor * t) {
    if (!shuffle_size_fits_u32(base) || !shuffle_dims_fit_u32(t) || !shuffle_strides_fit_u32_aligned(t)) return false;

    uint64_t max_byte = static_cast<uint64_t>(base);
    for (int i = 0; i < 4; ++i) {
        const uint64_t ne = static_cast<uint64_t>(t->ne[i]);
        const uint64_t nb = static_cast<uint64_t>(t->nb[i]);
        const uint64_t add = (ne - 1ull) * nb;
        if (add > 0xFFFFFFFFull - max_byte) return false;
        max_byte += add;
    }
    return max_byte <= 0xFFFFFFFFull - 3ull;
}

inline bool shuffle_linear_f32_span_fits_u32(size_t base, uint64_t count) {
    if (!shuffle_size_fits_u32(base)) return false;
    if (count == 0) return true;
    const uint64_t start = static_cast<uint64_t>(base);
    const uint64_t bytes_to_last = (count - 1ull) * 4ull + 3ull;
    return bytes_to_last <= 0xFFFFFFFFull - start;
}

inline bool supports_op_shuffle(const ggml_tensor * op) {
    if (op == nullptr) return false;

    switch (op->op) {
        case GGML_OP_CONT: {
            const ggml_tensor * src = op->src[0];
            if (src == nullptr) return false;
            if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op)) return false;
            if (!ggml_are_same_shape(op, src)) return false;
            return shuffle_dims_fit_u32(op);
        }
        case GGML_OP_REPEAT: {
            const ggml_tensor * src = op->src[0];
            if (src == nullptr) return false;
            if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op)) return false;
            if (!shuffle_dims_fit_u32(src) || !shuffle_dims_fit_u32(op)) return false;
            return ggml_can_repeat(src, op);
        }
        case GGML_OP_PAD: {
            const ggml_tensor * src = op->src[0];
            if (src == nullptr) return false;
            if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op)) return false;
            if (!shuffle_dims_fit_u32(src) || !shuffle_dims_fit_u32(op)) return false;
            if (op->op_params[8] != 0) return false;

            for (int i = 0; i < 4; ++i) {
                const int32_t lp = op->op_params[2*i + 0];
                const int32_t rp = op->op_params[2*i + 1];
                if (lp < 0 || rp < 0) return false;
                if (src->ne[i] + static_cast<int64_t>(lp) + static_cast<int64_t>(rp) != op->ne[i]) return false;
            }
            return true;
        }
        case GGML_OP_GET_ROWS: {
            const ggml_tensor * src = op->src[0];
            const ggml_tensor * idx = op->src[1];
            if (src == nullptr || idx == nullptr) return false;
            if (src->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op)) return false;
            if (!shuffle_dims_fit_u32(src) || !shuffle_dims_fit_u32(idx) || !shuffle_dims_fit_u32(op)) return false;
            if (src->ne[2] != idx->ne[1] || src->ne[3] != idx->ne[2] || idx->ne[3] != 1) return false;
            return op->ne[0] == src->ne[0] &&
                   op->ne[1] == idx->ne[0] &&
                   op->ne[2] == idx->ne[1] &&
                   op->ne[3] == idx->ne[2];
        }
        default:
            return false;
    }
}

inline bool dispatch_shuffle(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_shuffle(node)) return false;

    const int64_t count_i64 = ggml_nelements(node);
    if (count_i64 <= 0) return true;
    const uint64_t count = static_cast<uint64_t>(count_i64);
    if (!shuffle_u64_fits_u32(count)) return true;

    const ggml_tensor * src = node->src[0];
    const tensor_resource sr = ctx_resolve_tensor(ctx, src);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!shuffle_resource_offset_ok(sr) || !shuffle_resource_offset_ok(dr)) return true;
    if (!shuffle_tensor_span_fits_u32(sr.offset_bytes, src)) return true;
    if (!shuffle_linear_f32_span_fits_u32(dr.offset_bytes, count)) return true;

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    if (node->op == GGML_OP_CONT) {
        D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
        const ggml_tensor * uavs[2] = { src, node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 11);
        if (rs == nullptr) return true;
        ID3D12PipelineState * pso = ctx.psos->get("cont_f32", rs, {});
        if (pso == nullptr) return true;

        const UINT consts[11] = {
            static_cast<UINT>(count),
            static_cast<UINT>(sr.offset_bytes),
            static_cast<UINT>(dr.offset_bytes),
            static_cast<UINT>(node->ne[0]),
            static_cast<UINT>(node->ne[1]),
            static_cast<UINT>(node->ne[2]),
            static_cast<UINT>(node->ne[3]),
            static_cast<UINT>(src->nb[0]),
            static_cast<UINT>(src->nb[1]),
            static_cast<UINT>(src->nb[2]),
            static_cast<UINT>(src->nb[3]),
        };
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 11)) return true;

        ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
        ctx_uav_barrier(ctx, node);
        return true;
    }

    if (node->op == GGML_OP_REPEAT) {
        D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
        const ggml_tensor * uavs[2] = { src, node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 15);
        if (rs == nullptr) return true;
        ID3D12PipelineState * pso = ctx.psos->get("repeat_f32", rs, {});
        if (pso == nullptr) return true;

        const UINT consts[15] = {
            static_cast<UINT>(count),
            static_cast<UINT>(sr.offset_bytes),
            static_cast<UINT>(dr.offset_bytes),
            static_cast<UINT>(node->ne[0]),
            static_cast<UINT>(node->ne[1]),
            static_cast<UINT>(node->ne[2]),
            static_cast<UINT>(node->ne[3]),
            static_cast<UINT>(src->ne[0]),
            static_cast<UINT>(src->ne[1]),
            static_cast<UINT>(src->ne[2]),
            static_cast<UINT>(src->ne[3]),
            static_cast<UINT>(src->nb[0]),
            static_cast<UINT>(src->nb[1]),
            static_cast<UINT>(src->nb[2]),
            static_cast<UINT>(src->nb[3]),
        };
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 15)) return true;

        ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
        ctx_uav_barrier(ctx, node);
        return true;
    }

    if (node->op == GGML_OP_PAD) {
        D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
        const ggml_tensor * uavs[2] = { src, node };
        if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

        ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 19);
        if (rs == nullptr) return true;
        ID3D12PipelineState * pso = ctx.psos->get("pad_f32", rs, {});
        if (pso == nullptr) return true;

        const UINT consts[19] = {
            static_cast<UINT>(count),
            static_cast<UINT>(sr.offset_bytes),
            static_cast<UINT>(dr.offset_bytes),
            static_cast<UINT>(node->ne[0]),
            static_cast<UINT>(node->ne[1]),
            static_cast<UINT>(node->ne[2]),
            static_cast<UINT>(node->ne[3]),
            static_cast<UINT>(src->ne[0]),
            static_cast<UINT>(src->ne[1]),
            static_cast<UINT>(src->ne[2]),
            static_cast<UINT>(src->ne[3]),
            static_cast<UINT>(src->nb[0]),
            static_cast<UINT>(src->nb[1]),
            static_cast<UINT>(src->nb[2]),
            static_cast<UINT>(src->nb[3]),
            static_cast<UINT>(node->op_params[0]),
            static_cast<UINT>(node->op_params[2]),
            static_cast<UINT>(node->op_params[4]),
            static_cast<UINT>(node->op_params[6]),
        };
        if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 19)) return true;

        ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
        ctx_uav_barrier(ctx, node);
        return true;
    }

    const ggml_tensor * idx = node->src[1];
    const tensor_resource ir = ctx_resolve_tensor(ctx, idx);
    if (!shuffle_resource_offset_ok(ir)) return true;
    if (!shuffle_tensor_span_fits_u32(ir.offset_bytes, idx)) return true;
    if (!shuffle_strides_fit_u32_aligned(idx)) return true;
    if (src->ne[1] > 2147483647ll) return true;

    if (!ctx_transition(ctx, idx, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { src, idx, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 3, 16);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get("get_rows_f32", rs, {});
    if (pso == nullptr) return true;

    const UINT consts[16] = {
        static_cast<UINT>(count),
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(ir.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        static_cast<UINT>(src->ne[0]),
        static_cast<UINT>(src->ne[1]),
        static_cast<UINT>(idx->ne[0]),
        static_cast<UINT>(idx->ne[1]),
        static_cast<UINT>(idx->ne[2]),
        static_cast<UINT>(src->nb[0]),
        static_cast<UINT>(src->nb[1]),
        static_cast<UINT>(src->nb[2]),
        static_cast<UINT>(src->nb[3]),
        static_cast<UINT>(idx->nb[0]),
        static_cast<UINT>(idx->nb[1]),
        static_cast<UINT>(idx->nb[2]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 3, consts, 16)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(count), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
