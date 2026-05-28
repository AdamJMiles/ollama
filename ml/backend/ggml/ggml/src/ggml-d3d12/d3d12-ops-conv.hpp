#pragma once

#include <cstdint>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool conv_u32(uint64_t v) {
    return v <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool conv_i32_nonnegative(int64_t v) {
    return v >= 0 && v <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

inline bool conv_addressable(size_t offset, uint64_t bytes) {
    constexpr uint64_t u32_range = uint64_t{1} << 32;
    return offset <= std::numeric_limits<UINT>::max() && bytes <= u32_range &&
           bytes <= u32_range - static_cast<uint64_t>(offset);
}

inline bool conv_shape_positive(const ggml_tensor * t) {
    return t != nullptr && t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] > 0 && t->ne[3] > 0;
}

inline bool supports_im2col_2d_f32_src(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr) return false;
    if (op->op != GGML_OP_IM2COL) return false;

    const int32_t * params = reinterpret_cast<const int32_t *>(op->op_params);
    if (params[6] != 1) return false;

    const ggml_tensor * kernel = op->src[0];
    const ggml_tensor * src = op->src[1];
    if (!conv_shape_positive(kernel) || !conv_shape_positive(src) || !conv_shape_positive(op)) return false;
    if (src->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) return false;
    if (op->type == GGML_TYPE_F16 && kernel->type != GGML_TYPE_F16) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;

    if (kernel->ne[2] != src->ne[2]) return false;
    if (op->ne[0] != kernel->ne[0] * kernel->ne[1] * kernel->ne[2]) return false;
    if (op->ne[3] != src->ne[3]) return false;
    return params[0] > 0 && params[1] > 0 && params[4] > 0 && params[5] > 0;
}

inline bool supports_pool_2d_f32(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;
    if (op->op != GGML_OP_POOL_2D) return false;

    const ggml_tensor * src = op->src[0];
    if (!conv_shape_positive(src) || !conv_shape_positive(op)) return false;
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    if (op->ne[2] != src->ne[2] || op->ne[3] != src->ne[3]) return false;

    const int32_t * params = reinterpret_cast<const int32_t *>(op->op_params);
    if (params[0] != GGML_OP_POOL_MAX && params[0] != GGML_OP_POOL_AVG) return false;
    return params[1] > 0 && params[2] > 0 && params[3] > 0 && params[4] > 0;
}

inline bool supports_op_conv(const ggml_tensor * op) {
    if (op == nullptr) return false;

    switch (op->op) {
        case GGML_OP_IM2COL:
            return supports_im2col_2d_f32_src(op);
        case GGML_OP_POOL_2D:
            return supports_pool_2d_f32(op);
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D_BACK:
        case GGML_OP_UPSCALE:
        default:
            return false;
    }
}

inline bool dispatch_im2col_2d(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * kernel = node->src[0];
    const ggml_tensor * src = node->src[1];

    const uint64_t total = static_cast<uint64_t>(ggml_nelements(node));
    if (total == 0 || !conv_u32(total)) return true;

    const bool dst_f16 = node->type == GGML_TYPE_F16;
    const uint64_t work_items = dst_f16 ? ((total + 1u) / 2u) : total;
    if (!conv_u32(work_items)) return true;

    const tensor_resource sr = ctx_resolve_tensor(ctx, src);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!sr.valid || !dr.valid) return true;
    if ((sr.offset_bytes % 4) != 0 || (dr.offset_bytes % 4) != 0) return true;

    const uint64_t src_bytes = static_cast<uint64_t>(ggml_nbytes(src));
    const uint64_t dst_bytes = dst_f16 ? (((total + 1u) / 2u) * 4u) : static_cast<uint64_t>(ggml_nbytes(node));
    if (!conv_addressable(sr.offset_bytes, src_bytes) || !conv_addressable(dr.offset_bytes, dst_bytes)) return true;
    if (sr.offset_bytes > sr.buffer_size_bytes || dr.offset_bytes > dr.buffer_size_bytes) return true;
    if (src_bytes > static_cast<uint64_t>(sr.buffer_size_bytes) - static_cast<uint64_t>(sr.offset_bytes)) return true;
    if (dst_bytes > static_cast<uint64_t>(dr.buffer_size_bytes) - static_cast<uint64_t>(dr.offset_bytes)) return true;

    const int32_t * params = reinterpret_cast<const int32_t *>(node->op_params);
    const int64_t dims[] = {
        src->ne[0], src->ne[1], src->ne[2], kernel->ne[0], kernel->ne[1], node->ne[1], node->ne[2], node->ne[0]
    };
    for (int64_t d : dims) {
        if (!conv_i32_nonnegative(d)) return true;
    }

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 18);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get("im2col_f32", rs, {});
    if (pso == nullptr) return true;

    const UINT consts[18] = {
        static_cast<UINT>(total),
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        dst_f16 ? 1u : 0u,
        static_cast<UINT>(src->ne[0]),
        static_cast<UINT>(src->ne[1]),
        static_cast<UINT>(src->ne[2]),
        static_cast<UINT>(kernel->ne[0]),
        static_cast<UINT>(kernel->ne[1]),
        static_cast<UINT>(node->ne[1]),
        static_cast<UINT>(node->ne[2]),
        static_cast<UINT>(node->ne[0]),
        static_cast<UINT>(params[0]),
        static_cast<UINT>(params[1]),
        static_cast<UINT>(params[2]),
        static_cast<UINT>(params[3]),
        static_cast<UINT>(params[4]),
        static_cast<UINT>(params[5]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 18)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(work_items), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

inline bool dispatch_pool_2d(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const uint64_t total = static_cast<uint64_t>(ggml_nelements(node));
    if (total == 0 || !conv_u32(total)) return true;

    const tensor_resource sr = ctx_resolve_tensor(ctx, src);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!sr.valid || !dr.valid) return true;
    if ((sr.offset_bytes % 4) != 0 || (dr.offset_bytes % 4) != 0) return true;
    const uint64_t src_bytes = static_cast<uint64_t>(ggml_nbytes(src));
    const uint64_t dst_bytes = static_cast<uint64_t>(ggml_nbytes(node));
    if (!conv_addressable(sr.offset_bytes, src_bytes) || !conv_addressable(dr.offset_bytes, dst_bytes)) return true;
    if (sr.offset_bytes > sr.buffer_size_bytes || dr.offset_bytes > dr.buffer_size_bytes) return true;
    if (src_bytes > static_cast<uint64_t>(sr.buffer_size_bytes) - static_cast<uint64_t>(sr.offset_bytes)) return true;
    if (dst_bytes > static_cast<uint64_t>(dr.buffer_size_bytes) - static_cast<uint64_t>(dr.offset_bytes)) return true;

    const int32_t * params = reinterpret_cast<const int32_t *>(node->op_params);
    const int64_t dims[] = { src->ne[0], src->ne[1], src->ne[2], node->ne[0], node->ne[1] };
    for (int64_t d : dims) {
        if (!conv_i32_nonnegative(d)) return true;
    }

    if (!ctx_transition(ctx, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[2] = { src, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 2, &uav_table)) return true;

    const char * shader = params[0] == GGML_OP_POOL_MAX ? "pool_2d_max_f32" : "pool_2d_avg_f32";
    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 2, 14);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (pso == nullptr) return true;

    const UINT consts[14] = {
        static_cast<UINT>(total),
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        static_cast<UINT>(src->ne[0]),
        static_cast<UINT>(src->ne[1]),
        static_cast<UINT>(node->ne[0]),
        static_cast<UINT>(node->ne[1]),
        static_cast<UINT>(src->ne[2]),
        static_cast<UINT>(params[1]),
        static_cast<UINT>(params[2]),
        static_cast<UINT>(params[3]),
        static_cast<UINT>(params[4]),
        static_cast<UINT>(params[5]),
        static_cast<UINT>(params[6]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 2, consts, 14)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(total), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

inline bool dispatch_conv(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_conv(node)) return false;

    switch (node->op) {
        case GGML_OP_IM2COL:
            return dispatch_im2col_2d(ctx, node);
        case GGML_OP_POOL_2D:
            return dispatch_pool_2d(ctx, node);
        default:
            return false;
    }
}

} // namespace ggml_d3d12
