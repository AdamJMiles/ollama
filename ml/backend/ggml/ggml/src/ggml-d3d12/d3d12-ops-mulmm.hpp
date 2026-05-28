#pragma once

#include <cstdint>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool mulmm_u32(uint64_t v) {
    return v <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool mulmm_dispatch_dim(uint64_t v) {
    return v > 0 && v <= 65535ull;
}

inline bool mulmm_addressable(size_t offset, uint64_t bytes) {
    constexpr uint64_t u32_range = uint64_t{1} << 32;
    return offset <= std::numeric_limits<UINT>::max() && bytes <= u32_range &&
           bytes <= u32_range - static_cast<uint64_t>(offset);
}

inline bool mulmm_shape_positive(const ggml_tensor * t) {
    return t != nullptr && t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] > 0 && t->ne[3] > 0;
}

inline bool mulmm_row_contiguous(const ggml_tensor * t) {
    if (!mulmm_shape_positive(t)) return false;
    const size_t es = ggml_type_size(t->type);
    return t->nb[0] == es && t->nb[1] == es * static_cast<size_t>(t->ne[0]);
}

inline bool supports_op_mulmm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) return false;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!mulmm_shape_positive(src0) || !mulmm_shape_positive(src1) || !mulmm_shape_positive(op)) return false;

    if (src1->ne[1] < 2) return false;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) return false;
    if (op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) return false;
    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) return false;

    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) return false;
    if (!ggml_is_contiguous(src0) && !mulmm_row_contiguous(src0)) return false;
    if (!mulmm_row_contiguous(src0)) return false;

    const uint64_t gx = (static_cast<uint64_t>(op->ne[1]) + 31u) / 32u;
    const uint64_t gy = (static_cast<uint64_t>(op->ne[0]) + 31u) / 32u;
    const uint64_t gz = static_cast<uint64_t>(src1->ne[2]) * static_cast<uint64_t>(src1->ne[3]);
    return mulmm_dispatch_dim(gx) && mulmm_dispatch_dim(gy) && mulmm_dispatch_dim(gz);
}

inline bool dispatch_mulmm(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_mulmm(node)) return false;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const char * shader = src0->type == GGML_TYPE_F16 ? "mul_mm_f16_f32" : "mul_mm_f32_f32";

    const uint64_t M = static_cast<uint64_t>(node->ne[0]);
    const uint64_t N = static_cast<uint64_t>(node->ne[1]);
    const uint64_t K = static_cast<uint64_t>(src0->ne[0]);
    const uint64_t batch_ne2 = static_cast<uint64_t>(src1->ne[2]);
    const uint64_t batch = batch_ne2 * static_cast<uint64_t>(src1->ne[3]);
    const uint64_t gx = (N + 31u) / 32u;
    const uint64_t gy = (M + 31u) / 32u;
    const uint64_t gz = batch;
    if (!mulmm_dispatch_dim(gx) || !mulmm_dispatch_dim(gy) || !mulmm_dispatch_dim(gz)) return true;

    const tensor_resource s0r = ctx_resolve_tensor(ctx, src0);
    const tensor_resource s1r = ctx_resolve_tensor(ctx, src1);
    const tensor_resource dr  = ctx_resolve_tensor(ctx, node);
    if (!s0r.valid || !s1r.valid || !dr.valid) return true;

    const uint64_t src0_bytes = static_cast<uint64_t>(ggml_nbytes(src0));
    const uint64_t src1_bytes = static_cast<uint64_t>(ggml_nbytes(src1));
    const uint64_t dst_bytes  = static_cast<uint64_t>(ggml_nbytes(node));
    if (!mulmm_addressable(s0r.offset_bytes, src0_bytes) ||
        !mulmm_addressable(s1r.offset_bytes, src1_bytes) ||
        !mulmm_addressable(dr.offset_bytes,  dst_bytes)) {
        return true;
    }
    if (s0r.offset_bytes > s0r.buffer_size_bytes || s1r.offset_bytes > s1r.buffer_size_bytes || dr.offset_bytes > dr.buffer_size_bytes) return true;
    if (src0_bytes > static_cast<uint64_t>(s0r.buffer_size_bytes - s0r.offset_bytes) ||
        src1_bytes > static_cast<uint64_t>(s1r.buffer_size_bytes - s1r.offset_bytes) ||
        dst_bytes  > static_cast<uint64_t>(dr.buffer_size_bytes  - dr.offset_bytes)) {
        return true;
    }

    const size_t src0_es = ggml_type_size(src0->type);
    if ((s0r.offset_bytes % src0_es) != 0 || (src0->nb[2] % src0_es) != 0 || (src0->nb[3] % src0_es) != 0) return true;
    if ((s1r.offset_bytes % sizeof(float)) != 0 || (dr.offset_bytes % sizeof(float)) != 0) return true;

    const uint64_t dims_and_strides[] = {
        M, N, K, batch_ne2,
        static_cast<uint64_t>(s0r.offset_bytes), static_cast<uint64_t>(s1r.offset_bytes), static_cast<uint64_t>(dr.offset_bytes),
        static_cast<uint64_t>(src0->nb[1]), static_cast<uint64_t>(src0->nb[2]), static_cast<uint64_t>(src0->nb[3]),
    };
    for (uint64_t v : dims_and_strides) {
        if (!mulmm_u32(v)) return true;
    }

    if (!ctx_transition(ctx, src0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, src1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { src0, src1, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 3, 10);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (pso == nullptr) return true;

    const UINT consts[10] = {
        static_cast<UINT>(M),
        static_cast<UINT>(N),
        static_cast<UINT>(K),
        static_cast<UINT>(batch_ne2),
        static_cast<UINT>(s0r.offset_bytes),
        static_cast<UINT>(s1r.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        static_cast<UINT>(src0->nb[1]),
        static_cast<UINT>(src0->nb[2]),
        static_cast<UINT>(src0->nb[3]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 3, consts, 10)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(gx), static_cast<UINT>(gy), static_cast<UINT>(gz));
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
