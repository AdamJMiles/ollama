#pragma once

#include <cstdint>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool ssm_u32(uint64_t v) {
    return v <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool ssm_positive_shape(const ggml_tensor * t) {
    return t != nullptr && t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] > 0 && t->ne[3] > 0;
}

inline bool ssm_f32_contiguous(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

inline bool ssm_resource_fits_u32(const tensor_resource & r, const ggml_tensor * t) {
    if (!r.valid || t == nullptr) return false;
    constexpr uint64_t u32_range = uint64_t{1} << 32;
    const uint64_t offset = static_cast<uint64_t>(r.offset_bytes);
    const uint64_t bytes  = static_cast<uint64_t>(ggml_nbytes(t));
    if (offset >= u32_range || bytes > u32_range || bytes > u32_range - offset) return false;
    if (offset > static_cast<uint64_t>(r.buffer_size_bytes)) return false;
    if (bytes > static_cast<uint64_t>(r.buffer_size_bytes) - offset) return false;
    return (offset % 4u) == 0;
}

inline bool supports_ssm_conv_f32(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr) return false;
    if (op->op != GGML_OP_SSM_CONV) return false;

    const ggml_tensor * sx = op->src[0];
    const ggml_tensor * c  = op->src[1];
    if (!ssm_f32_contiguous(op) || !ssm_f32_contiguous(sx) || !ssm_f32_contiguous(c)) return false;
    if (!ssm_positive_shape(op) || !ssm_positive_shape(sx) || !ssm_positive_shape(c)) return false;

    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    const int64_t n_t     = op->ne[1];
    const int64_t n_s     = op->ne[2];

    if (sx->ne[0] != d_conv - 1 + n_t) return false;
    if (sx->ne[1] != d_inner || sx->ne[2] != n_s || sx->ne[3] != 1) return false;
    if (op->ne[0] != d_inner || op->ne[3] != 1) return false;
    if (c->ne[2] != 1 || c->ne[3] != 1) return false;
    return true;
}

inline bool supports_ssm_scan_f32(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SSM_SCAN || op->type != GGML_TYPE_F32) return false;
    for (int i = 0; i < 7; ++i) {
        if (op->src[i] == nullptr) return false;
    }

    const ggml_tensor * s   = op->src[0];
    const ggml_tensor * x   = op->src[1];
    const ggml_tensor * dt  = op->src[2];
    const ggml_tensor * A   = op->src[3];
    const ggml_tensor * B   = op->src[4];
    const ggml_tensor * C   = op->src[5];
    const ggml_tensor * ids = op->src[6];

    if (!ssm_f32_contiguous(op) || !ssm_f32_contiguous(s) || !ssm_f32_contiguous(x) ||
        !ssm_f32_contiguous(dt) || !ssm_f32_contiguous(A) || !ssm_f32_contiguous(B) ||
        !ssm_f32_contiguous(C)) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids)) return false;
    if (!ssm_positive_shape(s) || !ssm_positive_shape(x) || !ssm_positive_shape(dt) ||
        !ssm_positive_shape(A) || !ssm_positive_shape(B) || !ssm_positive_shape(C) ||
        !ssm_positive_shape(ids)) {
        return false;
    }

    const int64_t d_state  = s->ne[0];
    const int64_t head_dim = s->ne[1];
    const int64_t n_head   = x->ne[1];
    const int64_t n_tok    = x->ne[2];
    const int64_t n_seq    = x->ne[3];
    const int64_t n_group  = B->ne[1];

    if (d_state > 256) return false; // one 256-thread group maps directly to the state dimension
    if (x->ne[0] != head_dim) return false;
    if (s->ne[2] != n_head || s->ne[3] < n_seq) return false;
    if (dt->ne[0] != n_head || dt->ne[1] != n_tok || dt->ne[2] != n_seq || dt->ne[3] != 1) return false;
    if (A->ne[0] != 1) return false; // first cut: Mamba-2 scalar A per head; reject Mamba-1 state-wise A
    if (A->ne[1] != n_head || A->ne[2] != 1 || A->ne[3] != 1) return false;
    if (!ggml_are_same_shape(B, C)) return false;
    if (B->ne[0] != d_state || B->ne[2] != n_tok || B->ne[3] != n_seq) return false;
    if (n_group <= 0 || n_head % n_group != 0) return false;
    if (ids->ne[0] != n_seq || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) return false;

    const uint64_t x_elems = static_cast<uint64_t>(ggml_nelements(x));
    const uint64_t state_elems = static_cast<uint64_t>(d_state) * static_cast<uint64_t>(head_dim) *
                                 static_cast<uint64_t>(n_head) * static_cast<uint64_t>(n_seq);
    if (x_elems > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - state_elems) return false;
    const uint64_t expected = x_elems + state_elems;
    if (static_cast<uint64_t>(ggml_nelements(op)) != expected) return false;
    return op->ne[0] == static_cast<int64_t>(expected) && op->ne[1] == 1 && op->ne[2] == 1 && op->ne[3] == 1;
}

inline bool supports_op_ssm(const ggml_tensor * op) {
    if (op == nullptr) return false;
    switch (op->op) {
        case GGML_OP_SSM_CONV:
            return supports_ssm_conv_f32(op);
        case GGML_OP_SSM_SCAN:
            return supports_ssm_scan_f32(op);
        default:
            return false;
    }
}

inline bool dispatch_ssm_conv(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * sx = node->src[0];
    const ggml_tensor * c  = node->src[1];

    const tensor_resource sr = ctx_resolve_tensor(ctx, sx);
    const tensor_resource cr = ctx_resolve_tensor(ctx, c);
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!ssm_resource_fits_u32(sr, sx) || !ssm_resource_fits_u32(cr, c) || !ssm_resource_fits_u32(dr, node)) return true;

    const uint64_t count = static_cast<uint64_t>(ggml_nelements(node));
    if (count == 0 || !ssm_u32(count)) return true;
    if (!ssm_u32(static_cast<uint64_t>(c->ne[0])) || !ssm_u32(static_cast<uint64_t>(sx->ne[0])) ||
        !ssm_u32(static_cast<uint64_t>(node->ne[0])) || !ssm_u32(static_cast<uint64_t>(node->ne[1]))) {
        return true;
    }

    if (!ctx_transition(ctx, sx,   D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, c,    D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { sx, c, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 3, 8);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get("ssm_conv_f32", rs, {});
    if (pso == nullptr) return true;

    const UINT consts[8] = {
        static_cast<UINT>(count),
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(cr.offset_bytes),
        static_cast<UINT>(dr.offset_bytes),
        static_cast<UINT>(c->ne[0]),
        static_cast<UINT>(sx->ne[0]),
        static_cast<UINT>(node->ne[0]),
        static_cast<UINT>(node->ne[1]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 3, consts, 8)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(count), 64);
    ctx_uav_barrier(ctx, node);
    return true;
}

inline bool dispatch_ssm_scan(dispatch_ctx & ctx, const ggml_tensor * node) {
    const ggml_tensor * s   = node->src[0];
    const ggml_tensor * x   = node->src[1];
    const ggml_tensor * dt  = node->src[2];
    const ggml_tensor * A   = node->src[3];
    const ggml_tensor * B   = node->src[4];
    const ggml_tensor * C   = node->src[5];
    const ggml_tensor * ids = node->src[6];

    const tensor_resource sr  = ctx_resolve_tensor(ctx, s);
    const tensor_resource xr  = ctx_resolve_tensor(ctx, x);
    const tensor_resource dtr = ctx_resolve_tensor(ctx, dt);
    const tensor_resource ar  = ctx_resolve_tensor(ctx, A);
    const tensor_resource br  = ctx_resolve_tensor(ctx, B);
    const tensor_resource cr  = ctx_resolve_tensor(ctx, C);
    const tensor_resource ir  = ctx_resolve_tensor(ctx, ids);
    const tensor_resource orr = ctx_resolve_tensor(ctx, node);
    if (!ssm_resource_fits_u32(sr, s) || !ssm_resource_fits_u32(xr, x) || !ssm_resource_fits_u32(dtr, dt) ||
        !ssm_resource_fits_u32(ar, A) || !ssm_resource_fits_u32(br, B) || !ssm_resource_fits_u32(cr, C) ||
        !ssm_resource_fits_u32(ir, ids) || !ssm_resource_fits_u32(orr, node)) {
        return true;
    }

    const uint64_t state_off_bytes = static_cast<uint64_t>(ggml_nelements(x)) * sizeof(float);
    if (!ssm_u32(state_off_bytes)) return true;

    const uint64_t dims[] = {
        static_cast<uint64_t>(s->ne[0]), static_cast<uint64_t>(s->ne[1]), static_cast<uint64_t>(x->ne[1]),
        static_cast<uint64_t>(B->ne[1]), static_cast<uint64_t>(x->ne[2]), static_cast<uint64_t>(x->ne[3])
    };
    for (uint64_t d : dims) {
        if (!ssm_u32(d)) return true;
    }

    const ggml_tensor * uavs[8] = { s, x, dt, A, B, C, ids, node };
    for (const ggml_tensor * t : uavs) {
        if (!ctx_transition(ctx, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    if (!ctx_bind_raw_uavs(ctx, uavs, 8, &uav_table)) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 8, 15);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get("ssm_scan_f32", rs, {});
    if (pso == nullptr) return true;

    const UINT consts[15] = {
        static_cast<UINT>(sr.offset_bytes),
        static_cast<UINT>(xr.offset_bytes),
        static_cast<UINT>(dtr.offset_bytes),
        static_cast<UINT>(ar.offset_bytes),
        static_cast<UINT>(br.offset_bytes),
        static_cast<UINT>(cr.offset_bytes),
        static_cast<UINT>(ir.offset_bytes),
        static_cast<UINT>(orr.offset_bytes),
        static_cast<UINT>(s->ne[0]),
        static_cast<UINT>(s->ne[1]),
        static_cast<UINT>(x->ne[1]),
        static_cast<UINT>(B->ne[1]),
        static_cast<UINT>(x->ne[2]),
        static_cast<UINT>(x->ne[3]),
        static_cast<UINT>(state_off_bytes),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 8, consts, 15)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(s->ne[1]), static_cast<UINT>(x->ne[1]), static_cast<UINT>(x->ne[3]));
    ctx_uav_barrier(ctx, node);
    return true;
}

inline bool dispatch_ssm(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_ssm(node)) return false;
    switch (node->op) {
        case GGML_OP_SSM_CONV:
            return dispatch_ssm_conv(ctx, node);
        case GGML_OP_SSM_SCAN:
            return dispatch_ssm_scan(ctx, node);
        default:
            return false;
    }
}

} // namespace ggml_d3d12
