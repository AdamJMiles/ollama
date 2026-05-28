#pragma once

#include <cstdint>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool dequant_is_legacy_quant_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool dequant_is_kquant_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool dequant_is_supported_quant_type(ggml_type type) {
    return dequant_is_legacy_quant_type(type) || dequant_is_kquant_type(type);
}

inline const char * dequant_shader_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return "get_rows_q4_0";
        case GGML_TYPE_Q4_1: return "get_rows_q4_1";
        case GGML_TYPE_Q5_0: return "get_rows_q5_0";
        case GGML_TYPE_Q5_1: return "get_rows_q5_1";
        case GGML_TYPE_Q8_0: return "get_rows_q8_0";
        case GGML_TYPE_Q2_K: return "get_rows_q2_K";
        case GGML_TYPE_Q3_K: return "get_rows_q3_K";
        case GGML_TYPE_Q4_K: return "get_rows_q4_K";
        case GGML_TYPE_Q5_K: return "get_rows_q5_K";
        case GGML_TYPE_Q6_K: return "get_rows_q6_K";
        default: return nullptr;
    }
}

inline bool supports_op_dequant(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) return false;

    const ggml_tensor * src0 = op->src[0];
    if (!dequant_is_supported_quant_type(src0->type)) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) return false;
    if (src0->ne[0] <= 0 || (src0->ne[0] % ggml_blck_size(src0->type)) != 0) return false;

    switch (op->op) {
        case GGML_OP_GET_ROWS: {
            const ggml_tensor * src1 = op->src[1];
            if (src1 == nullptr || src1->type != GGML_TYPE_I32 || !ggml_is_contiguous(src1)) return false;
            if (src0->ne[2] != src1->ne[1] || src0->ne[3] != src1->ne[2] || src1->ne[3] != 1) return false;
            return op->ne[0] == src0->ne[0] && op->ne[1] == src1->ne[0] && op->ne[2] == src1->ne[1] && op->ne[3] == src1->ne[2];
        }
        case GGML_OP_CPY:
            return ggml_are_same_shape(op, src0);
        default:
            return false;
    }
}

inline bool dequant_u64_fits_u32(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool dequant_range_fits_u32_address(uint64_t offset, uint64_t bytes) {
    constexpr uint64_t u32_range = static_cast<uint64_t>(std::numeric_limits<UINT>::max()) + 1ull;
    return offset <= static_cast<uint64_t>(std::numeric_limits<UINT>::max()) && bytes <= u32_range - offset;
}

inline bool dispatch_dequant(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_dequant(node)) return false;

    const ggml_tensor * src0 = node->src[0];
    const bool is_get_rows = node->op == GGML_OP_GET_ROWS;
    const ggml_tensor * src1 = is_get_rows ? node->src[1] : src0;
    const char * shader = dequant_shader_name(src0->type);
    if (shader == nullptr) return true;

    const tensor_resource qr = ctx_resolve_tensor(ctx, src0);
    const tensor_resource ir = is_get_rows ? ctx_resolve_tensor(ctx, src1) : qr;
    const tensor_resource dr = ctx_resolve_tensor(ctx, node);
    if (!qr.valid || !ir.valid || !dr.valid) return true;

    const uint64_t total = static_cast<uint64_t>(ggml_nelements(node));
    const uint64_t src_bytes = static_cast<uint64_t>(ggml_nbytes(src0));
    const uint64_t dst_bytes = static_cast<uint64_t>(ggml_nbytes(node));
    const uint64_t idx_bytes = is_get_rows ? static_cast<uint64_t>(ggml_nbytes(src1)) : 0ull;
    const uint64_t idx_ne0 = is_get_rows ? static_cast<uint64_t>(src1->ne[0]) : static_cast<uint64_t>(src0->ne[1]);
    const uint64_t idx_ne1 = is_get_rows ? static_cast<uint64_t>(src1->ne[1]) : static_cast<uint64_t>(src0->ne[2]);
    if (total == 0) return true;
    if (!dequant_u64_fits_u32(total) ||
        !dequant_u64_fits_u32(static_cast<uint64_t>(src0->ne[0])) ||
        !dequant_u64_fits_u32(static_cast<uint64_t>(src0->ne[1])) ||
        !dequant_u64_fits_u32(idx_ne0) || !dequant_u64_fits_u32(idx_ne1)) {
        return true;
    }
    if (!dequant_range_fits_u32_address(static_cast<uint64_t>(qr.offset_bytes), src_bytes) ||
        !dequant_range_fits_u32_address(static_cast<uint64_t>(dr.offset_bytes), dst_bytes)) {
        return true;
    }
    if (is_get_rows && !dequant_range_fits_u32_address(static_cast<uint64_t>(ir.offset_bytes), idx_bytes)) return true;
    if (is_get_rows && (ir.offset_bytes % sizeof(int32_t)) != 0) return true;
    if ((dr.offset_bytes % sizeof(float)) != 0) return true;

    if (!ctx_transition(ctx, src0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (is_get_rows && !ctx_transition(ctx, src1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { src0, src1, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 9);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[9] = {
        static_cast<UINT>(total),
        static_cast<UINT>(src0->ne[0]),
        static_cast<UINT>(idx_ne0),
        static_cast<UINT>(idx_ne1),
        static_cast<UINT>(src0->ne[1]),
        static_cast<UINT>(qr.offset_bytes),
        is_get_rows ? static_cast<UINT>(ir.offset_bytes) : 0u,
        static_cast<UINT>(dr.offset_bytes),
        is_get_rows ? 0u : 1u,
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 9)) return true;

    ctx_dispatch_1d(ctx, static_cast<UINT>(total), 256);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12