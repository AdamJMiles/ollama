#pragma once

#include <cstdint>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool mulmatvec_supported_src0_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline const char * mulmatvec_shader_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "mul_mat_vec_f32_f32";
        case GGML_TYPE_F16:  return "mul_mat_vec_f16_f32";
        case GGML_TYPE_Q4_0: return "mul_mat_vec_q4_0_f32";
        case GGML_TYPE_Q8_0: return "mul_mat_vec_q8_0_f32";
        default: return nullptr;
    }
}

inline bool mulmatvec_fits_u32(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool mulmatvec_src0_row_contiguous(const ggml_tensor * src0) {
    if (src0 == nullptr || src0->ne[0] <= 0) return false;
    const size_t block_bytes = ggml_type_size(src0->type);
    const size_t row_bytes = ggml_row_size(src0->type, src0->ne[0]);
    return src0->nb[0] == block_bytes && src0->nb[1] == row_bytes;
}

inline bool supports_op_mulmatvec(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) return false;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) return false;

    if (src1->ne[1] != 1) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!mulmatvec_supported_src0_type(src0->type)) return false;
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) return false;

    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src1->ne[2] <= 0 || src1->ne[3] <= 0) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (op->ne[0] != src0->ne[1] || op->ne[1] != 1 || op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) return false;

    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) return false;

    const int64_t blck = ggml_blck_size(src0->type);
    if (ggml_is_quantized(src0->type) && (blck <= 0 || (src0->ne[0] % blck) != 0)) return false;
    if (!mulmatvec_src0_row_contiguous(src0)) return false;

    return true;
}

inline bool dispatch_mulmatvec(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_mulmatvec(node)) return false;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const char * shader = mulmatvec_shader_name(src0->type);
    if (shader == nullptr) return true;

    const tensor_resource w_res = ctx_resolve_tensor(ctx, src0);
    const tensor_resource x_res = ctx_resolve_tensor(ctx, src1);
    const tensor_resource y_res = ctx_resolve_tensor(ctx, node);
    if (!w_res.valid || !x_res.valid || !y_res.valid) return true;

    const uint64_t K = static_cast<uint64_t>(src0->ne[0]);
    const uint64_t M = static_cast<uint64_t>(src0->ne[1]);
    const uint64_t ne2 = static_cast<uint64_t>(src1->ne[2]);
    const uint64_t batch = ne2 * static_cast<uint64_t>(src1->ne[3]);
    if (K == 0 || M == 0 || batch == 0) return true;

    const uint64_t constants[] = {
        K,
        M,
        batch,
        static_cast<uint64_t>(src0->nb[1]),
        static_cast<uint64_t>(src1->nb[2]),
        static_cast<uint64_t>(node->nb[2]),
        static_cast<uint64_t>(w_res.offset_bytes),
        static_cast<uint64_t>(x_res.offset_bytes),
        static_cast<uint64_t>(y_res.offset_bytes),
        ne2,
        static_cast<uint64_t>(src0->nb[2]),
        static_cast<uint64_t>(src0->nb[3]),
        static_cast<uint64_t>(src1->nb[2]),
        static_cast<uint64_t>(src1->nb[3]),
        static_cast<uint64_t>(node->nb[2]),
        static_cast<uint64_t>(node->nb[3]),
    };
    for (uint64_t value : constants) {
        if (!mulmatvec_fits_u32(value)) return true;
    }
    if ((x_res.offset_bytes % sizeof(float)) != 0 || (y_res.offset_bytes % sizeof(float)) != 0) return true;
    if (src0->type == GGML_TYPE_F32 && (w_res.offset_bytes % sizeof(float)) != 0) return true;

    if (!ctx_transition(ctx, src0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, src1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * uavs[3] = { src0, src1, node };
    if (!ctx_bind_raw_uavs(ctx, uavs, 3, &uav_table)) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 16);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[16] = {
        static_cast<UINT>(K),
        static_cast<UINT>(M),
        static_cast<UINT>(batch),
        static_cast<UINT>(src0->nb[1]),
        static_cast<UINT>(src1->nb[2]),
        static_cast<UINT>(node->nb[2]),
        static_cast<UINT>(w_res.offset_bytes),
        static_cast<UINT>(x_res.offset_bytes),
        static_cast<UINT>(y_res.offset_bytes),
        static_cast<UINT>(ne2),
        static_cast<UINT>(src0->nb[2]),
        static_cast<UINT>(src0->nb[3]),
        static_cast<UINT>(src1->nb[2]),
        static_cast<UINT>(src1->nb[3]),
        static_cast<UINT>(node->nb[2]),
        static_cast<UINT>(node->nb[3]),
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 16)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(M), static_cast<UINT>(batch), 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
