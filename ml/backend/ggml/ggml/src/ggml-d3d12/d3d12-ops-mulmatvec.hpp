#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool mulmatvec_wave_enabled(dispatch_ctx & ctx) {
    if (std::getenv("GGML_D3D12_DISABLE_WAVE") != nullptr || ctx.device == nullptr) return false;

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts = {};
    if (FAILED(ctx.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts, sizeof(opts)))) {
        return false;
    }
    return opts.WaveOps != FALSE;
}

inline bool mulmatvec_fp16_enabled(ID3D12Device * device) {
    if (std::getenv("GGML_D3D12_DISABLE_FP16") != nullptr || device == nullptr) {
        return false;
    }

    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_8 };
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_2) {
        return false;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS4 opt4 = {};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opt4, sizeof(opt4))) && opt4.Native16BitShaderOpsSupported != FALSE;
}

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

inline bool mulmatvec_dp4a_disabled() {
    return std::getenv("GGML_D3D12_DISABLE_DP4A") != nullptr;
}

inline bool mulmatvec_dp4a_enabled(dispatch_ctx & ctx, bool dp4a_disabled) {
    if (dp4a_disabled || ctx.device == nullptr) return false;

    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_4 };
    return SUCCEEDED(ctx.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
           sm.HighestShaderModel >= D3D_SHADER_MODEL_6_4;
}

inline const char * mulmatvec_shader_name(ggml_type type, bool native_fp16, bool use_wave, bool use_dp4a) {
    switch (type) {
        case GGML_TYPE_F32:
            return use_wave ? "mul_mat_vec_f32_f32_wave" : "mul_mat_vec_f32_f32";
        case GGML_TYPE_F16:
            return native_fp16 ? "mul_mat_vec_f16_f32_fp16" : (use_wave ? "mul_mat_vec_f16_f32_wave" : "mul_mat_vec_f16_f32");
        case GGML_TYPE_Q4_0:
            return use_wave ? "mul_mat_vec_q4_0_f32_wave" : (use_dp4a ? "mul_mat_vec_q4_0_dp4a_f32" : "mul_mat_vec_q4_0_f32");
        case GGML_TYPE_Q8_0:
            return use_wave ? "mul_mat_vec_q8_0_f32_wave" : (use_dp4a ? "mul_mat_vec_q8_0_dp4a_f32" : "mul_mat_vec_q8_0_f32");
        default:
            return nullptr;
    }
}

inline bool mulmatvec_fits_u32(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool mulmatvec_src0_row_contiguous(const ggml_tensor * src0) {
    if (src0 == nullptr || src0->ne[0] <= 0) return false;
    // We only need contiguity *within* a row (so the shader can step by
    // nb[0] == elem_size to walk K elements). The row-to-row stride is
    // passed in as src0_row_stride (= nb[1]) and works for permuted views
    // (e.g. the unified KV cache laid out as [head_dim, n_kv, n_ctx] but
    // viewed as [head_dim, n_ctx, n_kv]).
    const size_t block_bytes = ggml_type_size(src0->type);
    return src0->nb[0] == block_bytes;
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

    // GQA broadcast: each src0 head/batch dim must divide src1's, so the shader
    // can index src0 via (i_n / broadcast_n) (i.e. each src0 head is shared
    // by `broadcast2` query heads, matching CUDA/Vulkan conventions).
    if (src0->ne[2] <= 0 || src0->ne[3] <= 0) return false;
    if ((src1->ne[2] % src0->ne[2]) != 0) return false;
    if ((src1->ne[3] % src0->ne[3]) != 0) return false;

    const int64_t blck = ggml_blck_size(src0->type);
    if (ggml_is_quantized(src0->type) && (blck <= 0 || (src0->ne[0] % blck) != 0)) return false;
    if (!mulmatvec_src0_row_contiguous(src0)) {
        if (std::getenv("GGML_D3D12_LOG_MMV_REJECT") != nullptr) {
            GGML_LOG_INFO("mmv reject row-contig: s0=%s[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] (need nb[0]=%zu nb[1]=%zu)\n",
                ggml_type_name(src0->type),
                (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                ggml_type_size(src0->type),
                ggml_row_size(src0->type, src0->ne[0]));
        }
        return false;
    }

    return true;
}

inline bool dispatch_mulmatvec(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_mulmatvec(node)) return false;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const bool dp4a_disabled = mulmatvec_dp4a_disabled();
    const char * shader = mulmatvec_shader_name(src0->type,
                                                mulmatvec_fp16_enabled(ctx.device),
                                                mulmatvec_wave_enabled(ctx),
                                                mulmatvec_dp4a_enabled(ctx, dp4a_disabled));
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

    const uint64_t broadcast2 = static_cast<uint64_t>(src1->ne[2]) / static_cast<uint64_t>(src0->ne[2]);
    const uint64_t broadcast3 = static_cast<uint64_t>(src1->ne[3]) / static_cast<uint64_t>(src0->ne[3]);

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
        broadcast2,
        broadcast3,
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

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 3, 18);
    if (root_sig == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const UINT consts[18] = {
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
        static_cast<UINT>(broadcast2),
        static_cast<UINT>(broadcast3),
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 3, consts, 18)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(M), static_cast<UINT>(batch), 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
