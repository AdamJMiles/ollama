#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

inline bool mulmm_fp16_enabled(ID3D12Device * device) {
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
    // Only require *within-row* contiguity (nb[0] == elem/block size). The
    // row-to-row stride is passed in as src0_nb1 and works for permuted
    // views (e.g. the unified KV cache laid out as [head_dim, n_kv, n_ctx]
    // but viewed as [head_dim, n_ctx, n_kv]).
    const size_t es = ggml_type_size(t->type);
    if (ggml_is_quantized(t->type)) {
        const int64_t blck = ggml_blck_size(t->type);
        if (blck <= 0 || (t->ne[0] % blck) != 0) return false;
        return t->nb[0] == es;
    }
    return t->nb[0] == es;
}

inline bool supports_op_mulmm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) return false;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!mulmm_shape_positive(src0) || !mulmm_shape_positive(src1) || !mulmm_shape_positive(op)) return false;

    if (src1->ne[1] < 2) return false;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0) return false;
    // Q8_0 mul_mm groundwork (dequant + tiled and naive shaders, plus the
    // ne-broadcast plumbing) is in place but currently produces incorrect
    // results for the Qwen2.5-style decode shapes that this path would
    // intercept (root cause TBD). Keep it opt-in until the bug is fixed
    // — without it the prompt-eval / chunked-decode Q8_0 mul_mats stay on
    // CPU as before.
    if (src0->type == GGML_TYPE_Q8_0 && std::getenv("GGML_D3D12_ENABLE_MULMM_Q8_0") == nullptr) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) return false;
    if (op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) return false;
    // GQA / broadcast: each src0 head shared by N = src1_ne / src0_ne
    // consecutive src1 heads (floor-div), matching ggml mul_mat broadcast.
    if (src0->ne[2] == 0 || src0->ne[3] == 0) return false;
    if (src1->ne[2] % src0->ne[2] != 0) return false;
    if (src1->ne[3] % src0->ne[3] != 0) return false;
    // Q8_0 needs K divisible by the 32-element block size for the per-block
    // dequantisation path; this is true for every real LLM weight matrix.
    if (src0->type == GGML_TYPE_Q8_0 && (src0->ne[0] % 32) != 0) return false;
    // GQA mul_mm is currently slower than the CPU fallback for the very
    // large attention shapes that Qwen-style models hit during decode (no
    // flash attention path yet); leave it opt-in until either the kernel is
    // tuned or FLASH_ATTN_EXT is implemented.
    if ((src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3]) &&
        std::getenv("GGML_D3D12_ENABLE_MULMM_GQA") == nullptr) {
        return false;
    }

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
    const char * shader;
    if (src0->type == GGML_TYPE_Q8_0) {
        shader = (std::getenv("GGML_D3D12_MULMM_Q8_NAIVE") != nullptr)
            ? "mul_mm_q8_0_f32_naive"
            : "mul_mm_q8_0_f32";
    } else if (src0->type == GGML_TYPE_F16 && mulmm_fp16_enabled(ctx.device)) {
        shader = "mul_mm_f16_f32_fp16";
    } else if (src0->type == GGML_TYPE_F16) {
        shader = "mul_mm_f16_f32";
    } else {
        shader = "mul_mm_f32_f32";
    }

    const uint64_t M = static_cast<uint64_t>(node->ne[0]);
    const uint64_t N = static_cast<uint64_t>(node->ne[1]);
    const uint64_t K = static_cast<uint64_t>(src0->ne[0]);
    const uint64_t batch_ne2 = static_cast<uint64_t>(src1->ne[2]);
    const uint64_t batch = batch_ne2 * static_cast<uint64_t>(src1->ne[3]);
    const uint64_t broadcast2 = static_cast<uint64_t>(src1->ne[2]) / static_cast<uint64_t>(src0->ne[2]);
    const uint64_t broadcast3 = static_cast<uint64_t>(src1->ne[3]) / static_cast<uint64_t>(src0->ne[3]);
    const uint64_t gx = (N + 31u) / 32u;
    const uint64_t gy = (M + 31u) / 32u;
    const uint64_t gz = batch;
    if (!mulmm_dispatch_dim(gx) || !mulmm_dispatch_dim(gy) || !mulmm_dispatch_dim(gz)) {
        return true;
    }

    const tensor_resource s0r = ctx_resolve_tensor(ctx, src0);
    const tensor_resource s1r = ctx_resolve_tensor(ctx, src1);
    const tensor_resource dr  = ctx_resolve_tensor(ctx, node);
    if (!s0r.valid || !s1r.valid || !dr.valid) return true;

    const uint64_t src0_bytes = static_cast<uint64_t>(ggml_nbytes(src0));
    const uint64_t src1_bytes = static_cast<uint64_t>(ggml_nbytes(src1));
    const uint64_t dst_bytes  = static_cast<uint64_t>(ggml_nbytes(node));
    // With sliding-window UAVs the relevant range is "tensor extent" (not
    // "parent buffer extent"), which only has to fit in 4 GB (uint shader
    // address). Real LLM weights are well under that; this is just a safety
    // guard.
    if (src0_bytes > (uint64_t{1} << 32) ||
        src1_bytes > (uint64_t{1} << 32) ||
        dst_bytes  > (uint64_t{1} << 32)) {
        return true;
    }
    if (s0r.offset_bytes > s0r.buffer_size_bytes || s1r.offset_bytes > s1r.buffer_size_bytes || dr.offset_bytes > dr.buffer_size_bytes) return true;
    if (src0_bytes > static_cast<uint64_t>(s0r.buffer_size_bytes - s0r.offset_bytes) ||
        src1_bytes > static_cast<uint64_t>(s1r.buffer_size_bytes - s1r.offset_bytes) ||
        dst_bytes  > static_cast<uint64_t>(dr.buffer_size_bytes  - dr.offset_bytes)) {
        return true;
    }

    // Alignment of nb[2]/nb[3] (used by the shader for batch indexing within
    // the sliding UAV window). F32 needs 4-byte; F16 and Q8_0 need 2-byte
    // because their loads go through load_u8.
    const size_t src0_off_align = (src0->type == GGML_TYPE_F32) ? 4u : 2u;
    if (src0->ne[2] > 1 && (src0->nb[2] % src0_off_align) != 0) return true;
    if (src0->ne[3] > 1 && (src0->nb[3] % src0_off_align) != 0) return true;

    const uint64_t dims_and_strides[] = {
        M, N, K, batch_ne2, broadcast2, broadcast3,
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
    uint32_t shader_offsets[3] = { 0, 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 3, &uav_table, shader_offsets)) return true;
    if ((shader_offsets[0] % src0_off_align) != 0) return true;
    if ((shader_offsets[1] % sizeof(float)) != 0 || (shader_offsets[2] % sizeof(float)) != 0) return true;

    ID3D12RootSignature * rs = ctx_get_root_sig(ctx, 3, 12);
    if (rs == nullptr) return true;
    ID3D12PipelineState * pso = ctx.psos->get(shader, rs, {});
    if (pso == nullptr) return true;

    const UINT consts[12] = {
        static_cast<UINT>(M),
        static_cast<UINT>(N),
        static_cast<UINT>(K),
        static_cast<UINT>(batch_ne2),
        static_cast<UINT>(broadcast2),
        static_cast<UINT>(broadcast3),
        shader_offsets[0],
        shader_offsets[1],
        shader_offsets[2],
        static_cast<UINT>(src0->nb[1]),
        static_cast<UINT>(src0->nb[2]),
        static_cast<UINT>(src0->nb[3]),
    };
    if (!ctx_bind_compute(ctx, pso, rs, uav_table, 3, consts, 12)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(gx), static_cast<UINT>(gy), static_cast<UINT>(gz));
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
