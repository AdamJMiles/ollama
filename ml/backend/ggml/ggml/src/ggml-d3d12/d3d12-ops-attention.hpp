#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "d3d12-ops-common.hpp"

namespace ggml_d3d12 {

static constexpr int64_t ATTENTION_MAX_KV = 4096;

inline float attention_param_f32(const ggml_tensor * op, int index) {
    float value = 0.0f;
    std::memcpy(&value, op->op_params + index, sizeof(value));
    return value;
}

inline UINT attention_f32_bits(float value) {
    UINT bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline bool attention_env_disable_fp16() {
    static const bool v = std::getenv("GGML_D3D12_DISABLE_FP16") != nullptr;
    return v;
}

inline bool attention_env_enable_fa_f16_mask() {
    static const bool v = std::getenv("GGML_D3D12_ENABLE_FA_F16_MASK") != nullptr;
    return v;
}

inline bool attention_fp16_enabled(dispatch_ctx & ctx) {
    return !attention_env_disable_fp16() && ctx.caps_native_fp16;
}

// Back-compat shim for legacy callsites that still pass a raw device pointer.
inline bool attention_fp16_enabled(ID3D12Device * device) {
    if (attention_env_disable_fp16() || device == nullptr) {
        return false;
    }

    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_8 };
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_2) {
        return false;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS4 opt4 = {};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opt4, sizeof(opt4))) && opt4.Native16BitShaderOpsSupported != FALSE;
}

inline bool attention_fits_u32(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<UINT>::max());
}

inline bool attention_add_span_term(uint64_t count, uint64_t stride, uint64_t & span) {
    if (count == 0) return false;
    const uint64_t max_u32 = static_cast<uint64_t>(std::numeric_limits<UINT>::max());
    const uint64_t n = count - 1u;
    if (stride != 0 && n > max_u32 / stride) return false;
    const uint64_t term = n * stride;
    if (span > max_u32 - term) return false;
    span += term;
    return true;
}

inline bool attention_tensor_span_fits_u32(size_t base, uint64_t ne0, uint64_t ne1, uint64_t ne2,
                                          size_t nb0, size_t nb1, size_t nb2, uint64_t elem_size) {
    const uint64_t u32_range = static_cast<uint64_t>(std::numeric_limits<UINT>::max()) + 1ull;
    if (base >= u32_range || elem_size == 0 || elem_size > u32_range) return false;

    uint64_t span = 0;
    if (!attention_add_span_term(ne0, static_cast<uint64_t>(nb0), span)) return false;
    if (!attention_add_span_term(ne1, static_cast<uint64_t>(nb1), span)) return false;
    if (!attention_add_span_term(ne2, static_cast<uint64_t>(nb2), span)) return false;

    const uint64_t base_u64 = static_cast<uint64_t>(base);
    if (span > u32_range - elem_size) return false;
    return base_u64 <= u32_range - elem_size - span;
}

inline bool supports_op_attention(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];
    if (q == nullptr || k == nullptr || v == nullptr) return false;
    if (op->src[4] != nullptr) return false; // sinks are not implemented in the first D3D12 cut

    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32) return false;
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) return false;
    if (v->type != k->type) return false;

    if (q->nb[0] != ggml_type_size(q->type) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || op->nb[0] != ggml_type_size(op->type)) {
        return false;
    }

    const int64_t D     = q->ne[0];
    const int64_t nq    = q->ne[1];
    const int64_t nh    = q->ne[2];
    const int64_t nkv   = k->ne[1];
    const int64_t nh_kv = k->ne[2];
    const int64_t Dv    = v->ne[0];

    if (D <= 0 || nq <= 0 || nh <= 0 || nkv <= 0 || nh_kv <= 0) return false;
    if (D > 256 || nkv > ATTENTION_MAX_KV) return false;
    if (Dv != D) return false;
    if (k->ne[0] != D || v->ne[1] != nkv || v->ne[2] != nh_kv) return false;
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || op->ne[3] != 1) return false;
    if (nh % nh_kv != 0) return false;

    if (op->ne[0] != Dv || op->ne[1] != nh || op->ne[2] != nq) return false;

    const float max_bias      = attention_param_f32(op, 1);
    const float logit_softcap = attention_param_f32(op, 2);
    if (max_bias != 0.0f || logit_softcap != 0.0f) return false;

    const int prec = op->op_params[3];
    if (prec != GGML_PREC_DEFAULT && prec != GGML_PREC_F32) return false;

    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) return false;
        if (mask->nb[0] != ggml_type_size(mask->type)) return false;
        if (mask->ne[0] < nkv || mask->ne[1] < nq || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
        // The current decode-time kernel is slower than the explicit
        // mul_mat_vec + softmax chain for the F16-mask decode shape until
        // the kernel is tuned; keep it opt-in.
        if (mask->type == GGML_TYPE_F16 &&
            !attention_env_enable_fa_f16_mask()) {
            return false;
        }
    }

    if (!attention_fits_u32(static_cast<uint64_t>(D)) || !attention_fits_u32(static_cast<uint64_t>(nq)) ||
        !attention_fits_u32(static_cast<uint64_t>(nh)) || !attention_fits_u32(static_cast<uint64_t>(nkv)) ||
        !attention_fits_u32(static_cast<uint64_t>(nh_kv))) {
        return false;
    }

    return true;
}

inline bool dispatch_attention(dispatch_ctx & ctx, const ggml_tensor * node) {
    if (!supports_op_attention(node)) return false;

    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask = node->src[3];

    const uint64_t D     = static_cast<uint64_t>(q->ne[0]);
    const uint64_t nq    = static_cast<uint64_t>(q->ne[1]);
    const uint64_t nh    = static_cast<uint64_t>(q->ne[2]);
    const uint64_t nkv   = static_cast<uint64_t>(k->ne[1]);
    const uint64_t nh_kv = static_cast<uint64_t>(k->ne[2]);

    if (!attention_fits_u32(q->nb[1]) || !attention_fits_u32(q->nb[2]) ||
        !attention_fits_u32(k->nb[1]) || !attention_fits_u32(k->nb[2]) ||
        !attention_fits_u32(v->nb[1]) || !attention_fits_u32(v->nb[2]) ||
        !attention_fits_u32(node->nb[1]) || !attention_fits_u32(node->nb[2]) ||
        (mask != nullptr && !attention_fits_u32(mask->nb[1]))) {
        return true;
    }

    if (!ctx_transition(ctx, q, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, k, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, v, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (mask != nullptr && !ctx_transition(ctx, mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;
    if (!ctx_transition(ctx, node, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return true;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_table = {};
    const ggml_tensor * mask_for_bind = mask != nullptr ? mask : q;
    const ggml_tensor * uavs[5] = { q, k, v, mask_for_bind, node };
    uint32_t off[5] = { 0, 0, 0, 0, 0 };
    if (!ctx_bind_raw_uavs_sliding(ctx, uavs, 5, &uav_table, off)) return true;
    // q (F32) and dst (F32) need 4-byte aligned shader address; k/v depend on
    // type; mask depends on type.
    if ((off[0] % 4) != 0 || (off[4] % 4) != 0) return true;
    if (k->type == GGML_TYPE_F32 && (off[1] % 4) != 0) return true;
    if (v->type == GGML_TYPE_F32 && (off[2] % 4) != 0) return true;
    if (mask != nullptr && mask->type == GGML_TYPE_F32 && (off[3] % 4) != 0) return true;
    if (mask != nullptr && mask->type == GGML_TYPE_F16 && (off[3] % 2) != 0) return true;

    ID3D12RootSignature * root_sig = ctx_get_root_sig(ctx, 5, 21);
    if (root_sig == nullptr) return true;

    const char * shader = k->type == GGML_TYPE_F16 && attention_fp16_enabled(ctx) ? "flash_attn_ext_f32_f16_fp16" :
        (k->type == GGML_TYPE_F16 ? "flash_attn_ext_f32_f16" : "flash_attn_ext_f32_f32");
    ID3D12PipelineState * pso = ctx.psos->get(shader, root_sig, {});
    if (pso == nullptr) return true;

    const float scale = attention_param_f32(node, 0);
    const UINT mask_is_f16 = (mask != nullptr && mask->type == GGML_TYPE_F16) ? 1u : 0u;
    const UINT flags = (mask != nullptr ? 1u : 0u) | (mask_is_f16 << 1);
    const UINT consts[21] = {
        static_cast<UINT>(D),
        static_cast<UINT>(nq),
        static_cast<UINT>(nkv),
        static_cast<UINT>(nh),
        static_cast<UINT>(nh_kv),
        off[0],
        off[1],
        off[2],
        mask != nullptr ? off[3] : 0u,
        off[4],
        static_cast<UINT>(q->nb[1]),
        static_cast<UINT>(q->nb[2]),
        static_cast<UINT>(k->nb[1]),
        static_cast<UINT>(k->nb[2]),
        static_cast<UINT>(v->nb[1]),
        static_cast<UINT>(v->nb[2]),
        static_cast<UINT>(node->nb[1]),
        static_cast<UINT>(node->nb[2]),
        mask != nullptr ? static_cast<UINT>(mask->nb[1]) : 0u,
        flags,
        attention_f32_bits(scale),
    };
    if (!ctx_bind_compute(ctx, pso, root_sig, uav_table, 5, consts, 21)) return true;

    ctx_dispatch_groups(ctx, static_cast<UINT>(nh), static_cast<UINT>(nq), 1);
    ctx_uav_barrier(ctx, node);
    return true;
}

} // namespace ggml_d3d12
