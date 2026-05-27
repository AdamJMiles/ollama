#pragma once

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if __has_include("ggml-d3d12-shaders.hpp")
#include "ggml-d3d12-shaders.hpp"
#define GGML_D3D12_HAS_SHADERS 1
#else
#define GGML_D3D12_HAS_SHADERS 0
#endif

using Microsoft::WRL::ComPtr;

namespace ggml_d3d12 {

// Compact specialization key for a compute PSO. Up to 8 uint32 slots
// (extensible later). All zero means "no specialization".
struct pso_spec {
    uint32_t v[8] = {};

    bool operator==(const pso_spec & other) const {
        return std::memcmp(v, other.v, sizeof(v)) == 0;
    }
};

struct pso_key {
    const char * shader_name;   // pointer into shader_table; stable
    pso_spec spec;
    // raw pointer to a root signature (lifetime owned elsewhere)
    ID3D12RootSignature * root_sig;
};

struct pso_key_hash {
    size_t operator()(const pso_key & k) const noexcept;
};

struct pso_key_eq {
    bool operator()(const pso_key & a, const pso_key & b) const noexcept;
};

class pso_cache {
public:
    bool init(ID3D12Device * device);

    // Look up or create a compute PSO. Returns nullptr on failure.
    // `shader_name` must match an entry in ggml_d3d12_shaders::shader_table;
    // when the table is not yet available (e.g. shader list is empty), we
    // fall back to a string comparison and return nullptr with a log.
    ID3D12PipelineState * get(const char * shader_name,
                              ID3D12RootSignature * root_sig,
                              const pso_spec & spec = {});

    // For diagnostics
    size_t size() const;

    // Release everything (caller must ensure GPU is idle)
    void clear();

private:
    ID3D12Device * device_ = nullptr;
    mutable std::mutex mutex_;
    std::unordered_map<pso_key, ComPtr<ID3D12PipelineState>, pso_key_hash, pso_key_eq> cache_;
};

inline size_t pso_key_hash::operator()(const pso_key & k) const noexcept {
    size_t h = 0;
    const auto mix = [&h](size_t x) {
        h ^= x + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (h << 6) + (h >> 2);
    };

    mix(static_cast<size_t>(reinterpret_cast<uintptr_t>(k.shader_name)));
    mix(static_cast<size_t>(reinterpret_cast<uintptr_t>(k.root_sig)));
    for (size_t i = 0; i < 8; ++i) {
        mix(static_cast<size_t>(k.spec.v[i]));
    }
    return h;
}

inline bool pso_key_eq::operator()(const pso_key & a, const pso_key & b) const noexcept {
    return a.shader_name == b.shader_name && a.root_sig == b.root_sig && a.spec == b.spec;
}

inline bool pso_cache::init(ID3D12Device * device) {
    device_ = device;
    // TODO: Consider ID3D12PipelineLibrary for persistent PSO caching.
    return true;
}

inline ID3D12PipelineState * pso_cache::get(const char * shader_name,
                                            ID3D12RootSignature * root_sig,
                                            const pso_spec & spec) {
    pso_key key{shader_name, spec, root_sig};

    std::lock_guard<std::mutex> lock(mutex_);

    const auto found = cache_.find(key);
    if (found != cache_.end()) {
        return found->second.Get();
    }

    if (shader_name == nullptr) {
        std::fprintf(stderr, "ggml-d3d12: pso_cache get called with null shader name\n");
        return nullptr;
    }

    if (device_ == nullptr) {
        std::fprintf(stderr, "ggml-d3d12: pso_cache get called before init for shader '%s'\n", shader_name);
        return nullptr;
    }

#if GGML_D3D12_HAS_SHADERS
    const ggml_d3d12_shaders::shader_entry * entry = nullptr;
    for (size_t i = 0; i < ggml_d3d12_shaders::shader_table_count; ++i) {
        const auto & candidate = ggml_d3d12_shaders::shader_table[i];
        if (candidate.name != nullptr && (candidate.name == shader_name || std::strcmp(candidate.name, shader_name) == 0)) {
            entry = &candidate;
            break;
        }
    }

    if (entry == nullptr) {
        std::fprintf(stderr, "ggml-d3d12: shader '%s' not found in generated shader table\n", shader_name);
        return nullptr;
    }

    if (entry->name != shader_name) {
        key.shader_name = entry->name;
        const auto canonical_found = cache_.find(key);
        if (canonical_found != cache_.end()) {
            return canonical_found->second.Get();
        }
    }

    if (entry->data == nullptr || entry->size == 0) {
        std::fprintf(stderr, "ggml-d3d12: shader '%s' has empty bytecode\n", shader_name);
        return nullptr;
    }

    bool has_spec = false;
    for (size_t i = 0; i < 8; ++i) {
        has_spec = has_spec || spec.v[i] != 0;
    }
    if (has_spec) {
        std::fprintf(stderr,
                     "ggml-d3d12: shader '%s' requested non-zero PSO specialization; bytecode specialization is not yet implemented\n",
                     shader_name);
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = root_sig;
    desc.CS = {entry->data, entry->size};
    desc.NodeMask = 0;
    desc.CachedPSO = {nullptr, 0};
    desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ComPtr<ID3D12PipelineState> pso;
    const HRESULT hr = device_->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        std::fprintf(stderr, "ggml-d3d12: failed to create PSO for shader '%s' (hr=0x%08lx)\n", shader_name, static_cast<unsigned long>(hr));
        return nullptr;
    }

    auto inserted = cache_.emplace(key, pso);
    return inserted.first->second.Get();
#else
    static bool warned_missing_shader_table = false;
    if (!warned_missing_shader_table) {
        std::fprintf(stderr,
                     "ggml-d3d12: generated shader table header is unavailable; PSO cache cannot create shader '%s'\n",
                     shader_name);
        warned_missing_shader_table = true;
    }
    return nullptr;
#endif
}

inline size_t pso_cache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

inline void pso_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace ggml_d3d12
