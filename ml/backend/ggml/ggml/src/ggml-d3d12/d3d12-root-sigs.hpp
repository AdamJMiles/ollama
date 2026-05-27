#pragma once

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

namespace ggml_d3d12 {

struct root_sig_key {
    uint8_t uav_count;
    uint8_t root_constant_dwords;

    bool operator==(const root_sig_key & o) const {
        return uav_count == o.uav_count && root_constant_dwords == o.root_constant_dwords;
    }
};

struct root_sig_key_hash {
    size_t operator()(const root_sig_key & k) const noexcept {
        return (size_t(k.uav_count) << 8) ^ size_t(k.root_constant_dwords);
    }
};

// Constant slot index for the root constants block within the root signature.
// Useful when callers call SetComputeRoot32BitConstants.
inline constexpr UINT ROOT_CONSTANTS_PARAM_INDEX_NO_UAV = 0;
inline constexpr UINT ROOT_CONSTANTS_PARAM_INDEX_WITH_UAV = 1;
inline constexpr UINT UAV_TABLE_PARAM_INDEX = 0;

class root_sig_cache {
public:
    bool init(ID3D12Device * device);

    // Return a cached root signature for the given template. Creates on
    // first use. Returns nullptr (and logs) on failure.
    ID3D12RootSignature * get(uint8_t uav_count, uint8_t root_constant_dwords);

    // Index for SetComputeRoot32BitConstants given a template. Returns
    // UINT_MAX if root_constant_dwords == 0.
    static UINT root_constants_param_index(uint8_t uav_count, uint8_t root_constant_dwords);
    // Index for SetComputeRootDescriptorTable. Returns UINT_MAX if uav_count == 0.
    static UINT uav_table_param_index(uint8_t uav_count);

    size_t size() const;
    void clear();

private:
    ID3D12Device * device_ = nullptr;
    mutable std::mutex mutex_;
    std::unordered_map<root_sig_key, ComPtr<ID3D12RootSignature>, root_sig_key_hash> cache_;

    // Internal: build a single root signature.
    ComPtr<ID3D12RootSignature> build(uint8_t uav_count, uint8_t root_constant_dwords);
};

inline bool root_sig_cache::init(ID3D12Device * device) {
    device_ = device;
    return true;
}

inline ID3D12RootSignature * root_sig_cache::get(uint8_t uav_count, uint8_t root_constant_dwords) {
    const root_sig_key key{ uav_count, root_constant_dwords };

    std::lock_guard<std::mutex> lock(mutex_);

    const auto found = cache_.find(key);
    if (found != cache_.end()) {
        return found->second.Get();
    }

    ComPtr<ID3D12RootSignature> root_sig = build(uav_count, root_constant_dwords);
    if (!root_sig) {
        return nullptr;
    }

    auto inserted = cache_.emplace(key, root_sig);
    return inserted.first->second.Get();
}

inline UINT root_sig_cache::root_constants_param_index(uint8_t uav_count, uint8_t root_constant_dwords) {
    if (root_constant_dwords == 0) {
        return UINT_MAX;
    }
    if (uav_count > 0) {
        return ROOT_CONSTANTS_PARAM_INDEX_WITH_UAV;
    }
    return ROOT_CONSTANTS_PARAM_INDEX_NO_UAV;
}

inline UINT root_sig_cache::uav_table_param_index(uint8_t uav_count) {
    if (uav_count == 0) {
        return UINT_MAX;
    }
    return UAV_TABLE_PARAM_INDEX;
}

inline size_t root_sig_cache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

inline void root_sig_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

inline ComPtr<ID3D12RootSignature> root_sig_cache::build(uint8_t uav_count, uint8_t root_constant_dwords) {
    if (device_ == nullptr || uav_count > 8 || root_constant_dwords > 64) {
        std::fprintf(stderr, "[d3d12] root_sig build failed: HR=0x%08lX\n", static_cast<unsigned long>(E_INVALIDARG));
        return {};
    }

    D3D12_DESCRIPTOR_RANGE1 range{};
    D3D12_ROOT_PARAMETER1 params[2]{};
    UINT param_count = 0;

    if (uav_count > 0) {
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range.NumDescriptors = uav_count;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        range.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;

        D3D12_ROOT_PARAMETER1 & p = params[param_count++];
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = 1;
        p.DescriptorTable.pDescriptorRanges = &range;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    if (root_constant_dwords > 0) {
        D3D12_ROOT_PARAMETER1 & p = params[param_count++];
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants.Num32BitValues = root_constant_dwords;
        p.Constants.ShaderRegister = 0;
        p.Constants.RegisterSpace = 0;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = param_count;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.NumStaticSamplers = 0;
    desc.Desc_1_1.pStaticSamplers = nullptr;
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &blob, &error_blob);
    if (FAILED(hr)) {
        std::fprintf(stderr, "[d3d12] root_sig build failed: HR=0x%08lX\n", static_cast<unsigned long>(hr));
        if (error_blob) {
            std::fprintf(stderr, "[d3d12] root_sig serializer error: %s\n", static_cast<const char *>(error_blob->GetBufferPointer()));
        }
        return {};
    }

    ComPtr<ID3D12RootSignature> root_sig;
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_sig));
    if (FAILED(hr)) {
        std::fprintf(stderr, "[d3d12] root_sig build failed: HR=0x%08lX\n", static_cast<unsigned long>(hr));
        return {};
    }

    return root_sig;
}

} // namespace ggml_d3d12
