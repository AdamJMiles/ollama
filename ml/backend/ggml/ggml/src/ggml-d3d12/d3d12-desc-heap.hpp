#pragma once

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace ggml_d3d12 {

struct desc_handle {
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = {};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = {};
    uint32_t index = 0;
    bool valid = false;
};

struct desc_range {
    desc_handle base;
    uint32_t    count    = 0;
    uint32_t    stride   = 0;
};

class desc_heap_ring {
public:
    bool init(ID3D12Device * device,
              D3D12_DESCRIPTOR_HEAP_TYPE type,
              uint32_t capacity,
              bool shader_visible);

    // Reserve `count` consecutive descriptors. Wraps to start if not enough
    // room before end-of-ring. Returns a desc_range describing the slice.
    // Caller is responsible for ensuring previously-allocated ranges have
    // finished executing on the GPU (e.g. via `mark_used` + `reclaim_to`).
    desc_range allocate(uint32_t count);

    // Record that everything allocated up to (and including) the current
    // head was submitted with the given fence value. After the GPU signals
    // that value, `reclaim_to(value)` makes the slots reusable.
    void mark_used(uint64_t fence_value);
    void reclaim_to(uint64_t completed_fence_value);

    ID3D12DescriptorHeap * heap() const { return heap_.Get(); }
    D3D12_DESCRIPTOR_HEAP_TYPE type() const { return type_; }
    uint32_t stride() const { return stride_; }
    uint32_t capacity() const { return capacity_; }
    bool shader_visible() const { return shader_visible_; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_base() const { return gpu_base_; }

    // Reset the ring entirely. Caller must ensure GPU is idle.
    void reset();

private:
    struct fence_marker { uint64_t fence_value; uint32_t head; };

    ID3D12Device * device_ = nullptr;
    ComPtr<ID3D12DescriptorHeap> heap_;
    D3D12_DESCRIPTOR_HEAP_TYPE type_ = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_base_ = {};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_base_ = {};
    uint32_t stride_ = 0;
    uint32_t capacity_ = 0;
    uint32_t head_ = 0;
    uint32_t tail_ = 0;        // oldest live allocation (slot index)
    bool shader_visible_ = false;
    std::vector<fence_marker> markers_;
    mutable std::mutex mutex_;
};

inline bool desc_heap_ring::init(ID3D12Device * device,
                                 D3D12_DESCRIPTOR_HEAP_TYPE type,
                                 uint32_t capacity,
                                 bool shader_visible) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == nullptr || capacity == 0) {
        std::fprintf(stderr, "[d3d12] desc_heap_ring::init failed: HR=0x%08lX\n", static_cast<unsigned long>(E_INVALIDARG));
        return false;
    }

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = type;
    desc.NumDescriptors = capacity;
    desc.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ComPtr<ID3D12DescriptorHeap> heap;
    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        std::fprintf(stderr, "[d3d12] desc_heap_ring::init failed: HR=0x%08lX\n", static_cast<unsigned long>(hr));
        return false;
    }

    device_ = device;
    heap_ = heap;
    type_ = type;
    stride_ = device->GetDescriptorHandleIncrementSize(type);
    capacity_ = capacity;
    shader_visible_ = shader_visible;
    cpu_base_ = heap_->GetCPUDescriptorHandleForHeapStart();
    gpu_base_ = shader_visible ? heap_->GetGPUDescriptorHandleForHeapStart() : D3D12_GPU_DESCRIPTOR_HANDLE{};
    head_ = 0;
    tail_ = 0;
    markers_.clear();

    return true;
}

inline desc_range desc_heap_ring::allocate(uint32_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    desc_range range = {};
    if (!heap_ || capacity_ == 0 || stride_ == 0) {
        std::fprintf(stderr, "[d3d12] desc_heap_ring::allocate failed: heap is not initialized\n");
        return range;
    }
    if (count > capacity_) {
        std::fprintf(stderr,
                     "[d3d12] desc_heap_ring::allocate failed: requested %u descriptors, capacity is %u\n",
                     count,
                     capacity_);
        return range;
    }

    bool wrapped = false;
    uint32_t index = head_;
    if (head_ + count > capacity_) {
        wrapped = true;
        index = 0;
        if (tail_ == 0 || count > tail_) {
            std::fprintf(stderr,
                         "[d3d12] desc_heap_ring::allocate warning: wrapped range [0, %u) overruns live tail %u\n",
                         count,
                         tail_);
        }
    } else if (head_ < tail_ && head_ + count > tail_) {
        std::fprintf(stderr,
                     "[d3d12] desc_heap_ring::allocate warning: range [%u, %u) overruns live tail %u\n",
                     head_,
                     head_ + count,
                     tail_);
    }

    range.base.cpu.ptr = cpu_base_.ptr + static_cast<SIZE_T>(index) * stride_;
    range.base.gpu = {};
    if (shader_visible_) {
        range.base.gpu.ptr = gpu_base_.ptr + static_cast<UINT64>(index) * stride_;
    }
    range.base.index = index;
    range.base.valid = true;
    range.count = count;
    range.stride = stride_;

    head_ = wrapped ? count : head_ + count;
    return range;
}

inline void desc_heap_ring::mark_used(uint64_t fence_value) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!markers_.empty() && markers_.back().fence_value == fence_value) {
        markers_.back().head = head_;
        return;
    }

    markers_.push_back(fence_marker{ fence_value, head_ });
}

inline void desc_heap_ring::reclaim_to(uint64_t completed_fence_value) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::size_t reclaimed = 0;
    uint32_t reclaimed_head = tail_;
    while (reclaimed < markers_.size() && markers_[reclaimed].fence_value <= completed_fence_value) {
        reclaimed_head = markers_[reclaimed].head;
        ++reclaimed;
    }

    if (reclaimed > 0) {
        markers_.erase(markers_.begin(), markers_.begin() + static_cast<std::ptrdiff_t>(reclaimed));
        tail_ = reclaimed_head;
    }
    if (markers_.empty()) {
        tail_ = head_;
    }
}

inline void desc_heap_ring::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    markers_.clear();
    head_ = 0;
    tail_ = 0;
}

} // namespace ggml_d3d12
