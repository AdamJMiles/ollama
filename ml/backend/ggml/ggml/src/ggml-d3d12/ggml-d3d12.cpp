#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <dxcapi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-d3d12.h"
#include "ggml-impl.h"

#include "d3d12-desc-heap.hpp"
#include "d3d12-pso-cache.hpp"
#include "d3d12-root-sigs.hpp"
#include "d3d12-quant.hpp"

#include "d3d12-ops-common.hpp"
#include "d3d12-ops-memops.hpp"
#include "d3d12-ops-unary.hpp"
#include "d3d12-ops-binary.hpp"
#include "d3d12-ops-reductions.hpp"
#include "d3d12-ops-softmax.hpp"
#include "d3d12-ops-glu.hpp"
#include "d3d12-ops-misc.hpp"

#ifndef GGML_D3D12_HAS_SHADERS
#if __has_include("ggml-d3d12-shaders.hpp")
#include "ggml-d3d12-shaders.hpp"
#define GGML_D3D12_HAS_SHADERS 1
#else
#define GGML_D3D12_HAS_SHADERS 0
#endif
#endif

using Microsoft::WRL::ComPtr;

static constexpr D3D_FEATURE_LEVEL D3D12_TARGET_FEATURE_LEVEL = D3D_FEATURE_LEVEL_12_0;
static constexpr size_t D3D12_DEFAULT_STAGING_SIZE = 64ull * 1024ull * 1024ull;
static constexpr size_t D3D12_BUFFER_ALIGNMENT = 256;

struct d3d12_device;

struct d3d12_buffer_type_context {
    d3d12_device * dev = nullptr;
};

struct d3d12_device {
    size_t index = 0;
    UINT adapter_index = 0;
    ComPtr<IDXGIAdapter4> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> cmd;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = NULL;
    std::atomic<uint64_t> fence_value{0};
    std::mutex submit_mutex;
    std::mutex init_mutex;
    std::string name;
    std::string description;
    DXGI_ADAPTER_DESC1 desc = {};
    LUID luid = {};
    ComPtr<ID3D12Resource> upload;
    void * upload_ptr = nullptr;
    size_t upload_size = 0;
    ComPtr<ID3D12Resource> readback;
    void * readback_ptr = nullptr;
    size_t readback_size = 0;
    d3d12_buffer_type_context buffer_type_context;
    ggml_backend_buffer_type buffer_type = {};
    ggml_d3d12::desc_heap_ring uav_heap;
    ggml_d3d12::pso_cache psos;
    ggml_d3d12::root_sig_cache root_sigs;
    bool compute_ready = false;

    ~d3d12_device() {
        if (upload && upload_ptr) {
            upload->Unmap(0, nullptr);
            upload_ptr = nullptr;
        }
        if (readback && readback_ptr) {
            readback->Unmap(0, nullptr);
            readback_ptr = nullptr;
        }
        if (fence_event != NULL) {
            CloseHandle(fence_event);
            fence_event = NULL;
        }
    }
};

struct d3d12_buffer {
    d3d12_device * dev = nullptr;
    ComPtr<ID3D12Resource> resource;
    void * mapped_ptr = nullptr;
    void * base = nullptr;
    size_t size = 0;
    bool host = false;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct d3d12_backend_context {
    d3d12_device * dev = nullptr;
    std::string name;
};

struct d3d12_instance_context {
    std::mutex mutex;
    bool initialized = false;
    ComPtr<IDXGIFactory6> factory;
    std::vector<std::unique_ptr<d3d12_device>> devices;
    std::vector<ggml_backend_dev_t> backend_devices;
};

static d3d12_instance_context g_d3d12;

static const char * ggml_backend_d3d12_buffer_type_get_name(ggml_backend_buffer_type_t buft);
static ggml_backend_buffer_t ggml_backend_d3d12_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static size_t ggml_backend_d3d12_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_d3d12_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static bool ggml_backend_d3d12_buffer_type_is_host(ggml_backend_buffer_type_t buft);

static const char * ggml_backend_d3d12_host_buffer_type_get_name(ggml_backend_buffer_type_t buft);
static ggml_backend_buffer_t ggml_backend_d3d12_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static bool ggml_backend_d3d12_host_buffer_type_is_host(ggml_backend_buffer_type_t buft);

static void ggml_backend_d3d12_buffer_free_buffer(ggml_backend_buffer_t buffer);
static void * ggml_backend_d3d12_buffer_get_base(ggml_backend_buffer_t buffer);
static enum ggml_status ggml_backend_d3d12_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
static void ggml_backend_d3d12_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
static void ggml_backend_d3d12_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_backend_d3d12_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
static bool ggml_backend_d3d12_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
static void ggml_backend_d3d12_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);

static const char * ggml_backend_d3d12_name(ggml_backend_t backend);
static void ggml_backend_d3d12_free(ggml_backend_t backend);
static void ggml_backend_d3d12_synchronize(ggml_backend_t backend);
static enum ggml_status ggml_backend_d3d12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph, int batch_size);

static const char * ggml_backend_d3d12_device_get_name(ggml_backend_dev_t dev);
static const char * ggml_backend_d3d12_device_get_description(ggml_backend_dev_t dev);
static void ggml_backend_d3d12_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total);
static enum ggml_backend_dev_type ggml_backend_d3d12_device_get_type(ggml_backend_dev_t dev);
static void ggml_backend_d3d12_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);
static ggml_backend_t ggml_backend_d3d12_device_init_backend(ggml_backend_dev_t dev, const char * params);
static ggml_backend_buffer_type_t ggml_backend_d3d12_device_get_buffer_type(ggml_backend_dev_t dev);
static ggml_backend_buffer_type_t ggml_backend_d3d12_device_get_host_buffer_type(ggml_backend_dev_t dev);
static bool ggml_backend_d3d12_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);
static bool ggml_backend_d3d12_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
static bool ggml_backend_d3d12_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);
static void ggml_backend_d3d12_device_reset(ggml_backend_dev_t dev);

static const char * ggml_backend_d3d12_reg_get_name(ggml_backend_reg_t reg);
static size_t ggml_backend_d3d12_reg_get_device_count(ggml_backend_reg_t reg);
static ggml_backend_dev_t ggml_backend_d3d12_reg_get_device(ggml_backend_reg_t reg, size_t index);

static const struct ggml_backend_buffer_type_i ggml_backend_d3d12_buffer_type_i = {
    /* .get_name         = */ ggml_backend_d3d12_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_d3d12_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_d3d12_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_d3d12_buffer_type_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ ggml_backend_d3d12_buffer_type_is_host,
    /* .noalloc_buffer   = */ nullptr,
};

static const struct ggml_backend_buffer_type_i ggml_backend_d3d12_host_buffer_type_i = {
    /* .get_name         = */ ggml_backend_d3d12_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_d3d12_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_d3d12_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_d3d12_buffer_type_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ ggml_backend_d3d12_host_buffer_type_is_host,
    /* .noalloc_buffer   = */ nullptr,
};

static const struct ggml_backend_buffer_i ggml_backend_d3d12_buffer_i = {
    /* .free_buffer     = */ ggml_backend_d3d12_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_d3d12_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_d3d12_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_d3d12_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_d3d12_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_d3d12_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_d3d12_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_d3d12_buffer_clear,
    /* .reset           = */ nullptr,
};

static const struct ggml_backend_i ggml_backend_d3d12_i = {
    /* .get_name                = */ ggml_backend_d3d12_name,
    /* .free                    = */ ggml_backend_d3d12_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_d3d12_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_d3d12_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
    /* .graph_reserve           = */ nullptr,
    /* .buffer_size             = */ nullptr,
    /* .reset                   = */ nullptr,
};

static const struct ggml_backend_device_i ggml_backend_d3d12_device_i = {
    /* .get_name             = */ ggml_backend_d3d12_device_get_name,
    /* .get_description      = */ ggml_backend_d3d12_device_get_description,
    /* .get_memory           = */ ggml_backend_d3d12_device_get_memory,
    /* .get_type             = */ ggml_backend_d3d12_device_get_type,
    /* .get_props            = */ ggml_backend_d3d12_device_get_props,
    /* .init_backend         = */ ggml_backend_d3d12_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_d3d12_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_d3d12_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_d3d12_device_supports_op,
    /* .supports_buft        = */ ggml_backend_d3d12_device_supports_buft,
    /* .offload_op           = */ ggml_backend_d3d12_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
    /* .reset                = */ ggml_backend_d3d12_device_reset,
};

static const struct ggml_backend_reg_i ggml_backend_d3d12_reg_i = {
    /* .get_name         = */ ggml_backend_d3d12_reg_get_name,
    /* .get_device_count = */ ggml_backend_d3d12_reg_get_device_count,
    /* .get_device       = */ ggml_backend_d3d12_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static std::string d3d12_wide_to_utf8(const wchar_t * text) {
    if (text == nullptr || text[0] == L'\0') {
        return {};
    }

    const int len = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) {
        return {};
    }

    std::string result(static_cast<size_t>(len), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, text, -1, &result[0], len, nullptr, nullptr);
    if (written <= 1) {
        return {};
    }
    result.resize(static_cast<size_t>(written - 1));
    return result;
}

static D3D12_HEAP_PROPERTIES d3d12_heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES props = {};
    props.Type = type;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    return props;
}

static D3D12_RESOURCE_DESC d3d12_buffer_desc(size_t size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = static_cast<UINT64>(std::max<size_t>(size, 1));
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

static bool d3d12_is_visible_adapter(UINT adapter_index) {
    const char * env = std::getenv("GGML_D3D12_VISIBLE_DEVICES");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }

    const char * p = env;
    while (*p != '\0') {
        char * end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end != p && value >= 0 && static_cast<UINT>(value) == adapter_index) {
            return true;
        }
        p = (end != p) ? end : p + 1;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        }
    }

    return false;
}

static void d3d12_log_hr(const char * what, HRESULT hr) {
    GGML_LOG_ERROR("ggml_d3d12: %s failed with HRESULT 0x%08x\n", what, static_cast<unsigned>(hr));
}

static void d3d12_setup_buffer_type(d3d12_device * dev) {
    dev->buffer_type_context.dev = dev;
    dev->buffer_type = ggml_backend_buffer_type {
        /* .iface    = */ ggml_backend_d3d12_buffer_type_i,
        /* .device   = */ nullptr,
        /* .context  = */ &dev->buffer_type_context,
        /* .no_alloc = */ false,
    };
}

static bool d3d12_instance_init() {
    std::lock_guard<std::mutex> lock(g_d3d12.mutex);
    if (g_d3d12.initialized) {
        return true;
    }
    g_d3d12.initialized = true;

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&g_d3d12.factory));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateDXGIFactory2", hr);
        return false;
    }

    GGML_LOG_INFO("ggml_d3d12: targeting D3D feature level 12_0 and Shader Model 6.6\n");

    for (UINT adapter_index = 0; g_d3d12.devices.size() < GGML_D3D12_MAX_DEVICES; ++adapter_index) {
        ComPtr<IDXGIAdapter4> adapter;
        hr = g_d3d12.factory->EnumAdapterByGpuPreference(
            adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            d3d12_log_hr("EnumAdapterByGpuPreference", hr);
            continue;
        }

        DXGI_ADAPTER_DESC1 desc = {};
        hr = adapter->GetDesc1(&desc);
        if (FAILED(hr)) {
            d3d12_log_hr("IDXGIAdapter::GetDesc1", hr);
            continue;
        }

        const std::string description = d3d12_wide_to_utf8(desc.Description);
        GGML_LOG_INFO("ggml_d3d12: adapter %u: %s\n", adapter_index, description.c_str());

        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            GGML_LOG_INFO("ggml_d3d12: adapter %u skipped (software adapter)\n", adapter_index);
            continue;
        }
        if (!d3d12_is_visible_adapter(adapter_index)) {
            GGML_LOG_INFO("ggml_d3d12: adapter %u skipped by GGML_D3D12_VISIBLE_DEVICES\n", adapter_index);
            continue;
        }

        hr = D3D12CreateDevice(adapter.Get(), D3D12_TARGET_FEATURE_LEVEL, __uuidof(ID3D12Device), nullptr);
        if (FAILED(hr)) {
            GGML_LOG_INFO("ggml_d3d12: adapter %u skipped (D3D feature level 12_0 unavailable, HRESULT 0x%08x)\n",
                          adapter_index, static_cast<unsigned>(hr));
            continue;
        }

        std::unique_ptr<d3d12_device> dev(new (std::nothrow) d3d12_device());
        if (!dev) {
            GGML_LOG_ERROR("ggml_d3d12: failed to allocate adapter context\n");
            continue;
        }

        dev->index = g_d3d12.devices.size();
        dev->adapter_index = adapter_index;
        dev->adapter = adapter;
        dev->desc = desc;
        dev->luid = desc.AdapterLuid;
        dev->name = std::string(GGML_D3D12_NAME) + "_" + std::to_string(dev->index);
        dev->description = description;
        d3d12_setup_buffer_type(dev.get());

        GGML_LOG_INFO("ggml_d3d12: selected %s (adapter %u, feature level 12_0)\n",
                      dev->name.c_str(), adapter_index);
        g_d3d12.devices.push_back(std::move(dev));
    }

    GGML_LOG_INFO("ggml_d3d12: found %zu suitable adapter(s)\n", g_d3d12.devices.size());
    return true;
}

static d3d12_device * d3d12_get_device(size_t index) {
    if (!d3d12_instance_init()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_d3d12.mutex);
    if (index >= g_d3d12.devices.size()) {
        return nullptr;
    }
    return g_d3d12.devices[index].get();
}

static void d3d12_release_runtime(d3d12_device * dev) {
    dev->psos.clear();
    dev->root_sigs.clear();
    dev->uav_heap.reset();
    dev->compute_ready = false;

    if (dev->upload && dev->upload_ptr) {
        dev->upload->Unmap(0, nullptr);
        dev->upload_ptr = nullptr;
    }
    if (dev->readback && dev->readback_ptr) {
        dev->readback->Unmap(0, nullptr);
        dev->readback_ptr = nullptr;
    }
    dev->upload.Reset();
    dev->upload_size = 0;
    dev->readback.Reset();
    dev->readback_size = 0;
    dev->cmd.Reset();
    dev->allocator.Reset();
    dev->queue.Reset();
    dev->fence.Reset();
    if (dev->fence_event != NULL) {
        CloseHandle(dev->fence_event);
        dev->fence_event = NULL;
    }
    dev->device.Reset();
    dev->fence_value.store(0);
}

static bool d3d12_device_ensure(d3d12_device * dev) {
    if (dev == nullptr) {
        return false;
    }
    if (dev->device) {
        return true;
    }

    std::lock_guard<std::mutex> init_lock(dev->init_mutex);
    if (dev->device) {
        return true;
    }

    HRESULT hr = D3D12CreateDevice(dev->adapter.Get(), D3D12_TARGET_FEATURE_LEVEL, IID_PPV_ARGS(&dev->device));
    if (FAILED(hr)) {
        d3d12_log_hr("D3D12CreateDevice", hr);
        d3d12_release_runtime(dev);
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;

    hr = dev->device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&dev->queue));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommandQueue", hr);
        d3d12_release_runtime(dev);
        return false;
    }

    hr = dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&dev->allocator));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommandAllocator", hr);
        d3d12_release_runtime(dev);
        return false;
    }

    hr = dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, dev->allocator.Get(), nullptr, IID_PPV_ARGS(&dev->cmd));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommandList", hr);
        d3d12_release_runtime(dev);
        return false;
    }
    hr = dev->cmd->Close();
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12GraphicsCommandList::Close", hr);
        d3d12_release_runtime(dev);
        return false;
    }

    hr = dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&dev->fence));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateFence", hr);
        d3d12_release_runtime(dev);
        return false;
    }

    dev->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (dev->fence_event == NULL) {
        GGML_LOG_ERROR("ggml_d3d12: CreateEventW failed with error %lu\n", GetLastError());
        d3d12_release_runtime(dev);
        return false;
    }

    if (!dev->uav_heap.init(dev->device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096, /*shader_visible=*/true)) {
        d3d12_release_runtime(dev);
        return false;
    }
    if (!dev->psos.init(dev->device.Get())) {
        d3d12_release_runtime(dev);
        return false;
    }
    if (!dev->root_sigs.init(dev->device.Get())) {
        d3d12_release_runtime(dev);
        return false;
    }
    dev->compute_ready = true;

    GGML_LOG_INFO("ggml_d3d12: initialized %s: %s (feature level 12_0, target shader model 6.6)\n",
                  dev->name.c_str(), dev->description.c_str());
    return true;
}

static bool d3d12_signal_and_wait_locked(d3d12_device * dev) {
    const uint64_t value = dev->fence_value.fetch_add(1) + 1;
    HRESULT hr = dev->queue->Signal(dev->fence.Get(), value);
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12CommandQueue::Signal", hr);
        return false;
    }

    if (dev->fence->GetCompletedValue() < value) {
        hr = dev->fence->SetEventOnCompletion(value, dev->fence_event);
        if (FAILED(hr)) {
            d3d12_log_hr("ID3D12Fence::SetEventOnCompletion", hr);
            return false;
        }
        const DWORD wait_res = WaitForSingleObject(dev->fence_event, INFINITE);
        if (wait_res != WAIT_OBJECT_0) {
            GGML_LOG_ERROR("ggml_d3d12: WaitForSingleObject failed with result %lu\n", wait_res);
            return false;
        }
    }

    return true;
}

static bool d3d12_begin_commands_locked(d3d12_device * dev) {
    HRESULT hr = dev->allocator->Reset();
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12CommandAllocator::Reset", hr);
        return false;
    }
    hr = dev->cmd->Reset(dev->allocator.Get(), nullptr);
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12GraphicsCommandList::Reset", hr);
        return false;
    }
    return true;
}

static bool d3d12_end_commands_and_wait_locked(d3d12_device * dev) {
    HRESULT hr = dev->cmd->Close();
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12GraphicsCommandList::Close", hr);
        return false;
    }

    ID3D12CommandList * lists[] = { dev->cmd.Get() };
    dev->queue->ExecuteCommandLists(1, lists);
    return d3d12_signal_and_wait_locked(dev);
}

static void d3d12_transition(ID3D12GraphicsCommandList * cmd, d3d12_buffer * buffer, D3D12_RESOURCE_STATES after) {
    if (buffer->state == after) {
        return;
    }

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = buffer->resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = buffer->state;
    barrier.Transition.StateAfter = after;
    cmd->ResourceBarrier(1, &barrier);
    buffer->state = after;
}

static bool d3d12_ensure_upload_locked(d3d12_device * dev, size_t size) {
    if (dev->upload && dev->upload_ptr != nullptr && dev->upload_size >= size) {
        return true;
    }

    if (dev->upload && dev->upload_ptr != nullptr) {
        dev->upload->Unmap(0, nullptr);
    }
    dev->upload.Reset();
    dev->upload_ptr = nullptr;
    dev->upload_size = 0;

    const size_t alloc_size = std::max<size_t>(size, D3D12_DEFAULT_STAGING_SIZE);
    D3D12_HEAP_PROPERTIES heap_props = d3d12_heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC desc = d3d12_buffer_desc(alloc_size, D3D12_RESOURCE_FLAG_NONE);

    HRESULT hr = dev->device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&dev->upload));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommittedResource(upload)", hr);
        return false;
    }

    D3D12_RANGE read_range = { 0, 0 };
    hr = dev->upload->Map(0, &read_range, &dev->upload_ptr);
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12Resource::Map(upload)", hr);
        dev->upload.Reset();
        return false;
    }

    dev->upload_size = alloc_size;
    return true;
}

static bool d3d12_ensure_readback_locked(d3d12_device * dev, size_t size) {
    if (dev->readback && dev->readback_ptr != nullptr && dev->readback_size >= size) {
        return true;
    }

    if (dev->readback && dev->readback_ptr != nullptr) {
        dev->readback->Unmap(0, nullptr);
    }
    dev->readback.Reset();
    dev->readback_ptr = nullptr;
    dev->readback_size = 0;

    const size_t alloc_size = std::max<size_t>(size, D3D12_DEFAULT_STAGING_SIZE);
    D3D12_HEAP_PROPERTIES heap_props = d3d12_heap_properties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC desc = d3d12_buffer_desc(alloc_size, D3D12_RESOURCE_FLAG_NONE);

    HRESULT hr = dev->device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&dev->readback));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommittedResource(readback)", hr);
        return false;
    }

    hr = dev->readback->Map(0, nullptr, &dev->readback_ptr);
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12Resource::Map(readback)", hr);
        dev->readback.Reset();
        return false;
    }

    dev->readback_size = alloc_size;
    return true;
}

static size_t d3d12_tensor_offset(const d3d12_buffer * buffer, const struct ggml_tensor * tensor, size_t offset) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer->base);
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    return static_cast<size_t>(data - base) + offset;
}

static ggml_backend_buffer_t d3d12_tensor_buffer(const struct ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

static void * d3d12_device_buffer_base(const d3d12_buffer * buffer) {
    uintptr_t base = reinterpret_cast<uintptr_t>(buffer);
    base = (base + (D3D12_BUFFER_ALIGNMENT - 1)) & ~(static_cast<uintptr_t>(D3D12_BUFFER_ALIGNMENT - 1));
    if (base == 0) {
        base = D3D12_BUFFER_ALIGNMENT;
    }
    return reinterpret_cast<void *>(base);
}

static bool d3d12_fill_device_buffer(d3d12_buffer * dst, size_t dst_offset, uint8_t value, size_t size) {
    if (size == 0) {
        return true;
    }
    if (!d3d12_device_ensure(dst->dev)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(dst->dev->submit_mutex);
    const size_t staging_size = std::min<size_t>(size, D3D12_DEFAULT_STAGING_SIZE);
    if (!d3d12_ensure_upload_locked(dst->dev, staging_size)) {
        return false;
    }
    std::memset(dst->dev->upload_ptr, value, staging_size);

    if (!d3d12_begin_commands_locked(dst->dev)) {
        return false;
    }
    d3d12_transition(dst->dev->cmd.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST);

    size_t remaining = size;
    size_t out_offset = dst_offset;
    while (remaining > 0) {
        const size_t chunk = std::min<size_t>(remaining, staging_size);
        dst->dev->cmd->CopyBufferRegion(dst->resource.Get(), static_cast<UINT64>(out_offset), dst->dev->upload.Get(), 0, static_cast<UINT64>(chunk));
        remaining -= chunk;
        out_offset += chunk;
    }

    return d3d12_end_commands_and_wait_locked(dst->dev);
}

static const char * ggml_backend_d3d12_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    d3d12_buffer_type_context * ctx = static_cast<d3d12_buffer_type_context *>(buft->context);
    return ctx->dev->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_d3d12_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    d3d12_buffer_type_context * ctx = static_cast<d3d12_buffer_type_context *>(buft->context);
    d3d12_device * dev = ctx->dev;
    if (!d3d12_device_ensure(dev)) {
        return nullptr;
    }

    d3d12_buffer * buffer_ctx = new (std::nothrow) d3d12_buffer();
    if (buffer_ctx == nullptr) {
        return nullptr;
    }
    buffer_ctx->dev = dev;
    buffer_ctx->size = size;
    buffer_ctx->host = false;
    buffer_ctx->state = D3D12_RESOURCE_STATE_COMMON;

    D3D12_HEAP_PROPERTIES heap_props = d3d12_heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC desc = d3d12_buffer_desc(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hr = dev->device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer_ctx->resource));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommittedResource(default)", hr);
        delete buffer_ctx;
        return nullptr;
    }

    buffer_ctx->base = d3d12_device_buffer_base(buffer_ctx);
    return ggml_backend_buffer_init(buft, ggml_backend_d3d12_buffer_i, buffer_ctx, size);
}

static size_t ggml_backend_d3d12_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return D3D12_BUFFER_ALIGNMENT;
}

static size_t ggml_backend_d3d12_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static bool ggml_backend_d3d12_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const char * ggml_backend_d3d12_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_D3D12_NAME "_Host";
}

static ggml_backend_buffer_t ggml_backend_d3d12_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    d3d12_device * dev = d3d12_get_device(0);
    if (!d3d12_device_ensure(dev)) {
        return nullptr;
    }

    d3d12_buffer * buffer_ctx = new (std::nothrow) d3d12_buffer();
    if (buffer_ctx == nullptr) {
        return nullptr;
    }
    buffer_ctx->dev = dev;
    buffer_ctx->size = size;
    buffer_ctx->host = true;
    buffer_ctx->state = D3D12_RESOURCE_STATE_GENERIC_READ;

    D3D12_HEAP_PROPERTIES heap_props = d3d12_heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC desc = d3d12_buffer_desc(size, D3D12_RESOURCE_FLAG_NONE);
    HRESULT hr = dev->device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer_ctx->resource));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommittedResource(host upload)", hr);
        delete buffer_ctx;
        return nullptr;
    }

    D3D12_RANGE read_range = { 0, 0 };
    hr = buffer_ctx->resource->Map(0, &read_range, &buffer_ctx->mapped_ptr);
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12Resource::Map(host upload)", hr);
        delete buffer_ctx;
        return nullptr;
    }

    buffer_ctx->base = buffer_ctx->mapped_ptr;
    return ggml_backend_buffer_init(buft, ggml_backend_d3d12_buffer_i, buffer_ctx, size);
}

static bool ggml_backend_d3d12_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static void ggml_backend_d3d12_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    if (ctx != nullptr) {
        if (ctx->host && ctx->resource && ctx->mapped_ptr != nullptr) {
            ctx->resource->Unmap(0, nullptr);
            ctx->mapped_ptr = nullptr;
        }
        delete ctx;
    }
    delete buffer;
}

static void * ggml_backend_d3d12_buffer_get_base(ggml_backend_buffer_t buffer) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    return ctx->host ? ctx->mapped_ptr : ctx->base;
}

static enum ggml_status ggml_backend_d3d12_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_d3d12_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    const size_t buffer_offset = d3d12_tensor_offset(ctx, tensor, offset);
    if (ctx->host) {
        std::memset(static_cast<char *>(ctx->mapped_ptr) + buffer_offset, value, size);
        return;
    }
    if (!d3d12_fill_device_buffer(ctx, buffer_offset, value, size)) {
        GGML_LOG_ERROR("ggml_d3d12: memset_tensor failed\n");
    }
}

static void ggml_backend_d3d12_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    const size_t dst_offset = d3d12_tensor_offset(ctx, tensor, offset);
    if (ctx->host) {
        std::memcpy(static_cast<char *>(ctx->mapped_ptr) + dst_offset, data, size);
        return;
    }
    if (!d3d12_device_ensure(ctx->dev)) {
        GGML_LOG_ERROR("ggml_d3d12: set_tensor failed to initialize device\n");
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->dev->submit_mutex);
    if (!d3d12_ensure_upload_locked(ctx->dev, size)) {
        return;
    }
    std::memcpy(ctx->dev->upload_ptr, data, size);

    if (!d3d12_begin_commands_locked(ctx->dev)) {
        return;
    }
    d3d12_transition(ctx->dev->cmd.Get(), ctx, D3D12_RESOURCE_STATE_COPY_DEST);
    ctx->dev->cmd->CopyBufferRegion(ctx->resource.Get(), static_cast<UINT64>(dst_offset), ctx->dev->upload.Get(), 0, static_cast<UINT64>(size));
    if (!d3d12_end_commands_and_wait_locked(ctx->dev)) {
        GGML_LOG_ERROR("ggml_d3d12: set_tensor copy submission failed\n");
    }
}

static void ggml_backend_d3d12_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    const size_t src_offset = d3d12_tensor_offset(ctx, tensor, offset);
    if (ctx->host) {
        std::memcpy(data, static_cast<const char *>(ctx->mapped_ptr) + src_offset, size);
        return;
    }
    if (!d3d12_device_ensure(ctx->dev)) {
        GGML_LOG_ERROR("ggml_d3d12: get_tensor failed to initialize device\n");
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->dev->submit_mutex);
    if (!d3d12_ensure_readback_locked(ctx->dev, size)) {
        return;
    }
    if (!d3d12_begin_commands_locked(ctx->dev)) {
        return;
    }
    d3d12_transition(ctx->dev->cmd.Get(), ctx, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ctx->dev->cmd->CopyBufferRegion(ctx->dev->readback.Get(), 0, ctx->resource.Get(), static_cast<UINT64>(src_offset), static_cast<UINT64>(size));
    if (!d3d12_end_commands_and_wait_locked(ctx->dev)) {
        GGML_LOG_ERROR("ggml_d3d12: get_tensor copy submission failed\n");
        return;
    }
    std::memcpy(data, ctx->dev->readback_ptr, size);
}

static bool ggml_backend_d3d12_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    d3d12_buffer * dst_ctx = static_cast<d3d12_buffer *>(buffer->context);
    if (dst_ctx->host) {
        return false;
    }

    ggml_backend_buffer_t src_buffer = d3d12_tensor_buffer(src);
    if (src_buffer == nullptr || src_buffer->buft->iface.get_name != ggml_backend_d3d12_buffer_type_get_name) {
        return false;
    }

    d3d12_buffer * src_ctx = static_cast<d3d12_buffer *>(src_buffer->context);
    if (src_ctx->host || src_ctx->dev != dst_ctx->dev) {
        return false;
    }
    if (!d3d12_device_ensure(dst_ctx->dev)) {
        return false;
    }

    const size_t src_offset = d3d12_tensor_offset(src_ctx, src, 0);
    const size_t dst_offset = d3d12_tensor_offset(dst_ctx, dst, 0);
    const size_t size = ggml_nbytes(src);

    std::lock_guard<std::mutex> lock(dst_ctx->dev->submit_mutex);
    if (!d3d12_begin_commands_locked(dst_ctx->dev)) {
        return false;
    }
    d3d12_transition(dst_ctx->dev->cmd.Get(), src_ctx, D3D12_RESOURCE_STATE_COPY_SOURCE);
    d3d12_transition(dst_ctx->dev->cmd.Get(), dst_ctx, D3D12_RESOURCE_STATE_COPY_DEST);
    dst_ctx->dev->cmd->CopyBufferRegion(dst_ctx->resource.Get(), static_cast<UINT64>(dst_offset), src_ctx->resource.Get(), static_cast<UINT64>(src_offset), static_cast<UINT64>(size));
    return d3d12_end_commands_and_wait_locked(dst_ctx->dev);
}

static void ggml_backend_d3d12_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    d3d12_buffer * ctx = static_cast<d3d12_buffer *>(buffer->context);
    if (ctx->host) {
        std::memset(ctx->mapped_ptr, value, buffer->size);
        return;
    }
    if (!d3d12_fill_device_buffer(ctx, 0, value, buffer->size)) {
        GGML_LOG_ERROR("ggml_d3d12: clear failed\n");
    }
}

static const char * ggml_backend_d3d12_name(ggml_backend_t backend) {
    d3d12_backend_context * ctx = static_cast<d3d12_backend_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_d3d12_free(ggml_backend_t backend) {
    ggml_backend_d3d12_synchronize(backend);
    d3d12_backend_context * ctx = static_cast<d3d12_backend_context *>(backend->context);
    delete ctx;
    delete backend;
}

static void ggml_backend_d3d12_synchronize(ggml_backend_t backend) {
    d3d12_backend_context * ctx = static_cast<d3d12_backend_context *>(backend->context);
    if (ctx == nullptr || ctx->dev == nullptr || !ctx->dev->queue) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx->dev->submit_mutex);
    (void) d3d12_signal_and_wait_locked(ctx->dev);
}

// ============================================================================
// dispatch_ctx helper implementations (declared in d3d12-ops-common.hpp).
// These are the only path through which op-category headers reach
// d3d12_device / d3d12_buffer internals.
// ============================================================================

namespace ggml_d3d12 {

static d3d12_buffer * resolve_buffer(const ggml_tensor * tensor) {
    if (tensor == nullptr) return nullptr;
    ggml_backend_buffer_t buf_b = d3d12_tensor_buffer(tensor);
    if (buf_b == nullptr) return nullptr;
    return static_cast<d3d12_buffer *>(buf_b->context);
}

tensor_resource ctx_resolve_tensor(dispatch_ctx & ctx, const ggml_tensor * tensor) {
    tensor_resource out{};
    d3d12_device * dev = static_cast<d3d12_device *>(ctx.dev_opaque);
    if (dev == nullptr) return out;

    d3d12_buffer * buf = resolve_buffer(tensor);
    if (buf == nullptr) return out;
    if (buf->host) return out;
    if (buf->dev != dev) return out;
    if (!buf->resource) return out;

    out.resource          = buf->resource.Get();
    out.offset_bytes      = d3d12_tensor_offset(buf, tensor, 0);
    out.buffer_size_bytes = buf->size;
    out.valid             = true;
    return out;
}

bool ctx_transition(dispatch_ctx & ctx, const ggml_tensor * tensor, D3D12_RESOURCE_STATES after) {
    d3d12_device * dev = static_cast<d3d12_device *>(ctx.dev_opaque);
    if (dev == nullptr || ctx.cmd == nullptr) return false;
    d3d12_buffer * buf = resolve_buffer(tensor);
    if (buf == nullptr || buf->dev != dev) return false;
    d3d12_transition(ctx.cmd, buf, after);
    return true;
}

void ctx_uav_barrier(dispatch_ctx & ctx, const ggml_tensor * tensor) {
    if (ctx.cmd == nullptr) return;
    d3d12_buffer * buf = resolve_buffer(tensor);
    if (buf == nullptr || !buf->resource) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type        = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buf->resource.Get();
    ctx.cmd->ResourceBarrier(1, &barrier);
}

bool ctx_bind_raw_uavs(dispatch_ctx & ctx,
                       const ggml_tensor * const * tensors,
                       size_t count,
                       D3D12_GPU_DESCRIPTOR_HANDLE * out_table_gpu) {
    if (ctx.device == nullptr || ctx.uav_heap == nullptr || out_table_gpu == nullptr) return false;
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) return false;

    desc_range r = ctx.uav_heap->allocate(static_cast<uint32_t>(count));
    if (!r.base.valid) return false;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.ViewDimension                  = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Format                         = DXGI_FORMAT_R32_TYPELESS;
    uav_desc.Buffer.FirstElement            = 0;
    uav_desc.Buffer.StructureByteStride     = 0;
    uav_desc.Buffer.CounterOffsetInBytes    = 0;
    uav_desc.Buffer.Flags                   = D3D12_BUFFER_UAV_FLAG_RAW;

    for (size_t i = 0; i < count; ++i) {
        tensor_resource res = ctx_resolve_tensor(ctx, tensors[i]);
        if (!res.valid) return false;
        uav_desc.Buffer.NumElements = static_cast<UINT>(res.buffer_size_bytes / 4);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = { r.base.cpu.ptr + r.stride * static_cast<UINT>(i) };
        ctx.device->CreateUnorderedAccessView(res.resource, nullptr, &uav_desc, cpu);
    }
    *out_table_gpu = r.base.gpu;
    return true;
}

ID3D12RootSignature * ctx_get_root_sig(dispatch_ctx & ctx,
                                       uint8_t uav_count,
                                       uint8_t root_constants_dwords) {
    if (ctx.root_sigs == nullptr) return nullptr;
    return ctx.root_sigs->get(uav_count, root_constants_dwords);
}

bool ctx_bind_compute(dispatch_ctx & ctx,
                      ID3D12PipelineState * pso,
                      ID3D12RootSignature * root_sig,
                      D3D12_GPU_DESCRIPTOR_HANDLE uav_table,
                      uint8_t uav_count,
                      const UINT * constants,
                      uint8_t constants_dwords) {
    if (ctx.cmd == nullptr || pso == nullptr || root_sig == nullptr) return false;
    ctx.cmd->SetComputeRootSignature(root_sig);
    ctx.cmd->SetPipelineState(pso);
    if (uav_count > 0) {
        const UINT slot = root_sig_cache::uav_table_param_index(uav_count);
        if (slot == UINT_MAX) return false;
        ctx.cmd->SetComputeRootDescriptorTable(slot, uav_table);
    }
    if (constants_dwords > 0) {
        if (constants == nullptr) return false;
        const UINT slot = root_sig_cache::root_constants_param_index(uav_count, constants_dwords);
        if (slot == UINT_MAX) return false;
        ctx.cmd->SetComputeRoot32BitConstants(slot, constants_dwords, constants, 0);
    }
    return true;
}

void ctx_dispatch_1d(dispatch_ctx & ctx, UINT threads, UINT threads_per_group) {
    if (ctx.cmd == nullptr || threads == 0 || threads_per_group == 0) return;
    const UINT groups = (threads + threads_per_group - 1) / threads_per_group;
    ctx.cmd->Dispatch(groups, 1, 1);
}

void ctx_dispatch_groups(dispatch_ctx & ctx, UINT gx, UINT gy, UINT gz) {
    if (ctx.cmd == nullptr || gx == 0 || gy == 0 || gz == 0) return;
    ctx.cmd->Dispatch(gx, gy, gz);
}

} // namespace ggml_d3d12

static enum ggml_status ggml_backend_d3d12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph, int batch_size) {
    GGML_UNUSED(batch_size);

    d3d12_backend_context * ctx = static_cast<d3d12_backend_context *>(backend->context);
    if (!d3d12_device_ensure(ctx->dev)) {
        return GGML_STATUS_FAILED;
    }

#if !GGML_D3D12_HAS_SHADERS
    return GGML_STATUS_FAILED;
#else
    d3d12_device * dev = ctx->dev;

    std::lock_guard<std::mutex> lock(dev->submit_mutex);
    if (!d3d12_begin_commands_locked(dev)) return GGML_STATUS_FAILED;

    // Bind the descriptor heap once for the whole graph. Each op handler
    // sets its own root signature + PSO + bindings.
    ID3D12DescriptorHeap * heaps[] = { dev->uav_heap.heap() };
    dev->cmd->SetDescriptorHeaps(1, heaps);

    ggml_d3d12::dispatch_ctx dctx;
    dctx.dev_opaque = dev;
    dctx.device     = dev->device.Get();
    dctx.cmd        = dev->cmd.Get();
    dctx.uav_heap   = &dev->uav_heap;
    dctx.psos       = &dev->psos;
    dctx.root_sigs  = &dev->root_sigs;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) continue;

        // Metadata-only ops do nothing at execution time.
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            default:
                break;
        }

        // === op dispatchers (one per category) ===
        // Each handler returns true if it claimed the op (whether it
        // succeeded or logged a failure). New Phase 5 op groups append
        // their dispatch_<cat> call here.
        bool handled = false;
        if (!handled && ggml_d3d12::dispatch_memops(dctx, node))     handled = true;
        if (!handled && ggml_d3d12::dispatch_unary(dctx, node))      handled = true;
        if (!handled && ggml_d3d12::dispatch_binary(dctx, node))     handled = true;
        if (!handled && ggml_d3d12::dispatch_reductions(dctx, node)) handled = true;
        if (!handled && ggml_d3d12::dispatch_softmax(dctx, node))    handled = true;
        if (!handled && ggml_d3d12::dispatch_glu(dctx, node))        handled = true;
        if (!handled && ggml_d3d12::dispatch_misc(dctx, node))       handled = true;
        // === end op dispatchers ===

        if (!handled) {
            GGML_LOG_ERROR("ggml_d3d12: unsupported op %s in graph_compute\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }

    if (!d3d12_end_commands_and_wait_locked(dev)) {
        return GGML_STATUS_FAILED;
    }
    dev->uav_heap.mark_used(dev->fence_value.load());
    dev->uav_heap.reclaim_to(dev->fence_value.load());

    return GGML_STATUS_SUCCESS;
#endif
}

static ggml_guid_t ggml_backend_d3d12_guid() {
    static ggml_guid guid = { 0x44, 0x33, 0x31, 0x32, 0xd3, 0xd1, 0x42, 0x21, 0x90, 0x12, 0x5a, 0xba, 0xd1, 0x2d, 0x12, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_d3d12_init(size_t dev_num) {
    d3d12_device * dev = d3d12_get_device(dev_num);
    if (!d3d12_device_ensure(dev)) {
        return nullptr;
    }

    d3d12_backend_context * ctx = new (std::nothrow) d3d12_backend_context();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->dev = dev;
    ctx->name = dev->name;

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_d3d12_guid(),
        /* .iface   = */ ggml_backend_d3d12_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_d3d12_reg(), dev_num),
        /* .context = */ ctx,
    };
    if (backend == nullptr) {
        delete ctx;
        return nullptr;
    }

    return backend;
}

bool ggml_backend_is_d3d12(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_d3d12_guid());
}

int ggml_backend_d3d12_get_device_count(void) {
    if (!d3d12_instance_init()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_d3d12.mutex);
    return static_cast<int>(g_d3d12.devices.size());
}

void ggml_backend_d3d12_get_device_description(int device, char * description, size_t description_size) {
    if (description == nullptr || description_size == 0) {
        return;
    }
    description[0] = '\0';

    d3d12_device * dev = device >= 0 ? d3d12_get_device(static_cast<size_t>(device)) : nullptr;
    if (dev == nullptr) {
        return;
    }

    std::snprintf(description, description_size, "%s", dev->description.c_str());
}

void ggml_backend_d3d12_get_device_memory(int device, size_t * free, size_t * total) {
    if (free != nullptr) {
        *free = 0;
    }
    if (total != nullptr) {
        *total = 0;
    }

    d3d12_device * dev = device >= 0 ? d3d12_get_device(static_cast<size_t>(device)) : nullptr;
    if (dev == nullptr) {
        return;
    }

    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    HRESULT hr = dev->adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
    if (FAILED(hr)) {
        d3d12_log_hr("IDXGIAdapter3::QueryVideoMemoryInfo", hr);
        return;
    }

    const UINT64 budget = info.Budget;
    const UINT64 usage = info.CurrentUsage;
    if (total != nullptr) {
        *total = static_cast<size_t>(budget);
    }
    if (free != nullptr) {
        *free = static_cast<size_t>(budget > usage ? budget - usage : 0);
    }
}

ggml_backend_buffer_type_t ggml_backend_d3d12_buffer_type(size_t dev_num) {
    d3d12_device * dev = d3d12_get_device(dev_num);
    if (dev == nullptr) {
        return nullptr;
    }
    if (dev->buffer_type.device == nullptr) {
        dev->buffer_type.device = ggml_backend_reg_dev_get(ggml_backend_d3d12_reg(), dev_num);
    }
    return &dev->buffer_type;
}

ggml_backend_buffer_type_t ggml_backend_d3d12_host_buffer_type(void) {
    static ggml_backend_buffer_type host_buffer_type = {
        /* .iface    = */ ggml_backend_d3d12_host_buffer_type_i,
        /* .device   = */ nullptr,
        /* .context  = */ nullptr,
        /* .no_alloc = */ false,
    };

    if (host_buffer_type.device == nullptr && ggml_backend_d3d12_get_device_count() > 0) {
        host_buffer_type.device = ggml_backend_reg_dev_get(ggml_backend_d3d12_reg(), 0);
    }
    return &host_buffer_type;
}

static const char * ggml_backend_d3d12_device_get_name(ggml_backend_dev_t dev) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    return ctx->name.c_str();
}

static const char * ggml_backend_d3d12_device_get_description(ggml_backend_dev_t dev) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    return ctx->description.c_str();
}

static void ggml_backend_d3d12_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    ggml_backend_d3d12_get_device_memory(static_cast<int>(ctx->index), free, total);
}

static enum ggml_backend_dev_type ggml_backend_d3d12_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_d3d12_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    props->name = ggml_backend_d3d12_device_get_name(dev);
    props->description = ggml_backend_d3d12_device_get_description(dev);
    props->id = nullptr;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    ggml_backend_d3d12_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
    props->driver_major = 0;
    props->driver_minor = 0;
    props->compute_major = 12;
    props->compute_minor = 0;
    props->integrated = ctx->desc.DedicatedVideoMemory == 0 ? 1 : 0;
    props->library = GGML_D3D12_NAME;
}

static ggml_backend_t ggml_backend_d3d12_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    return ggml_backend_d3d12_init(ctx->index);
}

static ggml_backend_buffer_type_t ggml_backend_d3d12_device_get_buffer_type(ggml_backend_dev_t dev) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    return ggml_backend_d3d12_buffer_type(ctx->index);
}

static ggml_backend_buffer_type_t ggml_backend_d3d12_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_d3d12_host_buffer_type();
}

static bool ggml_backend_d3d12_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op == nullptr) return false;

    // Metadata-only ops are always supported.
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    // === op support checks (one per category) ===
    // New Phase 5 op groups append their supports_op_<cat> call here.
    if (ggml_d3d12::supports_op_memops(op))     return true;
    if (ggml_d3d12::supports_op_unary(op))      return true;
    if (ggml_d3d12::supports_op_binary(op))     return true;
    if (ggml_d3d12::supports_op_reductions(op)) return true;
    if (ggml_d3d12::supports_op_softmax(op))    return true;
    if (ggml_d3d12::supports_op_glu(op))        return true;
    if (ggml_d3d12::supports_op_misc(op))       return true;
    // === end op support checks ===

    return false;
}

static bool ggml_backend_d3d12_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft == ggml_backend_d3d12_host_buffer_type()) {
        return true;
    }
    if (buft == nullptr || buft->iface.get_name != ggml_backend_d3d12_buffer_type_get_name) {
        return false;
    }

    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    d3d12_buffer_type_context * buft_ctx = static_cast<d3d12_buffer_type_context *>(buft->context);
    return buft_ctx != nullptr && buft_ctx->dev == ctx;
}

static bool ggml_backend_d3d12_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static void ggml_backend_d3d12_device_reset(ggml_backend_dev_t dev) {
    d3d12_device * ctx = static_cast<d3d12_device *>(dev->context);
    if (!d3d12_device_ensure(ctx)) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->submit_mutex);
    if (!d3d12_signal_and_wait_locked(ctx)) {
        return;
    }

    ComPtr<ID3D12CommandAllocator> allocator;
    HRESULT hr = ctx->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommandAllocator(reset)", hr);
        return;
    }

    ComPtr<ID3D12GraphicsCommandList> cmd;
    hr = ctx->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr, IID_PPV_ARGS(&cmd));
    if (FAILED(hr)) {
        d3d12_log_hr("CreateCommandList(reset)", hr);
        return;
    }
    hr = cmd->Close();
    if (FAILED(hr)) {
        d3d12_log_hr("ID3D12GraphicsCommandList::Close(reset)", hr);
        return;
    }

    ctx->allocator = allocator;
    ctx->cmd = cmd;
}

static const char * ggml_backend_d3d12_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_D3D12_NAME;
}

static size_t ggml_backend_d3d12_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return static_cast<size_t>(ggml_backend_d3d12_get_device_count());
}

static ggml_backend_dev_t ggml_backend_d3d12_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (!d3d12_instance_init()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_d3d12.mutex);
    if (g_d3d12.backend_devices.empty()) {
        g_d3d12.backend_devices.reserve(g_d3d12.devices.size());
        for (std::unique_ptr<d3d12_device> & d3d12_dev : g_d3d12.devices) {
            ggml_backend_dev_t backend_dev = new ggml_backend_device {
                /* .iface   = */ ggml_backend_d3d12_device_i,
                /* .reg     = */ reg,
                /* .context = */ d3d12_dev.get(),
            };
            d3d12_dev->buffer_type.device = backend_dev;
            g_d3d12.backend_devices.push_back(backend_dev);
        }
    }

    GGML_ASSERT(index < g_d3d12.backend_devices.size());
    return g_d3d12.backend_devices[index];
}

ggml_backend_reg_t ggml_backend_d3d12_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_d3d12_reg_i,
        /* .context     = */ nullptr,
    };

    d3d12_instance_init();
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_d3d12_reg)
