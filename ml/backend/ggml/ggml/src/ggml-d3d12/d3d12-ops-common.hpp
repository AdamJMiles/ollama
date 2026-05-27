#pragma once

// Shared dispatch context + helpers for D3D12 op kernels.
//
// Each op category lives in its own d3d12-ops-<cat>.hpp header which provides
// two pure functions in `namespace ggml_d3d12`:
//
//     bool supports_op_<cat>(const ggml_tensor * op);
//     bool dispatch_<cat>(dispatch_ctx & ctx, const ggml_tensor * node);
//
// `dispatch_<cat>` returns true if it handled the op (whether successfully
// or with a failure that was already logged). Returns false if the op is
// not in this category. This lets `ggml_backend_d3d12_graph_compute` try
// handlers in order until one matches.
//
// The `ctx_*` helpers declared below are implemented in ggml-d3d12.cpp and
// give op kernels access to internal d3d12_device / d3d12_buffer state
// without those types leaking into op headers.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>

#include <cstddef>
#include <cstdint>

#include "ggml.h"

#include "d3d12-desc-heap.hpp"
#include "d3d12-pso-cache.hpp"
#include "d3d12-root-sigs.hpp"
#include "d3d12-quant.hpp"

namespace ggml_d3d12 {

// Thin pointer-pack passed to every op handler. Implementations of the
// `ctx_*` helpers in ggml-d3d12.cpp use `dev_opaque` to recover the
// underlying `d3d12_device *` without exposing that type publicly.
struct dispatch_ctx {
    void *                       dev_opaque   = nullptr;
    ID3D12Device *               device       = nullptr;
    ID3D12GraphicsCommandList *  cmd          = nullptr;
    desc_heap_ring *             uav_heap     = nullptr;
    pso_cache *                  psos         = nullptr;
    root_sig_cache *             root_sigs    = nullptr;
};

// Resolved info about a tensor's underlying D3D12 storage.
struct tensor_resource {
    ID3D12Resource * resource          = nullptr;
    size_t           offset_bytes      = 0;   // byte offset of tensor->data within the resource
    size_t           buffer_size_bytes = 0;   // total size of the underlying buffer
    bool             valid             = false;
};

// Returns valid=false if `tensor` is null, has no backend buffer, or is not
// on the same d3d12 device as `ctx`.
tensor_resource ctx_resolve_tensor(dispatch_ctx & ctx, const ggml_tensor * tensor);

// Transition the underlying resource of `tensor` to `after`. No-op (returns
// true) if already in that state. Returns false on bad input.
bool ctx_transition(dispatch_ctx & ctx, const ggml_tensor * tensor, D3D12_RESOURCE_STATES after);

// Insert a UAV barrier for the underlying resource of `tensor`.
void ctx_uav_barrier(dispatch_ctx & ctx, const ggml_tensor * tensor);

// Bind a contiguous descriptor range of raw (R32_TYPELESS + FLAG_RAW) UAVs
// for the given tensors. The descriptor range is allocated from the
// dispatch_ctx descriptor heap ring. On success returns true and fills
// *out_table_gpu with the GPU descriptor handle for the first UAV.
bool ctx_bind_raw_uavs(dispatch_ctx & ctx,
                       const ggml_tensor * const * tensors,
                       size_t count,
                       D3D12_GPU_DESCRIPTOR_HANDLE * out_table_gpu);

// Look up (or create) a cached root signature. Returns nullptr on failure.
ID3D12RootSignature * ctx_get_root_sig(dispatch_ctx & ctx,
                                       uint8_t uav_count,
                                       uint8_t root_constants_dwords);

// Bind the supplied compute PSO + root signature + UAV table + root constants.
// Caller is expected to have ensured that the descriptor heap is already
// bound on `ctx.cmd` (ggml_backend_d3d12_graph_compute does this once at
// the start of the graph).
//
// If `uav_count == 0` the UAV table is skipped. If `constants_dwords == 0`
// the root constants are skipped. Returns false on bad input.
bool ctx_bind_compute(dispatch_ctx & ctx,
                      ID3D12PipelineState * pso,
                      ID3D12RootSignature * root_sig,
                      D3D12_GPU_DESCRIPTOR_HANDLE uav_table,
                      uint8_t uav_count,
                      const UINT * constants,
                      uint8_t constants_dwords);

// 1D thread dispatch. Issues ceil(threads / threads_per_group) groups along x.
void ctx_dispatch_1d(dispatch_ctx & ctx, UINT threads, UINT threads_per_group);

// Tracks the per-graph-compute resource state on the device's underlying
// d3d12_buffer objects. This is called once at the start of graph_compute
// so that ctx_transition can know the current state.
//
// (Implementation detail; ctx_* helpers maintain state via dev_opaque.)

} // namespace ggml_d3d12
