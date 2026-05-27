#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"

namespace ggml_d3d12 {

// Per-tensor metadata cached at tensor-init time to avoid recomputing
// strides / row sizes / block counts inside hot dispatch paths.
//
// Owned by the buffer (allocated/freed alongside the resource) and
// referenced via `tensor->extra`. Trivially copyable.
struct tensor_extra {
    // Byte offset of the tensor within its backing ID3D12Resource.
    size_t   offset_in_buffer = 0;
    // Cached from tensor->ne / nb (so quant shader code doesn't have to
    // chase tensor->ne for every element).
    int64_t  ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    size_t   nb[GGML_MAX_DIMS] = {};
    // Element type
    enum ggml_type type = GGML_TYPE_F32;
    // Number of quant blocks per row (== ne[0] / blck_size, 1 for non-quant)
    int64_t  blocks_per_row = 0;
    // Byte size of one quant block (== ggml_type_size, == sizeof(elt) for non-quant)
    size_t   block_byte_size = 0;
    // Elements per block (1 for non-quant)
    int64_t  block_size = 1;
    // True if `type` is a quantized type.
    bool     quantized = false;
    // True if the tensor is logically contiguous (matches ggml_is_contiguous).
    bool     contiguous = false;
};

// Populate a tensor_extra from a ggml_tensor. Safe to call multiple times.
inline void fill_tensor_extra(tensor_extra & ex,
                              const struct ggml_tensor * t,
                              size_t offset_in_buffer) {
    ex.offset_in_buffer = offset_in_buffer;
    ex.type = t->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        ex.ne[i] = t->ne[i];
        ex.nb[i] = t->nb[i];
    }
    ex.quantized = ggml_is_quantized(t->type);
    ex.block_size = ggml_blck_size(t->type);
    ex.block_byte_size = ggml_type_size(t->type);
    ex.blocks_per_row = ex.block_size > 0 ? (t->ne[0] / ex.block_size) : t->ne[0];
    ex.contiguous = ggml_is_contiguous(t);
}

// Compute the on-disk byte size of a tensor row (== ggml_row_size).
inline size_t row_bytes(enum ggml_type type, int64_t ne0) {
    return ggml_row_size(type, ne0);
}

// Compute total tensor byte size including any block-alignment padding.
// Mirrors ggml_nbytes().
inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return ggml_nbytes(t);
}

// Return true for quantized types that consist of fixed-size blocks.
inline bool is_blocked_quant(enum ggml_type type) {
    return ggml_is_quantized(type);
}

// Return the descriptor format hint that should be used when binding this
// tensor as a UAV/SRV in D3D12. Callers cast the returned value to DXGI_FORMAT.
// Quant and non-quant tensors are exposed as raw byte buffers by default.
inline uint32_t srv_format_hint(enum ggml_type type) {
    (void) type;
    return static_cast<uint32_t>(0);
}

} // namespace ggml_d3d12
