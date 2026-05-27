# D3D12 shaders

This directory holds HLSL compute shader sources for the ggml D3D12 backend.

`d3d12-shaders-gen` recursively compiles every `*.hlsl` file with DXC and emits `ggml-d3d12-shaders.hpp`, embedding each DXIL blob as a byte array plus a registry table.

Shader names come from paths relative to this directory with `.hlsl` removed and non-`[A-Za-z0-9_]` characters replaced by `_`; for example, `gemm/mul_mm_q4_k.hlsl` becomes `gemm_mul_mm_q4_k`.
