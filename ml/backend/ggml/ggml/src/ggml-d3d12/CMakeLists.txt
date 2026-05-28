cmake_minimum_required(VERSION 3.21)

include(ExternalProject)

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "DXC_EXECUTABLE is required to build ggml-d3d12 shaders")
endif()

set(D3D12_SHADERS_HPP "${CMAKE_CURRENT_BINARY_DIR}/ggml-d3d12-shaders.hpp")
set(D3D12_SHADERGEN_CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release)
if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND D3D12_SHADERGEN_CMAKE_ARGS "-DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}")
endif()

ExternalProject_Add(d3d12-shaders-gen
    SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/d3d12-shaders
    BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR}/d3d12-shaders-gen-build
    INSTALL_COMMAND ""
    BUILD_ALWAYS OFF
    CMAKE_ARGS ${D3D12_SHADERGEN_CMAKE_ARGS}
)
ExternalProject_Get_Property(d3d12-shaders-gen BINARY_DIR)
if(CMAKE_CONFIGURATION_TYPES)
    set(D3D12_SHADERGEN_EXE "${BINARY_DIR}/$<CONFIG>/d3d12-shaders-gen.exe")
else()
    set(D3D12_SHADERGEN_EXE "${BINARY_DIR}/d3d12-shaders-gen.exe")
endif()

file(GLOB_RECURSE D3D12_HLSL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/d3d12-shaders/*.hlsl")

add_custom_command(
    OUTPUT ${D3D12_SHADERS_HPP}
    COMMAND "${D3D12_SHADERGEN_EXE}" --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/d3d12-shaders" --output "${D3D12_SHADERS_HPP}" --dxc "${DXC_EXECUTABLE}" --depfile "${D3D12_SHADERS_HPP}.d"
    DEPENDS d3d12-shaders-gen ${D3D12_HLSL_SOURCES}
    DEPFILE "${D3D12_SHADERS_HPP}.d"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating ggml-d3d12-shaders.hpp"
    VERBATIM
)
add_custom_target(d3d12-shaders DEPENDS ${D3D12_SHADERS_HPP})

ggml_add_backend_library(ggml-d3d12
                         ggml-d3d12.cpp
                         ../../include/ggml-d3d12.h
                        )

# Output the built DLL into a `d3d12/` subdirectory of the runtime lib path
# rather than the shared root. The Go-side discovery (runner.go) treats this
# subdirectory as a filterable library and skips loading it unless
# OLLAMA_D3D12=1 is set. Leaving the DLL alongside ggml.dll / ggml-cpu-*.dll
# would cause it to be loaded unconditionally by every runner subprocess and
# its host buffer type (UPLOAD-heap, write-combined CPU memory) would be
# picked by the scheduler for "host buffer" allocations, breaking CPU reads.
set_target_properties(ggml-d3d12 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         "${OLLAMA_BUILD_DIR}/d3d12"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${OLLAMA_BUILD_DIR}/d3d12"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OLLAMA_BUILD_DIR}/d3d12"
    LIBRARY_OUTPUT_DIRECTORY         "${OLLAMA_BUILD_DIR}/d3d12"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${OLLAMA_BUILD_DIR}/d3d12"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OLLAMA_BUILD_DIR}/d3d12"
)

add_dependencies(ggml-d3d12 d3d12-shaders)
target_include_directories(ggml-d3d12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(ggml-d3d12 PRIVATE
    ${D3D12_LIB}
    ${DXGI_LIB}
    ${DXGUID_LIB}
    ${DXCOMPILER_LIB}
)
