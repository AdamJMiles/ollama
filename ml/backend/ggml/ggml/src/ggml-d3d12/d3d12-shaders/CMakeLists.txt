cmake_minimum_required(VERSION 3.21)
project(d3d12-shaders-gen CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(d3d12-shaders-gen d3d12-shaders-gen.cpp)

# This host tool has no ggml or D3D12 link dependencies; the parent build decides how to invoke it.
