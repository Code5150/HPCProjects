cmake_minimum_required(VERSION 3.20)
project(Lab3_Substring CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDAToolkit REQUIRED)

add_executable(Lab3_Substring main.cu)

set_target_properties(Lab3_Substring PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Lab3_Substring PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(Lab3_Substring PUBLIC <cstdint> [["cuda_runtime.h"]] <iostream>)
