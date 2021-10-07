cmake_minimum_required(VERSION 3.19.2)
project(Lab1_VecMul CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(cpu_vmul STATIC cpu_vmul.h cpu_vmul.cpp)
target_link_libraries(cpu_vmul PUBLIC OpenMP::OpenMP_CXX)
target_precompile_headers(cpu_vmul PUBLIC [["omp.h"]] <cstdio>)

add_executable(Lab1_VecMul main.cu)
target_link_directories(Lab1_VecMul PUBLIC ${CUDAToolkit_LIBRARY_DIR})

set_target_properties(Lab1_VecMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Lab1_VecMul PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(Lab1_VecMul PUBLIC <cassert> <cstdio> [["cuda_runtime.h"]] [["curand.h"]] [["cublas.h"]] [["cublas_v2.h"]] <iostream> [["cpu_vmul.h"]])
target_link_libraries(Lab1_VecMul PUBLIC cublas)
target_link_libraries(Lab1_VecMul PUBLIC curand)
target_link_libraries(Lab1_VecMul PRIVATE cpu_vmul)