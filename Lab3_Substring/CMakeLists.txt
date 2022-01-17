cmake_minimum_required(VERSION 3.20)
project(Lab3_Substring CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message('DEBUG BUILD')
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(CpuSubstring STATIC CpuSubstring.h CpuSubstring.cpp)
target_link_libraries(CpuSubstring PUBLIC OpenMP::OpenMP_CXX)
target_precompile_headers(CpuSubstring PUBLIC [["omp.h"]] <cstdio>)

add_executable(Lab3_Substring main.cu)

set_target_properties(Lab3_Substring PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Lab3_Substring PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(Lab3_Substring PUBLIC <cstdint> [["cuda_runtime.h"]] <iostream> <vector> <string>
        <thrust/random.h> <thrust/device_vector.h> <thrust/host_vector.h> <thrust/device_ptr.h> <thrust/device_malloc.h>
        <thrust/random/linear_congruential_engine.h> <thrust/random/uniform_int_distribution.h> <thrust/for_each.h>
        <thrust/tuple.h> <thrust/iterator/counting_iterator.h> [["CpuSubstring.h"]])
target_link_libraries(Lab3_Substring PRIVATE CpuSubstring)