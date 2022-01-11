cmake_minimum_required(VERSION 3.20)
project(Lab2_SaltNPepper CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDAToolkit REQUIRED)

add_executable(Lab2_SaltNPepper main.cu)

set_target_properties(Lab2_SaltNPepper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Lab2_SaltNPepper PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
