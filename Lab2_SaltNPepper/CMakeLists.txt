cmake_minimum_required(VERSION 3.20)
project(Lab2_SaltNPepper CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message('DEBUG BUILD')
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
endif()

#Linking EasyBMP
set(EBMP_PATH "${PROJECT_SOURCE_DIR}/EBMP") #Your path to library
add_library(EasyBMP STATIC "${EBMP_PATH}/EasyBMP.cpp")
target_include_directories(EasyBMP PRIVATE ${EBMP_PATH})

find_package(CUDAToolkit REQUIRED)

add_executable(Lab2_SaltNPepper main.cu)

set_target_properties(Lab2_SaltNPepper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Lab2_SaltNPepper PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(Lab2_SaltNPepper PUBLIC <ctime> [["cuda_runtime.h"]] <iostream> <algorithm> [["EBMP/EasyBMP.h"]])

target_link_libraries(Lab2_SaltNPepper PRIVATE EasyBMP)
