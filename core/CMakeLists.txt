cmake_minimum_required(VERSION 3.10)

find_package(CUDAToolkit REQUIRED)

add_library(supereight-core INTERFACE)

target_link_libraries(supereight-core INTERFACE CUDA::cudart)
target_include_directories(supereight-core INTERFACE include)

target_compile_definitions(supereight-core INTERFACE SE_CUDA_VF)
