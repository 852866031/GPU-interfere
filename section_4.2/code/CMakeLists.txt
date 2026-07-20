# Self-contained build for the paper's Section 4.2 (intra-SM) experiments.
# Tuned for RTX 5090 (Blackwell, sm_120).
cmake_minimum_required(VERSION 3.22)
project(section_4_2 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 5090; change for other GPUs

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(gpu_util_bench gpu_util_bench_lib/kernels.cu)
target_include_directories(gpu_util_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu_util_bench_lib)
target_link_libraries(gpu_util_bench PUBLIC nvidia-ml)

add_executable(l1_cache  src/l1_cache.cu)    # 4.2.1
add_executable(ipc       src/ipc.cu)         # 4.2.2
add_executable(pipelines src/pipelines.cu)   # 4.2.3

foreach(exe l1_cache ipc pipelines)
  target_link_libraries(${exe} PUBLIC gpu_util_bench)
endforeach()
