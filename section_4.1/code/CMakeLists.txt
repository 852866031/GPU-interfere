# Self-contained build for the paper's Section 4.1 (inter-SM) experiments.
# Tuned for RTX 5090 (Blackwell, sm_120).
cmake_minimum_required(VERSION 3.22)
project(section_4_1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 5090; change for other GPUs

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# shared kernel/helper library
add_library(gpu_util_bench gpu_util_bench_lib/kernels.cu)
target_include_directories(gpu_util_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu_util_bench_lib)
target_link_libraries(gpu_util_bench PUBLIC nvidia-ml)

# the three Section 4.1 experiments
add_executable(tb_scheduler src/tb_scheduler.cu)   # 4.1.1
add_executable(l2_cache     src/l2_cache.cu)        # 4.1.2
add_executable(mem_bw       src/mem_bw.cu)          # 4.1.3

foreach(exe tb_scheduler l2_cache mem_bw)
  target_link_libraries(${exe} PUBLIC gpu_util_bench)
endforeach()
