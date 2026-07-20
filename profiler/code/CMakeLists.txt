cmake_minimum_required(VERSION 3.22)
project(coloc_profiler LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 5090
add_executable(probe probe.cu)
