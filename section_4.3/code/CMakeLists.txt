# Self-contained build for the paper's Section 4.3 (ML-kernel interference).
# Builds the CTypes shared library exposing the fma_fp32_ilp4 competitor kernel.
# Tuned for RTX 5090 (Blackwell, sm_120).
cmake_minimum_required(VERSION 3.22)
project(section_4_3 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_library(python_interface SHARED python_interface.cu)
set_target_properties(python_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(python_interface PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
