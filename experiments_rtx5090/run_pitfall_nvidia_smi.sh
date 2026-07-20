#!/bin/bash
# §3 Pitfall: nvidia-smi/NVML utilization is misleading  (RTX 5090 tuned)
# One block reports ~100% "utilization", yet a 2nd identical kernel runs for free
# (colocated latency ~= single-kernel latency) -> the GPU was mostly idle.
source "$(dirname "$0")/common.sh"

num_tb=1
num_threads_per_tb=$HALF_THREADS   # 768
num_itrs=30000000
num_itrs_prof=40000000

echo "--- NVML utilization while 1 block runs (expect ~100%) ---"
$BUILD_DIR/nvml_util 0 $num_tb $num_threads_per_tb $num_itrs_prof
echo "--- single compute kernel latency ---"
$BUILD_DIR/nvml_util 1 $num_tb $num_threads_per_tb $num_itrs
echo "--- two colocated kernels latency (expect ~same as single) ---"
$BUILD_DIR/nvml_util 3 $num_tb $num_threads_per_tb $num_itrs
