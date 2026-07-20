#!/bin/bash
# §3 Pitfall: achieved occupancy as an SM requirement (MPS variant)
# Restricts each kernel to `achieved_occupancy` % of SMs via MPS. Shows achieved
# occupancy poorly predicts how many SMs a kernel actually needs.
source "$(dirname "$0")/common.sh"

num_tb=$NUM_SM             # 170
num_threads_per_tb=128     # 4 warps
num_itrs=100000000
# TODO: set to the compute kernel's achieved occupancy from an NCU profile:
#   ncu -f -o occ.ncu-rep --set full $BUILD_DIR/achieved_occupancy 0 170 128 1000000
#   -> read sm__warps_active.avg.pct_of_peak_sustained_active
# 4 warps / 48 max warps per SM on RTX 5090 ~= 8.3 %  (placeholder; verify)
achieved_occupancy=8.3

echo "----------------------------------------"
echo "Running WITHOUT MPS"
$BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs   # single, all SMs
$BUILD_DIR/achieved_occupancy 2 $num_tb $num_threads_per_tb $num_itrs   # sequential, all SMs

echo "----------------------------------------"
echo "Starting MPS"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

function cleanup() {
    echo "Shutting down MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
    echo "MPS control daemon shut down."
}
trap "cleanup; exit" SIGINT

echo "Two colocated compute kernels WITH MPS, each on $achieved_occupancy % of SMs"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$achieved_occupancy $BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$achieved_occupancy $BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs
wait

cleanup
