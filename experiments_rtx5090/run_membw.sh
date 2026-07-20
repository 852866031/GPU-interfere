#!/bin/bash
# §4.1.3 Memory-bandwidth interference  (RTX 5090 tuned)
# Two copy kernels, each pinned to 50% of SMs via MPS, sweeping thread blocks.
# Compares single-kernel latency vs two colocated kernels contending for DRAM.
source "$(dirname "$0")/common.sh"

NUM_THREADS_PER_BLOCK=$HALF_THREADS   # 768 = max_threads_per_sm / 2
NUM_ITRS=50
NUM_BYTES=4294967296                  # 4 GB (fits in 32 GB, exceeds L2)

echo "------------------------------------"
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

# Sweep blocks per kernel. With 50% MPS each kernel gets ~85 SMs.
# Keep it to full waves of 2 blocks/SM: 85 -> 170 in steps of ~42.
# TODO: watch for incomplete last waves (paper footnote) that skew bandwidth.
for NUM_TB in 42 85 128 170; do
    echo "--- $NUM_TB thread blocks per kernel ---"
    # single copy kernel on 50% of SMs
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES
    # two copy kernels, each on its own 50% of SMs, concurrently
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES &
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES
    wait
done

cleanup
