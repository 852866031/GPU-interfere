#!/bin/bash
# §3 Pitfall: achieved occupancy as a compute requirement (CUDA streams variant)
# A 4-warp, high-ILP kernel already saturates the FMA pipeline, so low "achieved
# occupancy" hides a fully busy SM. Colocating a 2nd copy of it does NOT speed up.
source "$(dirname "$0")/common.sh"

num_tb=$NUM_SM             # 170: one block per SM
num_threads_per_tb=128     # 4 warps
num_itrs=100000000

echo "--- single compute kernel (all SMs) ---";     $BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs
echo "--- two compute kernels sequential ---";       $BUILD_DIR/achieved_occupancy 2 $num_tb $num_threads_per_tb $num_itrs
echo "--- two compute kernels colocated (streams) ---"; $BUILD_DIR/achieved_occupancy 3 $num_tb $num_threads_per_tb $num_itrs
