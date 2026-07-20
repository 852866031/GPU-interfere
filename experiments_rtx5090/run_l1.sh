#!/bin/bash
# §4.2.2 L1 cache interference  (RTX 5090 tuned)
# Two per-block copy kernels, each occupying half the SMSPs, sweeping bytes/block.
# NOTE: RTX 5090 unified L1+shared is 128 KB/SM (vs H100's 256 KB), so both the
# alignment size and the sweep are half the paper's H100 values.
source "$(dirname "$0")/common.sh"

num_threads_per_tb=$L1_THREADS       # 64 = SMSP*32/2 -> half the subpartitions
unified_l1_cache_size=$UNIFIED_L1_KB # 128 KB
num_itrs=15000

# Sweep 16 KB -> 80 KB per block in 16 KB steps (straddles the 128 KB L1).
# TODO: narrow once you see where colocated latency diverges from sequential.
for num_bytes_per_tb in {16384..81920..16384}; do
    echo "--- $((num_bytes_per_tb/1024)) KB per block ---"
    $BUILD_DIR/l1_cache 2 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs   # sequential
    $BUILD_DIR/l1_cache 3 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs   # colocated
done
