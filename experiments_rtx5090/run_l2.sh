#!/bin/bash
# §4.1.2 L2 cache interference  (RTX 5090 tuned)
# Copy kernel alone vs colocated, sweeping copy size. Each kernel uses half the
# SMs (85) so any interference is L2, not intra-SM.
# NOTE: RTX 5090 L2 is large (~96 MB) vs H100's 50 MB, so the sweep goes higher.
source "$(dirname "$0")/common.sh"

num_tb=$HALF_SM            # 85: half the SMs -> separate SM sets, isolates L2
num_threads_per_tb=1024
num_itrs=10000

# Sweep 16 MB -> 128 MB in 16 MB steps (straddles the ~96 MB L2).
# TODO: narrow the range once you see where colocated latency departs from alone.
for NUM_BYTES in {16777216..134217728..16777216}; do
    echo "--- copy size = $((NUM_BYTES/1024/1024)) MB ---"
    $BUILD_DIR/l2_cache 1 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES   # alone
    $BUILD_DIR/l2_cache 3 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES   # colocated
done
