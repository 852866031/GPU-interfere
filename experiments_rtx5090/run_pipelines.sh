#!/bin/bash
# §4.2.3 Compute-pipeline interference  (RTX 5090 tuned)
# Two FP64 compute kernels sharing SMs, sweeping ILP. Shows the FP64/FMA pipeline
# saturating (and interfering) before the warp scheduler does.
# NOTE: FP64 throughput varies a lot by arch; on consumer Blackwell FP64 is very
# limited, so this should show pipeline contention readily. If results look odd,
# the README suggests trying a different data-type kernel.
source "$(dirname "$0")/common.sh"

num_threads_per_tb=$PIPE_THREADS   # 128 = 1 warp per SMSP
num_itrs=20000000

for ILP in {1..4..1}; do
    echo "--- ILP $ILP ---"
    $BUILD_DIR/pipelines 2 $ILP $num_threads_per_tb $num_itrs   # sequential
    $BUILD_DIR/pipelines 3 $ILP $num_threads_per_tb $num_itrs   # colocated
done
