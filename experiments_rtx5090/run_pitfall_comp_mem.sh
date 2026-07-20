#!/bin/bash
# §3 Pitfall: colocating a compute- and a memory-bound kernel  (RTX 5090 tuned)
# "Complementary" kernels still interfere because both saturate the warp
# scheduler (per-SM instruction issue rate).
source "$(dirname "$0")/common.sh"

NUM_THREADS_PER_BLOCK=$HALF_THREADS   # 768 -> one compute + one copy block co-reside on an SM
NUM_ITERS_COMP_BENCH=30000000   # TODO: tune so isolated compute ~= isolated copy runtime
NUM_ITERS_MEM_BENCH=180         # TODO: tune to match

echo "--- compute & copy in isolation ---"
$BUILD_DIR/comp_mem_ipc 1 $NUM_THREADS_PER_BLOCK $NUM_ITERS_COMP_BENCH $NUM_ITERS_MEM_BENCH
echo "--- compute & copy colocated ---"
$BUILD_DIR/comp_mem_ipc 3 $NUM_THREADS_PER_BLOCK $NUM_ITERS_COMP_BENCH $NUM_ITERS_MEM_BENCH
