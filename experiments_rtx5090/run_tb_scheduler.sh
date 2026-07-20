#!/bin/bash
# §4.1.1 Thread-block scheduler interference  (RTX 5090 tuned)
# Run 1 (1 block/SM): colocated ~= 2x faster than sequential  -> concurrent, no interference
# Run 2 (2 blocks/SM): colocated ~= sequential                 -> scheduler serializes them
source "$(dirname "$0")/common.sh"

num_tb=$NUM_SM              # 170: one block per SM
num_threads_per_tb=$HALF_THREADS   # 768: half the per-SM thread budget
num_itrs=100000

echo "### Run 1: one block per SM (expect concurrency) ###"
$BUILD_DIR/tb_scheduler 1 $num_tb $num_threads_per_tb $num_itrs   # alone
$BUILD_DIR/tb_scheduler 2 $num_tb $num_threads_per_tb $num_itrs   # sequential
$BUILD_DIR/tb_scheduler 3 $num_tb $num_threads_per_tb $num_itrs   # colocated

echo "### Run 2: two blocks per SM (expect serialization) ###"
$BUILD_DIR/tb_scheduler 1 $((num_tb * 2)) $num_threads_per_tb $num_itrs
$BUILD_DIR/tb_scheduler 2 $((num_tb * 2)) $num_threads_per_tb $num_itrs
$BUILD_DIR/tb_scheduler 3 $((num_tb * 2)) $num_threads_per_tb $num_itrs
