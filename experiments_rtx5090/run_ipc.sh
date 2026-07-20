#!/bin/bash
# §4.2.1 Warp-scheduler / IPC interference  (RTX 5090 tuned)
# A copy (memory) kernel + a high-ILP compute kernel share an SM. Despite
# complementary resource profiles, they contend at the warp scheduler.
source "$(dirname "$0")/common.sh"

threads_per_tb_copy=$HALF_THREADS   # 768 so copy + compute co-reside on one SM
threads_per_tb_comp=128
num_itrs_comp=20000000   # TODO: tune so isolated compute ~= isolated copy runtime
num_itrs_copy=40         # TODO: tune to match
num_bytes=4294967296     # 4 GB
ILP=4                    # compute kernel ILP degree (mul_fp32_ilp4)

echo "--- copy alone ---";        $BUILD_DIR/ipc 1 0    $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes
echo "--- compute alone ---";     $BUILD_DIR/ipc 1 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes
echo "--- sequential ---";        $BUILD_DIR/ipc 2 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes
echo "--- colocated ---";         $BUILD_DIR/ipc 3 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes
