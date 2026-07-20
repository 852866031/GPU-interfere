#!/bin/bash
# 4.2.2 Warp-scheduler / IPC interference.
# A memory-bound copy kernel + a compute-bound high-ILP kernel share the SM.
# Their resource profiles are "complementary" (memory vs FMA pipeline), yet they
# both need the warp scheduler to issue instructions -> they interfere there.
# Kernel iterations are tuned so copy-alone ~= compute-alone (~200 ms).
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_DIR/ipc.log"; : > "$OUT"
threads_copy=512
threads_comp=$PIPE_THREADS      # 128
itrs_copy=31000
itrs_comp=115000000
bytes=16777216                  # 16 MB -> fits in the 96 MB L2, so the copy kernel
                                # issues loads continuously (issue-bound) instead of
                                # stalling on DRAM. This is what makes it compete with
                                # the compute kernel for the warp scheduler.
ILP=4                           # mul_fp32_ilp4 -> high IPC

cfg "exp=ipc case=copy_alone"    | tee -a "$OUT"
"$BUILD_DIR/ipc" 1 0    $threads_copy $threads_comp $itrs_copy $itrs_comp $bytes | tee -a "$OUT"
cfg "exp=ipc case=compute_alone" | tee -a "$OUT"
"$BUILD_DIR/ipc" 1 $ILP $threads_copy $threads_comp $itrs_copy $itrs_comp $bytes | tee -a "$OUT"
cfg "exp=ipc case=sequential"    | tee -a "$OUT"
"$BUILD_DIR/ipc" 2 $ILP $threads_copy $threads_comp $itrs_copy $itrs_comp $bytes | tee -a "$OUT"
cfg "exp=ipc case=colocated"     | tee -a "$OUT"
"$BUILD_DIR/ipc" 3 $ILP $threads_copy $threads_comp $itrs_copy $itrs_comp $bytes | tee -a "$OUT"
echo "Saved -> $OUT"
