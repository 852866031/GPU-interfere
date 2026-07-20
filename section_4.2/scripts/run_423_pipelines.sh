#!/bin/bash
# 4.2.3 Compute-pipeline interference.
# Two FP64 (mul_fp64_ilp*) compute kernels share the SM, one warp per SMSP each.
# Sweep the ILP degree: higher ILP drives more instructions/cycle into the FP64
# pipeline. On consumer Blackwell FP64 is heavily rate-limited, so as ILP rises
# the two kernels increasingly serialize on the shared FP64 pipeline.
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_DIR/pipelines.log"; : > "$OUT"
threads=$PIPE_THREADS      # 128 = 1 warp per SMSP
itrs=2000000

for ILP in 1 2 3 4; do
  cfg "exp=pipe ilp=$ILP mode=alone"      | tee -a "$OUT"
  "$BUILD_DIR/pipelines" 1 $ILP $threads $itrs | tee -a "$OUT"
  cfg "exp=pipe ilp=$ILP mode=sequential" | tee -a "$OUT"
  "$BUILD_DIR/pipelines" 2 $ILP $threads $itrs | tee -a "$OUT"
  cfg "exp=pipe ilp=$ILP mode=colocated"  | tee -a "$OUT"
  "$BUILD_DIR/pipelines" 3 $ILP $threads $itrs | tee -a "$OUT"
done
echo "Saved -> $OUT"
