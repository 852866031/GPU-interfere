#!/bin/bash
# 4.1.1 Thread-block scheduler interference.
# Run A: 1 block/SM  -> expect colocated == alone (concurrent).
# Run B: 2 blocks/SM -> expect colocated == sequential (serialized).
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_DIR/tb_scheduler.log"
: > "$OUT"
threads=$HALF_THREADS
itrs=100000

for blocks_per_sm in 1 2; do
  num_tb=$(( NUM_SM * blocks_per_sm ))
  for mode in 1 2 3; do
    cfg "exp=tb blocks_per_sm=$blocks_per_sm num_tb=$num_tb mode=$mode" | tee -a "$OUT"
    "$BUILD_DIR/tb_scheduler" $mode $num_tb $threads $itrs | tee -a "$OUT"
  done
done
echo "Saved -> $OUT"
