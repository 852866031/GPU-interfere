#!/bin/bash
# 4.2.1 L1 cache interference.
# Two copy_kernel_per_tb kernels, each block using 64 threads (= half the SMSPs).
# Colocated, the two kernels put one block each on every SM, so they SHARE the SM
# and its L1 cache, but ideally land on different SMSPs (no warp-scheduler sharing).
# Sweep bytes/block across the 128 KB unified L1 to find the interference cliff.
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_DIR/l1_cache.log"; : > "$OUT"
threads=$L1_THREADS
l1kb=$UNIFIED_L1_KB
itrs=15000

# per-block copy sizes in KB (per-block footprint = 2x this; two blocks = 4x).
# knee expected where 4x size ~ 128 KB L1  => ~24-32 KB.
sizes_kb=(4 8 12 16 20 24 28 32 40 48 56 64)

for kb in "${sizes_kb[@]}"; do
  bytes=$(( kb * 1024 ))
  cfg "exp=l1 size_kb=$kb mode=alone"     | tee -a "$OUT"
  "$BUILD_DIR/l1_cache" 1 $threads $bytes $l1kb $itrs | tee -a "$OUT"
  cfg "exp=l1 size_kb=$kb mode=colocated" | tee -a "$OUT"
  "$BUILD_DIR/l1_cache" 3 $threads $bytes $l1kb $itrs | tee -a "$OUT"
done
echo "Saved -> $OUT"
