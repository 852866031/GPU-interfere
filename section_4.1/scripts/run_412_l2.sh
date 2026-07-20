#!/bin/bash
# 4.1.2 L2 cache interference.
# Two copy kernels, each on half the SMs (85) so there is NO intra-SM sharing;
# any interference is L2. Sweep the per-kernel copy size across the ~96 MB L2.
# Expect: colocated ~= alone while both fit in L2; colocated > alone once the
# combined footprint thrashes L2.
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_DIR/l2_cache.log"
: > "$OUT"
num_tb=$HALF_SM        # 85 -> separate SM sets
threads=1024
itrs=3000

# per-kernel copy sizes in MB (straddle the 96 MB L2; knee expected near ~48 MB)
sizes_mb=(4 8 16 24 32 40 48 56 64 72 80 96 112 128)

for mb in "${sizes_mb[@]}"; do
  bytes=$(( mb * 1024 * 1024 ))
  cfg "exp=l2 size_mb=$mb mode=alone"     | tee -a "$OUT"
  "$BUILD_DIR/l2_cache" 1 $num_tb $threads $itrs $bytes | tee -a "$OUT"
  cfg "exp=l2 size_mb=$mb mode=colocated" | tee -a "$OUT"
  "$BUILD_DIR/l2_cache" 3 $num_tb $threads $itrs $bytes | tee -a "$OUT"
done
echo "Saved -> $OUT"
