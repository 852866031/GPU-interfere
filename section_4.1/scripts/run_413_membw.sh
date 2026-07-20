#!/bin/bash
# 4.1.3 Memory-bandwidth interference.
# Part A (saturation): a single copy kernel on the full GPU, sweeping thread
#   blocks, to show achieved DRAM bandwidth rising and then saturating.
# Part B (interference, MPS): each kernel restricted to 50% of the SMs. Compare
#   one kernel alone on 50% SMs vs two kernels each on their own 50% concurrently.
#   If DRAM is the bottleneck, the two aggregate to ~the single-kernel peak
#   (each gets ~half) -> bandwidth interference across SMs.
source "$(dirname "$0")/common.sh"
OUT_SAT="$RESULTS_DIR/mem_bw_saturation.log"
OUT_MPS="$RESULTS_DIR/mem_bw_mps.log"
: > "$OUT_SAT"; : > "$OUT_MPS"
threads=$HALF_THREADS       # 768 = max/2 -> 2 blocks/SM at full occupancy
itrs=50
bytes=4294967296            # 4 GB (>> L2, so it is a true DRAM test)

echo "===== Part A: single-kernel bandwidth saturation ====="
for num_tb in 8 16 32 64 85 128 170 255 340; do
  cfg "exp=membw_sat num_tb=$num_tb" | tee -a "$OUT_SAT"
  "$BUILD_DIR/mem_bw" 1 $num_tb $threads $itrs $bytes | tee -a "$OUT_SAT"
done

echo "===== Part B: 50%-SM interference via MPS ====="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
cleanup() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap "cleanup; exit" SIGINT

num_tb=170   # 2 blocks/SM over 85 SMs -> saturates each kernel's half
cfg "exp=membw_mps case=alone_half" | tee -a "$OUT_MPS"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 "$BUILD_DIR/mem_bw" 1 $num_tb $threads $itrs $bytes | tee -a "$OUT_MPS"

cfg "exp=membw_mps case=colocated_half instance=1" | tee -a "$OUT_MPS"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 "$BUILD_DIR/mem_bw" 1 $num_tb $threads $itrs $bytes > "$RESULTS_DIR/_mps_inst1.log" 2>&1 &
P1=$!
cfg "exp=membw_mps case=colocated_half instance=2" | tee -a "$OUT_MPS"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 "$BUILD_DIR/mem_bw" 1 $num_tb $threads $itrs $bytes > "$RESULTS_DIR/_mps_inst2.log" 2>&1 &
P2=$!
wait $P1; wait $P2
echo "--- instance 1 ---" | tee -a "$OUT_MPS"; cat "$RESULTS_DIR/_mps_inst1.log" | tee -a "$OUT_MPS"
echo "--- instance 2 ---" | tee -a "$OUT_MPS"; cat "$RESULTS_DIR/_mps_inst2.log" | tee -a "$OUT_MPS"
rm -f "$RESULTS_DIR/_mps_inst1.log" "$RESULTS_DIR/_mps_inst2.log"
cleanup
echo "Saved -> $OUT_SAT and $OUT_MPS"
