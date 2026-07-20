#!/bin/bash
# Shared configuration for RTX 5090 (Blackwell, sm_120) experiments.
# Source this from each run script:  source "$(dirname "$0")/common.sh"

# ---- Build location -------------------------------------------------------
# Points at the build/ dir inside the paper repo. Override by exporting
# BUILD_DIR yourself before running any script.
: "${BUILD_DIR:=/home/jiaxuan/Documents/Projects/gpu-interfere/gpu-util-interference/build}"
export BUILD_DIR

# ---- Pin to a single GPU --------------------------------------------------
# This machine has 2x RTX 5090; every experiment must run on one card.
export CUDA_VISIBLE_DEVICES=0

# ---- RTX 5090 hardware constants -----------------------------------------
# Verified via cudaGetDeviceProperties on this machine.
export NUM_SM=170                 # multiProcessorCount
export HALF_SM=85                 # NUM_SM / 2  (for inter-SM isolation via MPS/split)
export MAX_THREADS_PER_SM=1536    # maxThreadsPerMultiProcessor
export HALF_THREADS=768           # MAX_THREADS_PER_SM / 2  (one block per SM, half occupancy)
export SMSP_PER_SM=4              # warp schedulers / subpartitions per SM
export L1_THREADS=64              # SMSP_PER_SM*32/2  -> half the SMSPs (L1 experiment)
export PIPE_THREADS=128           # SMSP_PER_SM*32    -> 1 warp per SMSP (pipeline experiment)

# ---- Approximate sizes (VERIFY / TUNE for your card) ----------------------
# RTX 5090 (GB202) has a large L2 (~96 MB) and a 128 KB unified L1+shared per SM.
# These are starting points for the size sweeps, not exact thresholds.
export UNIFIED_L1_KB=128          # unified L1+shared per SM (used for alignment)
# (L2 sweep range is defined inside run_l2.sh)

if [ ! -x "$BUILD_DIR/tb_scheduler" ]; then
  echo "WARNING: benchmarks not found in BUILD_DIR=$BUILD_DIR" >&2
  echo "         Build first:  cmake -S <repo> -B <repo>/build && cmake --build <repo>/build -j" >&2
fi
