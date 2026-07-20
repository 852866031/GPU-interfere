#!/bin/bash
# Shared config for Section 4.2 (intra-SM) experiments on RTX 5090 (sm_120).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export BUILD_DIR="$ROOT/build"
export RESULTS_DIR="$ROOT/results"
export CUDA_VISIBLE_DEVICES=0

# RTX 5090 constants
export NUM_SM=170
export SMSP_PER_SM=4
export MAX_THREADS_PER_SM=1536
export L1_THREADS=64          # SMSP*32/2 -> each kernel's block uses half the SMSPs
export PIPE_THREADS=128       # SMSP*32   -> 1 warp per SMSP
export UNIFIED_L1_KB=128      # unified L1+shared per SM

mkdir -p "$RESULTS_DIR"
cfg() { echo "@CONFIG $*"; }
