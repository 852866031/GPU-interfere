#!/bin/bash
# Shared config for Section 4.1 experiments on RTX 5090 (sm_120).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export BUILD_DIR="$ROOT/build"
export RESULTS_DIR="$ROOT/results"
export CUDA_VISIBLE_DEVICES=0

# RTX 5090 constants
export NUM_SM=170
export HALF_SM=85
export MAX_THREADS_PER_SM=1536
export HALF_THREADS=768

mkdir -p "$RESULTS_DIR"
# emit a parseable config marker before each run
cfg() { echo "@CONFIG $*"; }
