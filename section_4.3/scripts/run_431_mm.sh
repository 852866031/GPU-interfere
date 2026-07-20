#!/bin/bash
# 4.3 PyTorch matmul vs custom FP32-FMA kernel interference (RTX 5090).
# Requires a Python env with torch (CUDA). Sweeps square matrix sizes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
OUT="$ROOT/results/mm_pytorch.log"
mkdir -p "$ROOT/results"

python3 "$HERE/run_43.py" \
    --sizes 256 512 1024 2048 4096 \
    --runs_mm 101 --iters_interf 3000000 --runs_interf 40 \
    --num_tb 170 --num_threads 128 \
    --shared_lib "$ROOT/build/libpython_interface.so" 2>&1 | tee "$OUT"
echo "Saved -> $OUT"
