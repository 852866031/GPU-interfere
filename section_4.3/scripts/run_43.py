#!/usr/bin/env python3
"""Section 4.3 — interference between a PyTorch matmul (torch.mm) and a custom
FP32-FMA compute kernel, colocated via CUDA streams + Python threads.

Sweeps square matrix sizes. For each size it measures the matmul latency ALONE
and COLOCATED with a persistent fma kernel, and the fma latency alone/colocated.
Emits @RESULT tags parsed by parse_and_plot.py.
"""
import argparse, statistics, threading, sys
import torch
from ctypes import CDLL

def emit(*a):
    print(*a); sys.stdout.flush()

def mm_median(m1, m2, num_runs, stream):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    with torch.cuda.stream(stream):
        for i in range(num_runs):
            starts[i].record(); torch.mm(m1, m2); ends[i].record()
    ends[-1].synchronize()
    lats = sorted(starts[i].elapsed_time(ends[i]) for i in range(num_runs))
    return statistics.median(lats)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--runs_mm", type=int, default=101)      # odd -> clean median
    p.add_argument("--iters_interf", type=int, default=3000000)
    p.add_argument("--runs_interf", type=int, default=40)   # persistent: covers the mm loop
    p.add_argument("--num_tb", type=int, default=170)       # RTX 5090 SM count
    p.add_argument("--num_threads", type=int, default=128)  # 1 warp per SMSP
    p.add_argument("--shared_lib", type=str, default="../build/libpython_interface.so")
    args = p.parse_args()

    c = CDLL(args.shared_lib)
    emit("Python C interface shared library loaded")
    dev = torch.cuda.get_device_name(0)
    emit(f"@ENV gpu={dev.replace(' ', '_')} num_tb={args.num_tb} num_threads={args.num_threads} "
         f"iters_interf={args.iters_interf} runs_interf={args.runs_interf}")
    stream = torch.cuda.Stream()

    for N in args.sizes:
        a = torch.randn(N, N, device="cuda"); b = torch.randn(N, N, device="cuda")
        mm_median(a, b, 3, stream)                       # warm up / preload this size
        torch.cuda.synchronize()

        # matmul alone
        mm_alone = mm_median(a, b, args.runs_mm, stream)
        emit(f"@RESULT exp=mm size={N} case=mm_alone ms={mm_alone:.6f}")

        # fma alone (C prints [FP32] Run i: Latency lines)
        emit(f"@CONFIG size={N} phase=fma_alone")
        c.run_fp32_fma_kernel(args.num_tb, args.num_threads, args.iters_interf, args.runs_interf)

        # colocated: launch fma competitor + mm loop concurrently
        emit(f"@CONFIG size={N} phase=colocated")
        result = {}
        t_fma = threading.Thread(target=c.run_fp32_fma_kernel,
                                 args=(args.num_tb, args.num_threads, args.iters_interf, args.runs_interf))
        t_mm = threading.Thread(target=lambda: result.__setitem__("mm", mm_median(a, b, args.runs_mm, stream)))
        t_fma.start(); t_mm.start()
        t_mm.join(); t_fma.join()
        emit(f"@RESULT exp=mm size={N} case=mm_coloc ms={result['mm']:.6f}")
        emit(f"@RESULT exp=mm size={N} case=slowdown ratio={result['mm']/mm_alone:.3f}")

    emit("Done!")
