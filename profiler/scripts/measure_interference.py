#!/usr/bin/env python3
"""PART 2 — measure interference between two kernels.

Directly colocates a pair of kernels (via the probe harness, no counters needed)
and returns the measured slowdown: how much the target kernel slows down when the
other kernel runs beside it. This is the ground truth for "do they interfere?".

Importable: `from measure_interference import measure_pair, measure_matrix`
CLI:        python3 measure_interference.py <target> <antagonist>
            python3 measure_interference.py --matrix [kernel ...]
"""
import subprocess, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = os.path.join(ROOT, "build", "probe")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

def measure_pair(target, antagonist):
    """Slowdown of `target` when co-run with `antagonist` (>=1.0)."""
    out = subprocess.run([PROBE, "coloc", target, antagonist],
                         capture_output=True, text=True, env=ENV).stdout
    d = dict(kv.split("=") for kv in out.split() if "=" in kv)
    return float(d["slowdown"])

def measure_matrix(kernels):
    """Full target x antagonist slowdown matrix."""
    return {(t, a): measure_pair(t, a) for t in kernels for a in kernels}

def matrix_markdown(kernels, mat):
    L = ["| target ↓ \\ with → | " + " | ".join(f"`{a}`" for a in kernels) + " |",
         "|" + "---|" * (len(kernels) + 1)]
    for t in kernels:
        row = " | ".join(f"{mat[(t,a)]:.2f}×" for a in kernels)
        L.append(f"| **`{t}`** | {row} |")
    return "\n".join(L)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--matrix":
        kernels = args[1:] or ["sleep", "dram", "l2", "l1", "fma", "fp64"]
        print(f"Part 2: measuring {len(kernels)}x{len(kernels)} interference matrix ...", file=sys.stderr)
        mat = measure_matrix(kernels)
        print(matrix_markdown(kernels, mat))
    elif len(args) == 2:
        s = measure_pair(args[0], args[1])
        print(f"`{args[0]}` slowed by `{args[1]}` = {s:.2f}x")
    else:
        sys.exit("usage: measure_interference.py <target> <antag>  |  --matrix [kernels...]")
