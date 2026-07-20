#!/usr/bin/env python3
"""PART 1 — measure each kernel individually.

Runs Nsight Compute on each kernel (in isolation) and returns its utilization of
every shared resource, as % of that resource's peak. No colocation here — this is
the per-kernel fingerprint.

Importable: `from measure_kernels import profile_kernels, METRICS`
CLI:        python3 measure_kernels.py [kernel ...]   (default: all)
"""
import subprocess, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = os.path.join(ROOT, "build", "probe")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

# key, NCU metric, label, experiment, inter/intra-SM
METRICS = [
    ("occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active",            "SM occupancy",   "4.1.1", "residency"),
    ("dram",      "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",       "DRAM bandwidth", "4.1.3", "inter-SM"),
    ("l2",        "lts__throughput.avg.pct_of_peak_sustained_elapsed",            "L2 cache",       "4.1.2", "inter-SM"),
    ("l1",        "l1tex__throughput.avg.pct_of_peak_sustained_active",           "L1 cache",       "4.2.1", "intra-SM"),
    ("issue",     "sm__inst_issued.avg.pct_of_peak_sustained_active",             "Warp scheduler", "4.2.2", "intra-SM"),
    ("fma",       "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",  "FP32 FMA pipe",  "4.2.2", "intra-SM"),
    ("fp64",      "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active", "FP64 pipe",      "4.2.3", "intra-SM"),
]
ALL_KERNELS = ["sleep", "dram", "l2", "l1", "fma", "fp64"]

# extra "context" counters: cache RESIDENCY (hit rate) and FOOTPRINT (cold-miss bytes).
# These are what let you reason about cache-*capacity* (vs the throughput rates above).
CONTEXT = [
    ("l1_hit",     "l1tex__t_sector_hit_rate.pct"),
    ("l2_hit",     "lts__t_sector_hit_rate.pct"),
    ("cold_bytes", "dram__bytes.sum"),
]

def profile_one(kernel):
    metric_list = ",".join([m[1] for m in METRICS] + [m[1] for m in CONTEXT])
    r = subprocess.run(["ncu", "--metrics", metric_list, "--launch-count", "1", "--csv",
                        PROBE, "once", kernel], capture_output=True, text=True, env=ENV)
    if "ERR_NVGPUCTRPERM" in r.stdout + r.stderr:
        sys.exit("ERROR: GPU performance counters are locked (ERR_NVGPUCTRPERM). See README to enable.")
    lines = r.stdout.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    except StopIteration:
        sys.exit(f"ncu produced no metrics for '{kernel}':\n{r.stdout}\n{r.stderr}")
    vals = {}
    for row in csv.DictReader(lines[start:]):
        try:
            vals[row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
        except ValueError:
            vals[row["Metric Name"]] = 0.0
    prof = {key: vals.get(metric, 0.0) for key, metric, *_ in METRICS}
    prof.update({key: vals.get(metric, 0.0) for key, metric in CONTEXT})
    return prof

def profile_kernels(kernels):
    return {k: profile_one(k) for k in kernels}

def _bar(pct):
    n = min(10, int(round(pct / 10)))
    return "█" * n + "░" * (10 - n)

def table_markdown(prof):
    L = [f"| kernel | " + " | ".join(m[2] for m in METRICS) + " |",
         "|" + "---|" * (len(METRICS) + 1)]
    for k, p in prof.items():
        L.append(f"| `{k}` | " + " | ".join(f"{p[key]:.1f}%" for key, *_ in METRICS) + " |")
    return "\n".join(L)

if __name__ == "__main__":
    kernels = sys.argv[1:] or ALL_KERNELS
    print(f"Part 1: measuring {len(kernels)} kernels with NCU ...", file=sys.stderr)
    prof = profile_kernels(kernels)
    print(table_markdown(prof))
