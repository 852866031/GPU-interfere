# GPU Colocation Interference Profiler

Takes GPU kernels and decides **whether two of them can run concurrently without
much interference**. It is split into two independent measurement stages plus a
report generator that ties them together:

| Stage | Script | What it does | Needs perf counters? |
|---|---|---|---|
| **Part 1 — measure each kernel** | `scripts/measure_kernels.py` | NCU-profiles each kernel *alone* → its % of peak on every shared resource | yes (Nsight Compute) |
| **Part 2 — measure interference** | `scripts/measure_interference.py` | directly colocates each pair → measured slowdown | no (just timing) |
| **Report** | `scripts/report.py` | Part 1 table → discussion/prediction → Part 2 matrix → verification | — |

It builds on the experiments in `../section_4.1`–`../section_4.3`: interference
happens when two kernels lean on the **same shared resource** and their combined
demand exceeds its capacity.

## Build & run

```bash
cd profiler
cmake -S code -B build && cmake --build build -j      # builds the `probe` harness (sm_120)

# the full report over a set of kernels (default: sleep dram l2 l1 fma fp64)
python3 scripts/report.py                             # -> reports/interference_report.md

# or run either stage on its own:
python3 scripts/measure_kernels.py fma fp64           # Part 1: per-kernel table
python3 scripts/measure_interference.py --matrix      # Part 2: slowdown matrix
python3 scripts/measure_interference.py l1 fma        # Part 2: one pair
```

Kernel names come from `build/probe list`: `sleep dram l2 l1 fma fp64`.

## What Part 1 measures (per kernel, % of peak)

| Resource | NCU counter | Experiment |
|---|---|---|
| SM occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | 4.1.1 |
| DRAM bandwidth | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 4.1.3 |
| L2 cache | `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.1.2 |
| L1 cache | `l1tex__throughput.avg.pct_of_peak_sustained_active` | 4.2.1 |
| Warp scheduler | `sm__inst_issued.avg.pct_of_peak_sustained_active` | 4.2.2 |
| FP32 FMA pipe | `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 4.2.2 |
| FP64 pipe | `sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active` | 4.2.3 |

Needs GPU **performance-counter access**. If you see `ERR_NVGPUCTRPERM`, enable it
(personal workstation): `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && sudo reboot`.

## The prediction model (discussion) and its verification

From Part 1, the report predicts interference with a **combined-demand** rule: for
each resource, if `kernelA% + kernelB% > 100%`, that resource is oversubscribed and
throttles both. Part 2 then measures the real slowdown and the report **verifies**
the prediction, splitting the misses into the two effects counters cannot see:

- **Cache capacity** — two kernels whose combined *working set* overflows a cache
  evict each other even though neither uses much cache *bandwidth* alone
  (e.g. `l1`+`l1` → ~2.8×). A footprint effect, not a rate.
- **Warp-scheduler starvation / shared FP datapath** — a kernel that saturates the
  issue slots (`fma`, 83%) starves an issue-light co-tenant; and on consumer
  Blackwell `fma`+`fp64` contend (~2.3×) because FP32-FMA and FP64 share datapath
  resources the two separate pipe counters treat as independent.

This is why Part 2 (direct measurement) is kept: it is the ground truth, and Part 1
explains *why*. The generated `reports/interference_report.md` shows the full
per-kernel table, the prediction, the measured matrix, and the verification.

## Counter-free fallback

If performance counters are unavailable, `scripts/profiler.py` estimates each
kernel's resource demand *without* NCU — by co-running it with canonical antagonist
kernels (one per resource) and measuring the slowdown. Same idea, lower fidelity.

## Profiling your own kernel

The registry lives in `code/probe.cu` → `make()`. Add a case that launches your
kernel (a single launch for Part 1, a timed loop for Part 2 — see existing
entries), rebuild, and pass its name to the scripts. No source annotations needed.
