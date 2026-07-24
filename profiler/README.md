# GPU Colocation Interference Profiler

Takes two GPU kernels and decides **whether they can run concurrently on one GPU without
much interference** — and explains *why*. Built for the "GPU interference one level
deeper" project: interference happens when two kernels lean on the **same shared
resource** at the **same scale** and their combined demand exceeds its capacity — not
when some coarse "GPU utilization" number is high.

The design is two **independent** measurement stages plus a report that ties them
together. Part 1 fingerprints each kernel *alone* (what it wants); Part 2 directly
colocates pairs (what actually happens). Part 2 is the ground truth; Part 1 explains it.

> Companion tool: `../visualizer/` **records and replays** a single workload's hardware
> usage. This tool **runs interference experiments** on kernel pairs. They share the
> `code/probe.cu` measurement harness.

---

## Quick start

```bash
cd profiler
cmake -S code -B build && cmake --build build -j     # build the probe harness (sm_120)

# the full report over a set of kernels (default: sleep dram l2 l1 fma fp64)
python3 scripts/report.py                            # -> reports/interference_report.md
```

That one command runs both stages and writes the complete report: per-kernel table →
prediction → measured interference matrix → verification.

Run either stage on its own:

```bash
python3 scripts/measure_kernels.py fma fp64          # Part 1: per-kernel counter table
python3 scripts/measure_interference.py --matrix     # Part 2: full slowdown matrix
python3 scripts/measure_interference.py l1 fma       # Part 2: one pair
python3 scripts/profiler.py l1 fma                   # counter-free fallback (no NCU)
```

Kernel names come from `build/probe list`: **`sleep dram l2 l1 fma fp64`**.

---

## How it works

```
                 ┌─ Part 1 ── measure_kernels.py ── NCU counters ─► per-kernel fingerprint
report.py  ──────┤                                                   (% of peak, 7 resources)
                 └─ Part 2 ── measure_interference.py ── probe coloc ─► measured slowdown matrix
                                       │
                             prediction (combined demand)  ⇄  verification  ─► reports/*.md
```

- **`code/probe.cu`** is the engine — a registry of 6 micro-kernels, each tuned to
  saturate exactly one shared resource, plus a timing harness. All GPU work happens
  here; the Python scripts only launch the binary and parse its stdout.
- **Part 1 (`measure_kernels.py`)** runs Nsight Compute (`ncu`) on each kernel *in
  isolation* and reads its % of peak on every shared resource. Needs perf counters.
- **Part 2 (`measure_interference.py`)** directly colocates each pair via the harness and
  measures the slowdown. **No counters** — just timing. This is the ground truth.
- **`report.py`** predicts Part 2 from Part 1 (combined-demand rule), then verifies the
  prediction against the measurement and categorizes every miss.
- **`profiler.py`** is a counter-free fallback: if NCU is unavailable, it estimates each
  kernel's demand by co-running it against canonical antagonists instead.

### The probe kernels (`code/probe.cu`)

Each stresses ~one resource, at a known scale on the interference ladder:

| kernel | what it does | resource | scale |
|---|---|---|---|
| `sleep` | `nanosleep` loop, no mem/compute | scheduler / occupancy | per-SM residency |
| `dram` | stream-copy a 512 MB array (> L2) | DRAM bandwidth | GPU-wide |
| `l2` | stream-copy a 16 MB array (L2-resident) | L2 bandwidth | GPU-wide |
| `l1` | per-block copy of a 32 KB region | L1 cache | per-SM |
| `fma` | high-ILP FP32 FMA loop | FMA pipe + issue slots | per-SMSP |
| `fp64` | high-ILP FP64 FMA loop | FP64 pipe | per-SMSP |

All launch one block per SM, isolating the per-SM/per-SMSP resource from block-scheduler
contention. `probe` commands: `list`, `static <k>` (occupancy-API residency),
`alone <k>` (median latency), `once <k>` (one launch, for clean NCU capture),
`coloc <t> <a>` (colocated slowdown), `trace <t> [a]` (`%smid`+`%globaltimer` block→SM
trace, used by the visualizer).

---

## What Part 1 measures (per kernel, % of peak)

Nsight Compute reads the GPU's hardware performance counters. Each value is a **spatial
average** across every copy of the unit (all 170 SMs, all L2 slices) and a **% of that
unit's peak sustained throughput** — how much of the resource the kernel consumes, not an
absolute rate. "Peak" is an architectural constant (per unit, per cycle), not measured.

| Resource | NCU counter | Experiment |
|---|---|---|
| SM occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | 4.1.1 |
| DRAM bandwidth | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 4.1.3 |
| L2 cache | `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.1.2 |
| L1 cache | `l1tex__throughput.avg.pct_of_peak_sustained_active` | 4.2.1 |
| Warp scheduler | `sm__inst_issued.avg.pct_of_peak_sustained_active` | 4.2.2 |
| FP32 FMA pipe | `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 4.2.2 |
| FP64 pipe | `sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active` | 4.2.3 |

**Every column is a rate (a flow), not state.** Counters count events, so they report
throughput and residency — never how much *space* a kernel holds in a cache. Two extra
"context" counters (L1/L2 hit rate, cold-miss DRAM bytes) are read as footprint hints but
are not part of the prediction. The generated report includes a legend explaining, per
column, what it really measures and what it structurally cannot.

Needs GPU **performance-counter access**. If you see `ERR_NVGPUCTRPERM`, enable it
(personal workstation): `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && sudo reboot`.
Then use `scripts/profiler.py` (counter-free) if you can't.

---

## The prediction model and its verification

From Part 1, the report predicts interference with a **combined-demand** rule: for each
shared resource, if `kernelA% + kernelB% > 100%`, that resource is oversubscribed and
throttles both. Part 2 measures the real slowdown; the report **verifies** the prediction
and splits the misses into the two effects per-kernel counters cannot see:

- **Cache capacity** — two kernels whose combined *working set* overflows a cache evict
  each other even though neither uses much cache *bandwidth* alone (e.g. `l1`+`l1` ≈ 2.5×).
  A footprint (space) effect, invisible to a throughput (rate) counter.
- **Warp-scheduler starvation / shared FP datapath** — a kernel that saturates the issue
  slots (`fma`, ~83%) starves an issue-light co-tenant even when the naive sum stays under
  100%; and on consumer Blackwell `fma`+`fp64` contend (~2.3×) because FP32-FMA and FP64
  share execution-datapath resources the two separate pipe counters treat as independent.

This is exactly why both stages exist: **Part 2 (direct colocation) is the reliable ground
truth, and Part 1 explains which resource and why.** No single utilization number can
answer whether two kernels can share a GPU.

---

## Profiling your own kernel

The registry lives in `code/probe.cu` → `make()`. Add a case that allocates your buffers
and sets `w.launch` (a single launch is enough — see the existing entries), rebuild, and
pass the new name to any script. No source annotations needed for Part 1/Part 2; the
`%smid` trace macros are already wired for `probe trace`.

---

## Repository layout

| path | what it is |
|---|---|
| `code/probe.cu` | kernel registry + timing harness (`list`/`static`/`alone`/`once`/`coloc`/`trace`) |
| `code/CMakeLists.txt` | standalone build for `sm_120` |
| `scripts/measure_kernels.py` | Part 1 — NCU per-kernel fingerprint (importable + CLI) |
| `scripts/measure_interference.py` | Part 2 — direct colocation slowdown (importable + CLI) |
| `scripts/report.py` | full report: fingerprint → prediction → matrix → verification |
| `scripts/profiler.py` | counter-free fallback (antagonist probing, no NCU) |
| `reports/` | generated markdown reports |

---

## Requirements & installation

Developed and tested on this exact stack (RTX 5090 / GB202, `sm_120`, 170 SMs):

| component | version | needed for |
|---|---|---|
| OS | Ubuntu 24.04.2 LTS | — |
| NVIDIA driver | 580.126.09 | — |
| CUDA Toolkit | 12.8 (`nvcc` V12.8.93) | build the probe harness |
| CMake | 3.26.4 | build |
| g++ | 13.3.0 | build |
| Nsight Compute (`ncu`) | 2025.1.1 | Part 1 (per-kernel counters) |
| Python | 3.12.9 | the scripts |
| numpy / pandas | 2.1.2 / 2.2.3 | optional, only if you post-process CSVs |

Install the pieces:

```bash
# nvcc, cmake toolchain, and ncu come with the CUDA Toolkit 12.8 (/usr/local/cuda-12.8)
sudo apt-get install -y cmake g++                     # if not present
# the profiler scripts use only the Python standard library — nothing to pip-install
```

- Build with `CMAKE_CUDA_ARCHITECTURES = 120` (set in `code/CMakeLists.txt`; the paper's
  upstream repo defaults to 90/H100).
- GPU **performance-counter access** must be unlocked for Part 1 / `report.py`. If you see
  `ERR_NVGPUCTRPERM`, enable it (see the Part 1 section) — or use `scripts/profiler.py`,
  which needs only timing and works without counters. Part 2 also needs no counters.
- The counter hardware is **single-client**: do not run `ncu` and `nsys` concurrently.
