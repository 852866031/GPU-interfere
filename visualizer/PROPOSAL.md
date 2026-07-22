# Proposal — GPU Workload Visualizer (machine view + block placement replay)

## Goal

A profiling + replay tool for a **whole workload** (e.g. one LLM prefill = hundreds of
kernel launches): record what the *hardware* was doing and *where blocks physically
landed*, then replay it on an architecture-shaped GUI (die diagram + time slider).
Built on the project thesis: show **which specific resource** is stressed **at which
scale**, per phase of the workload — not one "GPU utilization" number.

Scope decision: **no NCU per-kernel fingerprinting** (no second run, no clean-room
methodology gap). Everything is captured in **one recording** of the real run. The
per-SM story comes from a block→SM placement trace, not from counters.

## What is measured with what

| # | signal | source | granularity | overhead | works on opaque kernels (PyTorch)? |
|---|---|---|---|---|---|
| S1 | kernel launch timeline: name, grid/block, stream, start/end; memcpys | **nsys** (CUPTI activity) | exact, per launch | ~none | ✅ |
| S2 | device-wide rate lanes: SM active %, warp occupancy, issue rate, tensor pipe, **L2 throughput (via custom `gb20x-l2.config` metric set — absent from NVIDIA's stock GB20x set)**, DRAM BW read+write, PCIe | **nsys `--gpu-metrics-devices`** (PM sampling) | ~100 µs samples | low | ✅ (unattributed) |
| S3 | power, clocks, temp, throttle reasons, VRAM used | **NVML** poller | ~10–100 ms | ~none | ✅ |
| S4 | **block→SM placement trace**: which SM each block ran on, block start/end times → per-SM tiles, blocks/SM over time, waves, two-tenant coloring | `%smid`+`%globaltimer` self-instrumentation (own kernels, e.g. `probe.cu`) / **NVBit** (opaque) | ~ns, per block | low–moderate | ⚠️ NVBit only |

**Derived (no extra measurement): in-situ demand estimates.** For any launch that runs
*alone* on the GPU (prefill is mostly serial) and lasts ≳100 µs, the S2 lane averages
over its S1 interval are that kernel's demand vector — measured in real context (warm
L2, boosted clocks), no isolation methodology. `analyze.py` computes these per launch
signature. Short launches smear (flagged, not reported).

**Honesty box — structurally impossible (do not fake in the GUI):**
- **Cache space held** (L1/L2 fill level): counters are flows, occupancy is state; no
  per-line ownership exists. Show throughput + hit rate only.
- **Per-SM counters**: all sampled counters are spatial averages. Per-SM resolution
  comes only from the S4 residency trace (which SM held which blocks, when) — it shows
  *placement and busy-time*, not per-SM IPC.
- **Counter attribution under concurrency**: overlapping launches blend into one
  device-wide lane value; attribution is inference (lane × who was resident).
- ~100 µs floor on the lane time axis (block trace itself is ~ns).
- **Placement is per-run**: block→SM assignment is not deterministic across runs — a
  trace describes *this* recording, which is exactly what a replay should show.

## Target use case: LLM prefill

Prefill is a kernel *parade* — embedding, RMSNorm, QKV GEMM, attention, MLP GEMMs,
per layer — alternating memory-bound and compute-bound phases. The tool makes visible:
- **rate-lane switching**: DRAM lane hot during norms/embedding, tensor/FMA lanes hot
  during GEMMs — the workload's resource *rhythm* per transformer layer;
- **underfill, spatially**: launches whose grid < 170 SMs (§4.3 danger zone) show as
  partially-lit SM grids — idle tiles are colocation headroom, visible directly;
- **wave structure and tail**: blocks/SM per wave, last-wave stragglers;
- **time attribution**: which launches own the wall clock;
- **colocation phase map**: per time bucket, dominant resource + idle-SM count →
  windows where a second workload would fit (feeds next-phase directions #2/#3).

## Workflow (single recording, no second run)

```
record.py "<workload cmd>" ──► one run:
    nsys profile --gpu-metrics-devices=0     (S1 + S2)
    NVML side-poller                         (S3)
    [--trace] LD_PRELOAD NVBit tool          (S4, opaque kernels)
                                             (own kernels: probe's built-in trace)
        │
extract.py ──► runs/<name>/kernels.csv  metrics.csv  nvml.csv  blocks.csv
        │
analyze.py ──► report.md   (time-share, in-situ demand per launch signature,
        │                   phase map, underfill/idle-SM windows)
        ▼
replay GUI (single-file HTML canvas) — die layout: 170 SM tiles lit by resident
blocks (colored per kernel/stream), L2/DRAM/PCIe gauges, power/clock strip,
launch lanes, time slider.
```

Compatibility caveat (M1 must verify): NVBit (`LD_PRELOAD` SASS injection) and nsys
(CUPTI) both hook the runtime and may conflict. If they can't share a run, fallback:
record twice — lanes run (nsys) + trace run (NVBit) — and align by launch sequence.
Lanes stay exact; placements are then *representative* rather than same-run (flagged
in the report). Own-kernel (`%smid`) tracing has no such conflict.

## Milestones

| M | deliverable | new code | validation |
|---|---|---|---|
| M1 | `record.py` + `extract.py` → CSVs; gpu-metrics availability check on 5090; NVBit×nsys compat check | nsys wrapper + sqlite SQL | probe kernels: `dram` lights only the DRAM lane, `fma` only issue/FMA lanes |
| M2 | block trace for own kernels: `probe trace <k>` → blocks.csv | ~20 lines CUDA + dump | blocks/SM matches occupancy-API prediction; §4.1.1 serialization visible |
| M3 | `analyze.py` → `report.md` (incl. in-situ demand) | join + tables | in-situ vector for `probe dram` ≈ its known Part-1 row |
| M4 | replay GUI: machine lanes + per-SM block animation | single-file HTML/JS | `coloc l1 fma` replay: both colors on same tiles + issue lane pegged |
| M5 | NVBit trace tool for opaque kernels | NVBit C++ tool | torch.mm placements consistent with grid size / §4.3 underfill |
| M6 | LLM prefill demo | none (usage) | torch 2.8 small-model prefill; phase map + idle-SM windows |

Validation strategy throughout: **run the tool on workloads whose answers we already
measured** (probe registry, §4.1–4.3) before pointing it at an LLM.

## NVBit feasibility check — RESULT (2026-07-21)

**sm_120 itself: supported.** NVBit ≥1.7.4 lists SM_120; v1.7.5 and v1.8 build with
nvcc 12.8 and instrument our own binaries on the RTX 5090 exactly (opcode_hist on
`probe alone fp64`: 1,088,000,000 DFMA = 170 blk × 4 warps × 400k it × 4 — exact),
via both `LD_PRELOAD` and `CUDA_INJECTION64_PATH`, driver 580.126.09.

**Root cause (bisected minimally, 2 earlier guesses were WRONG): NVBit's callbacks do
not fire in a `python3` process in THIS environment.** Not torch-, CUPTI-, or
cuBLAS-specific.

Bisection (v1.7.5 & v1.8, driver 580, sm_120), each the SAME trivial `k_lib` kernel:
| host process | NVBit result |
|---|---|
| C exe (kernel in exe) | ✅ instruments |
| C exe dlopening `libmini.so` (early or late CUDA) | ✅ instruments |
| **python3 `ctypes.CDLL('libmini.so').run()`** | ❌ zero events (banner loads, no callback) |
| torch | ❌ zero (same as plain python) |

The identical `.so` is instrumented from any C host but never from python3. Eliminated
as causes: CUPTI slot (full 19-sym libcupti stub → torch runs with no real CUPTI, still
zero; stub doesn't break NVBit on mini), lazy module loading (EAGER doesn't help),
wheel vs system cudart (both work for C), libcuda identity (torch uses system libcuda),
re-exec (none; banner loads once), symbol scope (RTLD_GLOBAL doesn't help), cuBLAS/
cuLibrary (a plain elementwise `add_` and even a trivial ctypes kernel fail too).

In python, neither `nvbit_at_cuda_event` (per-kernel) nor `nvbit_at_term` (exit) fires —
NVBit's driver callbacks never activate, though the tool's constructor runs (banner).
Even an explicit shell `CUDA_INJECTION64_PATH=nvbit` (the official driver-injection path,
which works for the C loader) does not activate NVBit in python. Tested a FRESH conda env
(python 3.11) — same failure, so NOT conda/env-specific; it's python-generic on this box.
Suspected: **driver 580.126.09 is newer than NVBit v1.8's tested ceiling** (issue
trackers cite ~575) — the python code path is where the incompatibility surfaces while
plain compiled binaries still work. Not confirmable without downgrading the driver.

**Decision: M5 stays deferred regardless**, because even if the python hook were fixed:
(a) NVBit is mutually exclusive with nsys, so per-SM data could never co-record with the
gpu-metrics lanes — always a separate pass with non-deterministic placement; (b) full
instrumentation overhead on an 8B model is heavy. Opaque-kernel runs keep the honest
device-avg tile fallback. `probe trace` (NVBit-style %smid in our OWN compiled binary)
is the working path and already gives per-SM data for the probe kernels.

## Risks

- GeForce support for `--gpu-metrics-devices` metric set (expected OK on 5090; checked
  first thing in M1).
- NVBit ↔ nsys coexistence (fallback: two-pass recording, above); NVBit version must
  match CUDA 12.8.
- NVBit overhead perturbs timing (entry/exit-only instrumentation keeps it low; report
  records overhead vs an uninstrumented run).
- nsys version drift in sqlite schema → pin version in README.
- Kernel-name mangling in PyTorch → demangle + dedup pass.
- Clock DVFS wobble across recordings → optionally lock clocks for comparable runs
  (needs sudo; ask before touching system config).
