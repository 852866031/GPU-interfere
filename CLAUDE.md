# CLAUDE.md — GPU Resource Interference Project

## What this project is

We are studying **GPU resource interference when two kernels are colocated** on one
GPU, based on the paper **"Understanding GPU Resource Interference One Level Deeper"**
(SoCC'25; arXiv 2501.16909) and its benchmark suite. The work so far has two outputs:

1. **A reproduction + teaching deliverable** — we re-ran the paper's experiments on
   local hardware, tuned them, plotted them, and wrote per-section reports. These
   feed a **presentation** (the user builds the slides; Claude generates the prose,
   figures, tables, and pseudocode — Claude does **not** see the slides).
2. **A colocation interference profiler** — a tool, built from the paper's insight,
   that predicts whether two kernels can share a GPU without much interference.

**The core thesis** (keep this in mind for everything): whether two kernels interfere
depends on **which specific shared resource** they contend for, and at **what scale**
it is shared — not on any coarse "GPU utilization" number. Interference lives "one
level deeper."

## Hardware & build environment

- **GPU:** 2× NVIDIA GeForce RTX 5090 (Blackwell, `sm_120`). **170 SMs**, 4 SMSPs/SM,
  **1536 threads/SM** (48 warps), **128 KB unified L1/SM**, **~96 MB L2**, 32 GB GDDR7,
  ~1200 GB/s achieved DRAM BW. Always pin to one card: `CUDA_VISIBLE_DEVICES=0`.
- **Toolchain:** CUDA 12.8 (`nvcc`), CMake 3.26, g++ 13.3. Build with
  `CMAKE_CUDA_ARCHITECTURES = 120` (the paper's repo defaults to 90/H100).
- **NCU is enabled:** GPU performance counters were unlocked (option 2:
  `NVreg_RestrictProfilingToAdminUsers=0` + reboot), so `ncu` reads real counters.
- **Python:** torch 2.8 (cu128) available; matplotlib/numpy/pandas for plotting.
- Do **not** install packages or modify system config without asking.

## Repository layout

| Path | What it is |
|---|---|
| `gpu-util-interference/` | The paper's original benchmark suite (upstream code). |
| `background/` | Slide background material (programming↔hardware mapping, `vecadd.cu`). |
| `section_4.1/` | **Inter-SM** experiments (kernels on *separate* SMs): thread-block scheduler, L2 cache, DRAM bandwidth. |
| `section_4.2/` | **Intra-SM** experiments (kernels *share* SMs): L1 cache, warp scheduler (IPC), FP64 pipeline. |
| `section_4.3/` | **Real ML kernel**: PyTorch `torch.mm` colocated with a compute kernel. |
| `profiler/` | The colocation interference profiler (two measurement stages + report). |
| `experiments_rtx5090/` | RTX-5090-tuned copies of the paper's run scripts. |
| `EXPERIMENTS_SUMMARY.md` | One-page ladder of all experiments, ordered by shared-resource scale. |

Each `section_*/` is **self-contained**: `code/` (copied CUDA + standalone CMake for
`sm_120`), `scripts/` (run scripts + `parse_and_plot.py`), `results/` (logs + CSVs),
`figures/` (PNGs), `report.md` (intro → workloads → per-experiment setup/results/
discussion), `README.md`. `build/` dirs are generated (gitignored upstream).

Reproduce any section: `cmake -S code -B build && cmake --build build -j` then
`bash scripts/run_*.sh` then `python3 scripts/parse_and_plot.py`.

## Key results (all measured on RTX 5090)

Interference organized largest→smallest shared resource (this is the talk's spine —
GPU-wide → per-SM → per-SMSP, mirroring the GPU→SM→SMSP zoom):

| Scope | Experiment | Shared resource | Result |
|---|---|---|---|
| GPU-wide | 4.1.3 DRAM bandwidth | memory controllers | each kernel throttled to ~60%; MPS does **not** help |
| GPU-wide | 4.1.2 L2 cache | 96 MB L2 | **capacity cliff**, up to 5.6×; converges to serial at the far end (falls back to scarce DRAM) |
| per-SM | 4.1.1 thread-block scheduler | thread/warp/block slots | 2 blocks/SM ⇒ serialized (2×) |
| per-SM | 4.2.1 L1 cache | 128 KB L1 | capacity cliff up to ~6×; **>64 KB the fall-off is 2× *faster* than serial** (spills to roomy L2) |
| per-SMSP | 4.2.2 warp scheduler (IPC) | issue slots (4/cycle) | complementary copy+compute still interfere; compute alone already 84% of issue slots |
| per-SMSP | 4.2.3 FP64 pipeline | FP64 units | colocated = sequential at all ILP (consumer FP64 ~1/64 of FP32) |
| — | 4.3 ML matmul | FMA pipe | slowdown **non-monotonic in size** (peaks 5.6× at 512³); depends on block-count vs #SMs |

Two recurring mechanisms: **capacity cliffs** (L1 & L2 — sudden, not gradual) and
**serialization** (scheduler & pipeline — colocated ≈ 2×). Gotchas documented in the
reports: the 4.2.2 copy must be **L2-resident** (issue-active) or it doesn't interfere;
the profiler must filter NCU output to bandwidth-only for mem_bw; parser handles
interleaved C/Python stdout.

## The profiler (`profiler/`)

Takes two kernels, decides if they can be colocated without much interference. Two
independent stages + a report:

- `scripts/measure_kernels.py` — **Part 1**: NCU-profiles each kernel alone → % of peak
  on 7 resources (occupancy, DRAM, L2, L1, issue slots, FMA pipe, FP64 pipe).
- `scripts/measure_interference.py` — **Part 2**: directly colocates pairs → measured
  slowdown (no counters needed).
- `scripts/report.py` — one report: per-kernel table → discussion/prediction
  (combined-demand `A%+B% > 100%`) → measured matrix → verification.
- `scripts/profiler.py` — counter-free fallback (antagonist probing) if NCU is locked.
- `code/probe.cu` — kernel registry + `once`/`alone`/`coloc`/`static` harness.

Known limits (stated honestly in the report): the combined-demand model captures
*rate* contention (bandwidth, pipes, issue slots) but **not cache-capacity thrashing**
(a footprint effect), so the verdict is driven by the **measured** slowdown and the
counters explain *why*. A memory kernel is latency-bound so it issues little (~12%),
which is why "complementary" memory+compute is near-saturation, not overbooking.

## Working conventions

- **Slides:** Claude can't see them. Produce markdown, figures (matplotlib → PNG),
  tables, and pseudocode. Slide bullets: concise, prefer `→`/`:` over dash connectors
  (the user dislikes "-" as a sentence connector). Latency plots: lower = better;
  bandwidth plots: higher = better — always label which.
- **Figures** live in `section_*/figures/`; regenerate with the section's
  `parse_and_plot.py`. Keep the `alone / 2×alone (serial) / colocated` framing.
- **Numbers must be measured**, never invented. Re-run rather than guess. Note
  run-to-run variance in the cache-thrash regions.
- Memory (persistent facts) lives in the auto-memory dir, not here.

## Next phase — build something the paper inspires

The reproduction and the profiler are done. The next step is to **develop a new system
or tool** that uses the paper's central lesson (interference = contention for a
*specific* shared resource at a *specific* scale). Candidate directions to discuss and
pick from:

1. **Interference-aware colocation scheduler.** Given a set of kernels/jobs and their
   resource fingerprints (from the profiler), decide *which pairs to co-run* and *how*
   (CUDA streams vs MPS SM-split vs don't colocate) to maximize throughput while
   bounding each job's slowdown. The profiler already gives the inputs; this adds the
   packing/decision policy. Maps cleanly onto the MPS-helps-intra-SM /
   MIG-needed-for-inter-SM distinction.
2. **Colocation advisor for ML serving.** Profile real inference kernels (attention,
   GEMMs, norms) and predict which can safely share a GPU — directly extends §4.3,
   where small/medium matmuls (under-filling the GPU) are the danger zone.
3. **Online interference detector.** A lightweight runtime monitor that, using the
   cheap counter fingerprints, flags when two colocated production kernels are
   contending on a specific resource, and recommends a fix (separate SMs, serialize,
   resize).
4. **Better prediction model.** Add a footprint-based cache-capacity term (from
   cold-miss traffic / hit rate) so the profiler predicts cache thrashing without
   needing the direct-colocation ground truth.

When starting the next phase: (a) confirm which direction with the user, (b) reuse the
`profiler/` primitives (`probe.cu`, `measure_*.py`) as the measurement layer, and
(c) keep the "which resource, which scale" framing as the design spine.
