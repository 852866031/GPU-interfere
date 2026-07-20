# Section 4.2 — Intra-SM Interference (self-contained)

Reproduces §4.2 of *Understanding GPU Resource Interference One Level Deeper* on an
**RTX 5090**. Companion to `../section_4.1` (inter-SM). Everything needed to build,
run, analyze, and report is here.

```
section_4.2/
├── code/                 # all CUDA source (self-contained build, sm_120)
│   ├── CMakeLists.txt
│   ├── gpu_util_bench_lib/   # shared kernels + timing helpers
│   └── src/                  # l1_cache.cu, ipc.cu, pipelines.cu
├── scripts/
│   ├── common.sh            # RTX 5090 constants + paths
│   ├── run_421_l1.sh        # 4.2.1 L1 cache sweep
│   ├── run_422_ipc.sh       # 4.2.2 warp scheduler (copy + compute)
│   ├── run_423_pipelines.sh # 4.2.3 FP64 pipeline (ILP sweep)
│   └── parse_and_plot.py    # logs -> CSVs + figures + summary
├── results/              # raw .log + parsed .csv  (generated)
├── figures/              # .png plots               (generated)
└── report.md             # full write-up: intro, results, discussion
```

## Run everything

```bash
cmake -S code -B build && cmake --build build -j
bash scripts/run_421_l1.sh
bash scripts/run_422_ipc.sh
bash scripts/run_423_pipelines.sh
python3 scripts/parse_and_plot.py
```

Then read **[report.md](report.md)**.

## Headline results (RTX 5090)

| Experiment | Shared SM resource | Result |
|---|---|---|
| 4.2.1 L1 cache | per-SM L1 (128 KB) | up to **6× slowdown** once combined footprint > L1 (kernels on *different* SMSPs) |
| 4.2.2 warp scheduler | instruction-issue slots | complementary copy+compute still **+40% toward serialization** when both are issue-bound |
| 4.2.3 FP64 pipeline | FP64 execution units | **colocated = sequential** at every ILP (total serialization; consumer FP64 is tiny) |

## Key idea vs Section 4.1

4.1 kept the two kernels on **separate** SMs and found they still interfere via shared
GPU-wide resources (scheduler, L2, DRAM). 4.2 forces them onto the **same** SM and finds
they interfere via shared *internal* resources (L1, warp scheduler, pipelines) — even
when a coarser view (different SMSPs, "memory vs compute", low occupancy) says they
should be independent. See `report.md` §6.
