# Section 4.1 — Inter-SM Interference (self-contained)

Reproduces §4.1 of *Understanding GPU Resource Interference One Level Deeper* on an
**RTX 5090**. Everything needed to build, run, analyze, and report is in this folder.

```
section_4.1/
├── code/                 # all CUDA source (self-contained build, sm_120)
│   ├── CMakeLists.txt
│   ├── gpu_util_bench_lib/   # shared kernels + timing helpers
│   └── src/                  # tb_scheduler.cu, l2_cache.cu, mem_bw.cu
├── scripts/
│   ├── common.sh            # RTX 5090 constants + paths
│   ├── run_411_tb.sh        # 4.1.1 thread-block scheduler
│   ├── run_412_l2.sh        # 4.1.2 L2 cache sweep
│   ├── run_413_membw.sh     # 4.1.3 memory bandwidth (saturation + MPS)
│   └── parse_and_plot.py    # logs -> CSVs + figures + summary
├── results/              # raw .log + parsed .csv  (generated)
├── figures/              # .png plots               (generated)
└── report.md             # full write-up: intro, results, discussion
```

## Run everything

```bash
cmake -S code -B build && cmake --build build -j
bash scripts/run_411_tb.sh
bash scripts/run_412_l2.sh
bash scripts/run_413_membw.sh
python3 scripts/parse_and_plot.py
```

Then read **[report.md](report.md)**.

## Headline results (RTX 5090)

| Experiment | Result |
|---|---|
| 4.1.1 scheduler | 1 block/SM → colocated = alone (concurrent); 2 blocks/SM → colocated = sequential (**serialized**) |
| 4.1.2 L2 cache | up to **5.6× slowdown** once combined footprint exceeds the 96 MB L2 |
| 4.1.3 bandwidth | two kernels on separate SM halves each throttled to **60%** — DRAM caps aggregate at the full-GPU peak |

## Adapting to another GPU

Edit `code/CMakeLists.txt` (`CMAKE_CUDA_ARCHITECTURES`) and the constants in
`scripts/common.sh` (`NUM_SM`, `HALF_THREADS`), plus the sweep ranges in the run
scripts (L2 size ↔ your L2; bandwidth block counts ↔ your SM count). See `report.md` §6.
