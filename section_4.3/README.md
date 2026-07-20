# Section 4.3 — Interference on a Real ML Kernel (self-contained)

Reproduces §4.3 of *Understanding GPU Resource Interference One Level Deeper* on an
**RTX 5090**: a PyTorch `torch.mm` matmul colocated with a custom FP32-FMA compute
kernel. Companion to `../section_4.1` (inter-SM) and `../section_4.2` (intra-SM).

```
section_4.3/
├── code/
│   ├── CMakeLists.txt
│   └── python_interface.cu   # fma_fp32_ilp4 competitor, exposed to Python via CTypes
├── scripts/
│   ├── run_43.py             # driver: sweeps matrix sizes, mm alone vs colocated
│   ├── run_431_mm.sh         # wrapper (sets sizes / competitor config)
│   └── parse_and_plot.py     # log -> CSV + figure
├── results/                  # raw .log + parsed .csv  (generated)
├── figures/                  # .png plot                (generated)
└── report.md                 # full write-up: intro, results, discussion
```

## Requirements

CUDA-enabled **PyTorch** in your Python env (validated: torch 2.8, cu128) plus the
same CUDA toolchain as the other sections.

## Run

```bash
cmake -S code -B build && cmake --build build -j
bash scripts/run_431_mm.sh
python3 scripts/parse_and_plot.py
```

Then read **[report.md](report.md)**.

## Headline result (RTX 5090)

A PyTorch FP32 matmul colocated with a compute kernel slows down by an amount that
**depends on the matmul size**:

| Matrix | slowdown |
|---|---|
| 256³ | 1.4× |
| **512³** | **5.6×** |
| 1024³ | 4.3× |
| 2048³ | 1.1× |
| 4096³ | 1.1× |

Mid-size, compute-bound matmuls (which don't fill the GPU) suffer most; large
matmuls that already saturate/turn memory-bound barely notice. Whether a real ML
kernel is hurt by colocation is **not** predictable from "is the GPU busy?" — see
`report.md` §3.
