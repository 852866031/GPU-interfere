# Experiments Summary — Shared Resources, Largest → Smallest

Every experiment colocates two kernels and measures how much they slow each other
down. What differs is **which shared resource they contend for** — and those
resources live at very different scales, from one GPU-wide unit serving all SMs
down to one unit per SM sub-partition. The table is ordered by that scale.

**Scale of the shared hardware on the RTX 5090** (why "L2/DRAM = large"):

```
GPU  ── DRAM / memory controllers ── 1 pool, serves all 170 SMs   ── LARGEST
     └─ L2 cache (~96 MB) ────────── 1 cache,  serves all 170 SMs
        └─ SM  (×170) ── thread/block slots, L1 cache ── per-SM
             └─ SMSP (×680) ── warp scheduler, exec pipelines ── SMALLEST
```

## Summary table

| # | Scope (large→small) | Experiment | Shared resource | Copies on RTX 5090 | Worst slowdown | Interference mechanism |
|---|---|---|---|---|---|---|
| 1 | **GPU-wide** | §4.1.3 Memory bandwidth | DRAM bandwidth (memory controllers / GDDR7) | **1** pool (all 170 SMs) | each kernel → **60%** (aggregate capped at peak) | fixed bandwidth budget split — two kernels on *separate* SMs still collide |
| 2 | **GPU-wide** | §4.1.2 L2 cache | L2 cache (~96 MB) | **1** cache (all 170 SMs) | **5.6×** | capacity cliff — combined footprint > L2 → mutual eviction |
| 3 | **GPU-wide sched → per-SM slots** | §4.1.1 Thread-block scheduler | per-SM thread / warp / block slots (dispatched by the GPU-wide block scheduler) | 1 scheduler → **170** SMs' slots | **2.0×** | serialization — once each kernel fills an SM's slots, blocks can't co-reside |
| 4 | **per-SM** | §4.2.1 L1 cache | L1 data cache (128 KB / SM) | **170** (1 per SM) | **6.0×** | capacity cliff — combined footprint > L1 (kernels on *different* SMSPs) |
| 5 | **per-SMSP** | §4.2.2 Warp scheduler (IPC) | instruction-issue slots (4 / SM) | **680** (4 per SM) | **1.4×** (40% toward serial) | issue-slot contention — even different exec units share the issue stage |
| 6 | **per-SMSP** | §4.2.3 Compute pipeline | FP64 execution pipeline | **680** (per SMSP) | **2.0×** | pipeline serialization — same rate-limited units, zero overlap |
| — | **per-SMSP (real kernel)** | §4.3 ML matmul | FP32 FMA pipeline + warp scheduler | **680** | **5.6×** | production `torch.mm` vs compute kernel — §4.2 mechanisms on a real workload |

## How to read the "scope"

- **GPU-wide (1 copy):** the resource is shared by *every* SM. Two kernels interfere
  **even if they run on completely separate SMs** — you cannot escape it by
  partitioning SMs (MPS does not help). These are §4.1's inter-SM experiments.
- **Per-SM (170 copies):** the resource is private to one SM but shared by that SM's
  4 SMSPs. Kernels interfere **only if they land on the same SM**; MPS/SM-partitioning
  *can* help.
- **Per-SMSP (680 copies):** the smallest scope — one warp scheduler / pipeline per
  sub-partition. Contention needs the kernels' warps on the same SMSP.

## The one-sentence point

Whether two kernels interfere — and how badly — depends entirely on *which* of these
resources they jointly stress, and at *what scale* it is shared. A single whole-GPU
"utilization" number collapses all six rows into one and therefore cannot predict any
of it — which is the paper's "one level deeper" thesis.

## Two mechanisms recur at multiple scales

- **Capacity cliff** appears at both cache levels — L2 (§4.1.2, GPU-wide) and L1
  (§4.2.1, per-SM) — with the same shape: fine until the *combined* working set
  overflows, then a sudden multi-× collapse (worst cases here, 5.6× and 6.0×).
- **Serialization** appears at both the scheduler (§4.1.1, blocks can't co-reside)
  and the pipeline (§4.2.3, same execution units) — colocated ≈ sequential (2×).
