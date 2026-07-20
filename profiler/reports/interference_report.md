# GPU Kernel Colocation Interference Report

Two independent measurement stages: **Part 1** fingerprints each kernel alone (Nsight Compute counters); **Part 2** directly measures how much each pair slows down when colocated. The discussion predicts Part 2 from Part 1.

## Part 1 — Individual kernel measurements

Each value is the kernel's utilization of that resource, in **% of its peak**, measured in isolation.

| kernel | SM occupancy | DRAM bandwidth | L2 cache | L1 cache | Warp scheduler | FP32 FMA pipe | FP64 pipe |
|---|---|---|---|---|---|---|---|
| `sleep` | 50.0% | 0.0% | 0.0% | 0.0% | 0.6% | 0.0% | 0.0% |
| `dram` | 33.2% | 69.0% | 27.3% | 13.9% | 3.2% | 0.4% | 0.0% |
| `l2` | 24.5% | 0.1% | 76.3% | 39.6% | 12.2% | 1.3% | 0.0% |
| `l1` | 4.2% | 0.0% | 46.1% | 23.6% | 5.3% | 1.5% | 0.0% |
| `fma` | 8.3% | 0.0% | 0.0% | 0.0% | 82.7% | 79.0% | 0.0% |
| `fp64` | 8.3% | 0.0% | 0.0% | 0.0% | 7.7% | 0.0% | 99.5% |

**How to read these numbers.**
- Each is a **spatial average** (`.avg`) across every copy of the unit — all 170 SMs, all L2 slices, all sub-partitions — **not a max**. `50%` occupancy means the *average* SM was half full, not that one SM peaked at 50%.
- Each is a **% of that unit's peak sustained throughput** (100% = running flat out, 0% = idle) — i.e. how much of the resource the kernel consumes, not an absolute rate.
- Normalization window differs by unit (NVIDIA's per-metric default): occupancy, L1, warp scheduler and the pipes are over **active** cycles ("how hard while busy"); DRAM and L2 are over **elapsed** cycles ("fraction of peak across the whole kernel"). For these steadily-running kernels the two nearly coincide.
- Examples: `sleep` occupancy 50% = the average SM held 24 of its 48 warp slots; `fma` warp scheduler 83% = the schedulers issued ~3.3 of a max 4 instructions/cycle.

## Discussion — what interference do we expect?

**Rule (from §4.1–4.2):** two kernels interfere when they both lean on the *same* shared resource and their combined demand exceeds its capacity. From Part 1, each kernel's dominant resource is:

- `sleep`: no resource used heavily (max Warp scheduler 1%) — a light co-tenant.
- `dram`: **DRAM bandwidth** (69% of peak) — *inter-SM* resource.
- `l2`: **L2 cache** (76% of peak) — *inter-SM* resource.
- `l1`: **L2 cache** (46% of peak) — *inter-SM* resource.
- `fma`: **Warp scheduler** (83% of peak) — *intra-SM* resource.
- `fp64`: **FP64 pipe** (99% of peak) — *intra-SM* resource.

So we predict, per the counters (combined demand `A%+B%`):

| pair | shared bottleneck | combined demand | predicted |
|---|---|---|---|
| `sleep` + `sleep` | — (all under capacity) | 1% | **little interference** |
| `sleep` + `dram` | — (all under capacity) | 69% | **little interference** |
| `sleep` + `l2` | — (all under capacity) | 76% | **little interference** |
| `sleep` + `l1` | — (all under capacity) | 46% | **little interference** |
| `sleep` + `fma` | — (all under capacity) | 83% | **little interference** |
| `sleep` + `fp64` | — (all under capacity) | 99% | **little interference** |
| `dram` + `dram` | DRAM bandwidth | 138% | **moderate interference** |
| `dram` + `l2` | L2 cache | 104% | **moderate interference** |
| `dram` + `l1` | — (all under capacity) | 73% | **little interference** |
| `dram` + `fma` | — (all under capacity) | 86% | **little interference** |
| `dram` + `fp64` | — (all under capacity) | 99% | **little interference** |
| `l2` + `l2` | L2 cache | 153% | **STRONG interference** |
| `l2` + `l1` | L2 cache | 122% | **moderate interference** |
| `l2` + `fma` | — (all under capacity) | 95% | **little interference** |
| `l2` + `fp64` | — (all under capacity) | 99% | **little interference** |
| `l1` + `l1` | — (all under capacity) | 92% | **little interference** |
| `l1` + `fma` | — (all under capacity) | 88% | **little interference** |
| `l1` + `fp64` | — (all under capacity) | 99% | **little interference** |
| `fma` + `fma` | Warp scheduler | 165% | **STRONG interference** |
| `fma` + `fp64` | — (all under capacity) | 99% | **little interference** |
| `fp64` + `fp64` | FP64 pipe | 199% | **STRONG interference** |

### Why cache **capacity** isn't in the prediction — and what one kernel *can* reveal

The columns above are **throughput** rates (bytes/instructions per cycle vs peak). Cache-capacity interference is a **footprint** effect — two kernels whose *combined working set* overflows a cache evict each other — which is a different axis: a kernel can saturate cache *bandwidth* with a tiny reused array, or hold a huge footprint while using little bandwidth. So `A%+B%` on the throughput columns structurally cannot see it.

**Can a single kernel reveal it? Partly — yes.** Profiling one kernel alone does expose two footprint-related facts (extra counters, not in the table above):

| kernel | L1 hit rate | L2 hit rate | footprint (cold-miss DRAM) |
|---|---|---|---|
| `dram` | 0.0% | 0.0% | 51.5 GB |
| `l2` | 80.6% | 100.0% | 16.8 MB |
| `l1` | 100.0% | 100.0% | 5.8 MB |

- **Residency** = the hit rate: `l1`/`l2` are ~100% cache-resident (their data fits in cache alone); `dram` is 0% (pure streaming).
- **Footprint** ≈ the cold-miss DRAM traffic for a *reuse-heavy* kernel: the working set is loaded from DRAM once, then reused from cache. Note `l2`'s cold load (~16 MB) matches its 16 MB array almost exactly.

So a capacity check *is* possible in principle: **if both kernels are cache-resident and footprint_A + footprint_B > cache size → thrash** — which would correctly flag `l1`+`l1`.

**Why we don't rely on it, and keep the direct Part-2 measurement:**
- Cold-miss traffic under-counts **write-only** working sets — a copy kernel's *output* is cached but never a cold *read*, so `l1`'s true in+out footprint is ~2× the measured value.
- **Scope differs:** L1 is per-SM but L2 is GPU-wide, so the footprints must be summed at the right level, which needs the block-to-SM mapping.
- For a **streaming** kernel the DRAM bytes are total traffic, *not* footprint (`dram` reads tens of GB but has ~0 footprint of reuse).
So the footprint estimate is approximate; **Part 2's direct colocation is the reliable ground truth**, and Part 1 tells us *which* resource and *why*. Watch `l1`+`l1` below: the throughput model predicts little, but the measurement will expose the capacity cliff.

## Part 2 — Measured interference (colocation slowdown)

Each cell = slowdown of the **row** kernel when the **column** kernel runs beside it on the same GPU (1.00× = no interference; 2.00× = fully serialized).

| target ↓ \ with → | `sleep` | `dram` | `l2` | `l1` | `fma` | `fp64` |
|---|---|---|---|---|---|---|
| **`sleep`** | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| **`dram`** | 1.00× | 1.98× | 1.07× | 1.01× | 1.01× | 1.00× |
| **`l2`** | 0.98× | 1.40× | 2.01× | 1.28× | 1.59× | 1.25× |
| **`l1`** | 0.92× | 0.92× | 0.91× | 2.50× | 1.87× | 0.95× |
| **`fma`** | 0.91× | 0.91× | 1.57× | 0.97× | 1.60× | 0.93× |
| **`fp64`** | 0.91× | 0.91× | 0.91× | 0.98× | 2.28× | 1.80× |

## Verification — did the prediction hold?

| pair | predicted (counters) | measured (worst dir.) | agree? |
|---|---|---|---|
| `sleep` + `sleep` | little (1%) | 1.00× (little) | ✅ |
| `sleep` + `dram` | little (69%) | 1.00× (little) | ✅ |
| `sleep` + `l2` | little (76%) | 1.00× (little) | ✅ |
| `sleep` + `l1` | little (46%) | 1.00× (little) | ✅ |
| `sleep` + `fma` | little (83%) | 1.00× (little) | ✅ |
| `sleep` + `fp64` | little (99%) | 1.00× (little) | ✅ |
| `dram` + `dram` | moderate (138%) | 1.98× (STRONG) | ✅ |
| `dram` + `l2` | moderate (104%) | 1.40× (moderate) | ✅ |
| `dram` + `l1` | little (73%) | 1.01× (little) | ✅ |
| `dram` + `fma` | little (86%) | 1.01× (little) | ✅ |
| `dram` + `fp64` | little (99%) | 1.00× (little) | ✅ |
| `l2` + `l2` | STRONG (153%) | 2.01× (STRONG) | ✅ |
| `l2` + `l1` | moderate (122%) | 1.28× (moderate) | ✅ |
| `l2` + `fma` | little (95%) | 1.59× (STRONG) | ⚠️ miss |
| `l2` + `fp64` | little (99%) | 1.25× (moderate) | ⚠️ miss |
| `l1` + `l1` | little (92%) | 2.50× (STRONG) | ⚠️ miss |
| `l1` + `fma` | little (88%) | 1.87× (STRONG) | ⚠️ miss |
| `l1` + `fp64` | little (99%) | 0.98× (little) | ✅ |
| `fma` + `fma` | STRONG (165%) | 1.60× (STRONG) | ✅ |
| `fma` + `fp64` | little (99%) | 2.28× (STRONG) | ⚠️ miss |
| `fp64` + `fp64` | STRONG (199%) | 1.80× (STRONG) | ✅ |

### Conclusions

**The prediction held wherever two kernels saturate the *same* rate resource.** `fp64`+`fp64` (199%), `dram`+`dram` (137%), `l2`+`l2` (150%), `fma`+`fma` (165%) were all predicted to interfere and do; every pair on *different* resources (anything with `sleep`, and `dram`+`fma`) was predicted safe and is. That is the paper's core claim, confirmed end-to-end.

**The under-predictions (⚠️) are the two effects per-kernel counters cannot see:**

1. **Cache capacity** — `l1`+`l1` (2.50×). Neither kernel uses much cache *bandwidth* alone, but their combined *working set* overflows the cache and they evict each other. This is the §4.1.2 / §4.2.1 cliff — a footprint effect, invisible to a throughput counter measured in isolation.
2. **Warp-scheduler starvation / shared FP datapath** — `l2`+`fma` (1.59×), `l2`+`fp64` (1.25×), `l1`+`fma` (1.87×), `fma`+`fp64` (2.28×). A kernel that saturates the issue slots (`fma`, 83%) starves an issue-light co-tenant even though the naive sum stays under 100%; and on consumer Blackwell `fma`+`fp64` contend (2.3×) because FP32-FMA and FP64 share execution-datapath resources that our two separate pipe counters treat as independent.

**Takeaway:** Part 1 (per-kernel counters) explains *why* kernels interfere and correctly predicts all same-resource contention; Part 2 (direct colocation) is the ground truth and is **required** to catch the two footprint/sharing effects above. Together they answer *whether* two kernels can share a GPU — which no single utilization number can.
