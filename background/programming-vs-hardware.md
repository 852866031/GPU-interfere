# Programming Model vs. GPU Hardware

## 1. Concept-to-Hardware Overview

| Program | Definition | Executed by | Scheduled by | Hardware mapping |
|---|---|---|---|---|
| **Kernel** | A GPU function invocation — the top of the hierarchy | Multiple SMs — potentially the whole GPU | CUDA stream & command queue; blocks dispatched to SMs | **No** — a collection of work, not a physical unit |
| **Grid** | The complete set of thread blocks that **one kernel** launch creates | Multiple SMs | Not scheduled as a unit; its blocks are scheduled incrementally | **No** — a logical collection of blocks |
| **Thread block** | One partition of the **grid**: threads that synchronize and share data via shared memory | One SM, for the block's entire lifetime | Thread-block scheduler assigns each block to an SM | **Partial** — pinned to one SM, but an SM can host many blocks |
| **Warp** | A **thread block** split into groups of ~32 threads that execute in lockstep | One SMSP and its execution pipelines | Warp scheduler issues one instruction per eligible warp | **Closest match** — directly managed by hardware |
| **Thread** | A single lane within a **warp** — the smallest execution instance the programmer sees | Execution lanes in CUDA cores, Tensor Cores, LSUs, etc. | Not scheduled alone; rides along with its warp | **No** — a logical context, not a dedicated core |

---

## 2. The Two Hierarchies

**A concrete example — vector addition:**

```cuda
// __global__ marks the kernel: the code ONE thread runs.
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's index
    if (i < n) C[i] = A[i] + B[i];                  // one thread → one element
}

// Launch: <<< grid, block >>> → 4096 blocks × 256 threads
vecAdd<<<4096, 256>>>(d_A, d_B, d_C, n);
```

One launch builds the whole hierarchy: **kernel** = `vecAdd`, **grid** = 4096 blocks,
**block** = 256 threads, **warp** = 8 per block (256 / 32), **thread** = one `C[i]`.

**Programming hierarchy:**

$$\text{Kernel} \rightarrow \text{Grid} \rightarrow \text{Thread Blocks} \rightarrow \text{Warps} \rightarrow \text{Threads}$$

**Simplified hardware hierarchy:**

$$\text{GPU} \rightarrow \text{SM} \rightarrow \text{SMSP / Warp Scheduler} \rightarrow \text{Execution Pipelines}$$

> The correspondence is **not strictly one-to-one.**

---

## 3. How They Line Up

```
Programming model                  Hardware organization
─────────────────                  ────────────────────
Kernel / Grid
      │
      │ contains many thread blocks
      ▼
Thread Block  ──────────────────>  assigned to one SM
      │
      │ divided into warps
      ▼
Warp          ──────────────────>  managed by one SMSP
      │                            selected by a warp scheduler
      │ instruction issued
      ▼
Threads       ──────────────────>  executed through hardware lanes
                                   in FP/INT units, Tensor Cores,
                                   load/store units, and other pipelines
```

---

## 4. Detailed Mapping

| Program | Created by | Resource-allocation | Hardware scheduler | Execution location |
|---|---|---|---|---|
| **Kernel** | CPU program through the CUDA runtime or driver | No fixed resources allocated | GPU command processor and thread-block scheduler | Multiple SMs |
| **Grid** | Defined by the kernel-launch configuration | Collection of pending and active blocks | Not directly scheduled | Multiple SMs |
| **Thread block** | Defined by the grid and block dimensions | Registers, shared memory, thread slots, warp slots, and block slots on one SM | Thread-block scheduler | Assigned to a single SM |
| **Warp** | Automatically formed from threads in a block | Warp slots and register state associated with an SMSP | Warp scheduler | Assigned to one SMSP |
| **Thread** | Defined by the programmer | Per-thread registers and logical execution state | Follows its warp | Execution lanes (CUDA Core, SFU, etc) |

---

## 5. Key Distinctions

**Kernel and grid are collections of work.**
A kernel launch creates a grid of blocks. Neither the kernel nor the grid corresponds to one fixed physical unit.

**A thread block is the SM-level resource-allocation and residency unit.**
A block can enter an SM only if the SM has enough:
- registers,
- shared memory,
- thread capacity,
- warp slots,
- block slots.

Once assigned, a block remains on that SM until it completes.

**A warp is the hardware scheduling and instruction-issue unit.**
A warp scheduler selects an eligible warp and issues one warp instruction. The instruction applies to the active threads in that warp.

**A thread is a logical execution instance, not a physical CUDA core.**
A thread's instructions may execute through different hardware pipelines depending on instruction type:

| Instruction type | Typical hardware resource |
|---|---|
| FP32 arithmetic | FP32 pipelines / CUDA cores |
| Integer arithmetic | Integer pipelines |
| FP16/BF16 matrix operations | Tensor Cores |
| FP64 arithmetic | FP64 pipelines |
| Global-memory access | Load/store units, L1/L2 cache, HBM |
| Shared-memory access | Shared-memory pipeline |
| Special functions | Special-function units |

---

## 6. Compact Summary

> **Kernel and grid define the workload.**
> **A thread block is the SM-level placement and resource-allocation unit.**
> **A warp is the SMSP-level scheduling and instruction-issue unit.**
> **A thread is the programmer-visible logical execution instance within a warp.**
