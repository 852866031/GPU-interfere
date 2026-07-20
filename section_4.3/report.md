# Section 4.3 — Interference on a Real ML Kernel

**Reproduction of "Understanding GPU Resource Interference One Level Deeper" (SoCC'25), §4.3**
Hardware: **NVIDIA GeForce RTX 5090** (Blackwell, `sm_120`, 170 SMs, 4 SMSPs/SM), CUDA 12.8, driver 580, PyTorch 2.8 (cu128). Single GPU (`CUDA_VISIBLE_DEVICES=0`).

---

## 1. Introduction

Sections 4.1 and 4.2 used synthetic micro-kernels to isolate *individual* shared resources. **Section 4.3 asks the practical question: does any of this actually happen to a real, production ML kernel?** The answer is yes — and this experiment shows it on PyTorch's matrix-multiply (`torch.mm`), the single most important operation in deep-learning workloads.

The setup is a concrete instance of GPU sharing that happens all the time in practice: two jobs are packed onto one GPU to raise utilization. Here one "job" is a PyTorch matmul running in a loop (standing in for an inference or training kernel); the other is a compute-heavy CUDA kernel (standing in for a co-tenant). They run **concurrently on the same GPU** via separate CUDA streams, launched from separate Python threads.

**What this shows.** The matmul and the competitor both lean on the **FP32 fused-multiply-add (FMA) pipeline** and the **warp schedulers** inside each SM (§4.2's resources). So when they are colocated, the matmul slows down — and, crucially, *how much* depends on the matmul's **size**: a large matmul launches enough thread blocks to *flood* the GPU and dominate the shared pipeline, so it barely notices the competitor, while a small-to-medium matmul *under-fills* the GPU and must split the pipeline with it, so it gets badly hurt. A single "GPU is busy" number cannot predict this.

**Method.** For each square matrix size we measure the matmul's latency (median over 101 runs) **alone**, then **colocated** with the competitor kernel, and report the slowdown. The competitor is sized to run continuously for the whole matmul loop, so the matmul is genuinely under contention the entire time it is measured.

### 1.1 The two "jobs"

| | Job A — the ML kernel | Job B — the competitor |
|---|---|---|
| What it is | `torch.mm(A, B)`, FP32, square `N×N` matrices | custom `fma_fp32_ilp4` CUDA kernel |
| Stands in for | a real inference/training matmul | a co-located compute-bound tenant |
| Launched from | a PyTorch CUDA stream (Python thread 1) | its own non-blocking CUDA stream via CTypes (Python thread 2) |
| Grid / block | chosen by cuBLAS for the size | 170 blocks × 128 threads (1 warp per SMSP, all SMs) |
| Primary hardware use | FP32 FMA pipeline (+ memory, for large `N`) | FP32 FMA pipeline, high IPC |

Both jobs run on **all 170 SMs**, so they **share every SM** and contend for its FMA pipelines and warp schedulers — this is the intra-SM interference of §4.2, now hitting a real kernel.

### 1.2 The competitor kernel

```cuda
__global__ void fma_fp32_ilp4(float *a, float *b, float *c, long long num_itrs) {
    float op1=a[threadIdx.x], op2=b[threadIdx.x], op3=0,op4=0,op5=0,op6=0;
    for (long long i=0;i<num_itrs;i++){          // 4 independent FMA chains -> high IPC
        op3=__fmaf_rn(op1,op2,op3); op4=__fmaf_rn(op1,op2,op4);
        op5=__fmaf_rn(op1,op2,op5); op6=__fmaf_rn(op1,op2,op6);
    }
    c[threadIdx.x]=op3+op4+op5+op6;
}
```

It does nothing but stream fused-multiply-adds through the FP32 pipeline at high instruction-level parallelism (the ILP-4 idea from §2 of the 4.2 report) — a pure, sustained load on exactly the resource `torch.mm` also needs.

---

## 2. Results

We swept square matmuls from 256³ to 4096³.

![4.3 matmul interference](figures/fig_431_mm_interference.png)

| Matrix size | matmul alone | matmul colocated | **slowdown** |
|---|---|---|---|
| 256³  | 0.0140 ms | 0.0201 ms | 1.4× |
| **512³**  | 0.0139 ms | 0.0772 ms | **5.6×** |
| 1024³ | 0.0507 ms | 0.2157 ms | **4.3×** |
| 2048³ | 0.2474 ms | 0.2843 ms | 1.1× |
| 4096³ | 2.0732 ms | 2.3319 ms | 1.1× |

**How to read the figure.**
- **Left (4.3a):** the matmul's latency alone (blue) vs colocated (red), on a **log scale** because latencies span 0.014 ms to 2.3 ms. The red/blue gap is the interference; it is widest at 512³–1024³.
- **Right (4.3b):** the same data as a **slowdown ratio** (colocated ÷ alone) vs matrix size. 1.0 (dotted) would mean no interference. The curve peaks sharply at **512³ (5.6×)** and falls to ~1.1× at both ends.

---

## 3. Discussion

**The interference is real, and it is large.** A common-case FP32 matmul (512³) runs **5.6× slower** simply because a compute kernel is sharing the GPU — even though both would report "100% utilization" on their own. This is the paper's headline applied to a production kernel: colocation that looks safe by coarse metrics can be very unsafe.

**What each kernel actually uses (measured with NCU).** The matmul and the competitor both compete for the **FP32 FMA pipeline**, so the two things that matter are (a) how many thread blocks the matmul launches versus the competitor's 170 — which decides whether it *dominates* or must *share* the pipeline — and (b) how much of the FMA pipe the matmul itself uses — which decides whether it even *cares*.

| Kernel | GEMM blocks | threads/block | blocks ÷ 170 SMs | FMA pipe (alone) | occupancy | **slowdown** |
|---|---|---|---|---|---|---|
| matmul 256³ | **64** | 256 (8 warps) | 0.4× (under-fills) | 6% | 16% | 1.4× |
| matmul 512³ | **128** | 128 (4 warps) | 0.8× (under-fills) | 43% | 8% | **5.6×** |
| matmul 1024³ | **256** | 128 (4 warps) | 1.5× | 57% | 14% | 4.3× |
| matmul 2048³ | **640** | 256 (8 warps) | 3.8× | 66% | 17% | 1.1× |
| matmul 4096³ | **1536** | 256 (8 warps) | 9.0× | 71% | 17% | 1.1× |
| **FMA competitor** | **170** | 128 (4 warps) | 1.0× | **79%** | 8% | *(always on)* |

(cuBLAS selects a `cutlass_..._simt_sgemm` kernel; tile/block size varies with size. Measured via [`scripts/mm_one.py`](scripts/mm_one.py).)

**Why the slowdown is non-monotonic in size.** The curve is not "bigger = worse"; it peaks in the middle, and the block counts above explain exactly why:

- **Small (256³):** only **64 blocks** — it doesn't even fill the GPU's 170 SMs — and it uses just **6% of the FMA pipe**. It is so tiny it is launch-overhead-bound, barely touching the contended pipeline. So although the competitor's 170 blocks outnumber it, there is almost nothing to contend over → modest **1.4×**.
- **Medium (512³–1024³):** **128–256 blocks** (512³ still *under-fills* the 170 SMs) while FMA-pipe use jumps to **43–57%**. Now the matmul is genuinely pipeline-active *and* cannot out-muscle the ever-present 170-block competitor — so they split the FMA pipeline roughly evenly and the matmul is throttled hard → **4–6×**. These are precisely the small-to-medium matmuls ubiquitous in real models (attention projections, per-head GEMMs, small MLPs).
- **Large (2048³, 4096³):** **640–1536 blocks flood the GPU 4–9× over**, outnumbering the competitor's 170 blocks ~4:1 to 9:1. On any SM the matmul owns the large majority of FMA-pipe cycles and the competitor is a rounding error → slowdown collapses to **1.1×**. Note this is **not** because the matmul becomes memory-bound — DRAM stays at only 5–10% and its FMA use actually *rises* to 71%; it is safe purely because it **dominates the pipeline by block count**.

In one line: **the peak sits where the matmul uses the FMA pipe heavily (high FMA%) but launches too few blocks to dominate it (blocks ≲ 170).** 512³ is the one bad cell — 43% FMA demand, only 128 blocks.

**This is the "one level deeper" thesis, end to end.** Whether a real ML kernel suffers interference is not predictable from "is the GPU busy?" — it depends on *which* internal resource the kernel bottlenecks on (§4.2's FMA pipeline) and whether it launches enough blocks to dominate that resource. The synthetic experiments in 4.1–4.2 explained the mechanism; here it plays out on `torch.mm` with a 5.6× worst case.

**Practical takeaway for colocation / scheduling.** Packing a compute-bound tenant next to an ML serving kernel is safe *only* when the ML kernel launches enough blocks to flood the GPU and dominate the shared pipeline (the large GEMMs). For the small-to-medium, compute-bound matmuls that under-fill the GPU yet dominate latency-sensitive inference, the same packing can inflate latency several-fold. A scheduler that reasons only about occupancy or `nvidia-smi` utilization — neither of which captures "blocks launched vs SMs" together with "FMA-pipe demand" — cannot tell these cases apart.

**RTX 5090 vs H100 (paper).** The mechanism reproduces; the exact peak location and magnitude depend on the GPU's FMA throughput, the SM count (which sets the "enough blocks to dominate" threshold), and cuBLAS kernel/tile selection per size. Absolute numbers here are RTX 5090 / cuBLAS-for-`sm_120`; the *shape* — worst interference for mid-size, compute-bound matmuls that under-fill the GPU — is the transferable result.

**Limitations.** Latencies are medians of 101 runs on an otherwise-idle GPU; `torch.mm` dispatches size-dependent cuBLAS kernels, so the curve partly reflects kernel-selection boundaries as well as interference. FP32 (not TF32/FP16 tensor-core) matmul is used, to keep the shared resource the same FP32 FMA pipeline the competitor stresses; a tensor-core matmul would contend on the tensor pipeline instead. The competitor is fixed at 170×128; a stronger or weaker competitor shifts the magnitudes but not the shape. Nsight Systems (`nsys`) traces would visually confirm the two kernels overlap on the timeline.

---

## 4. How to reproduce

Requires a Python environment with CUDA-enabled PyTorch (validated: torch 2.8, cu128).

```bash
cd section_4.3
cmake -S code -B build && cmake --build build -j   # build libpython_interface.so (sm_120)
bash scripts/run_431_mm.sh                          # sweep 256..4096, ~1-2 min
python3 scripts/parse_and_plot.py                   # -> results/mm_pytorch.csv, figures/*.png
```

**Artifacts.** Raw log + parsed CSV in [`results/`](results/); figure in [`figures/`](figures/).
