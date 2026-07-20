# RTX 5090 – tuned experiment scripts

Drop-in replacements for the paper's H100 run scripts, retuned for this machine's
**2× RTX 5090** (Blackwell, `sm_120`, 170 SMs, 1536 threads/SM, 128 KB unified L1/SM,
~96 MB L2, 32 GB). All hardware constants live in [`common.sh`](common.sh).

## Setup (once)

The paper repo is already built for `sm_120` (see `../gpu-util-interference/CMakeLists.txt`,
arch set to `120`). If you need to rebuild:

```bash
cd ../gpu-util-interference
cmake -S . -B build && cmake --build build -j
```

`common.sh` defaults `BUILD_DIR` to `../gpu-util-interference/build` and pins
`CUDA_VISIBLE_DEVICES=0`. Override `BUILD_DIR` if you build elsewhere.

## Run

```bash
cd experiments_rtx5090
bash run_tb_scheduler.sh          # §4.1.1 thread-block scheduler   (no profiler needed)
bash run_l2.sh                    # §4.1.2 L2 cache
bash run_membw.sh                 # §4.1.3 memory bandwidth (uses MPS)
bash run_ipc.sh                   # §4.2.1 warp scheduler / IPC
bash run_l1.sh                    # §4.2.2 L1 cache
bash run_pipelines.sh             # §4.2.3 FP64/FMA pipeline
bash run_pitfall_nvidia_smi.sh    # §3 nvidia-smi is misleading
bash run_pitfall_comp_mem.sh      # §3 compute+memory still interfere
bash run_pitfall_occupancy_stream.sh   # §3 occupancy pitfall (streams)
bash run_pitfall_occupancy_mps.sh      # §3 occupancy pitfall (MPS)
```

## Reading the output

Each benchmark prints latencies for the modes it runs. The signal is always the
same comparison:

| Observation | Interpretation |
|---|---|
| colocated ≈ ½ × sequential (≈ alone) | kernels run **concurrently**, no interference |
| colocated ≈ sequential | hardware **serialized** them → **interference** on that resource |

## What still needs your input

These are correct to launch, but a few values are **starting points**, not final:

- **`run_l2.sh` / `run_l1.sh`** – the size sweeps straddle the RTX 5090 cache sizes
  but the exact interference threshold must be found by narrowing the range.
- **`run_ipc.sh` / `run_pitfall_comp_mem.sh`** – `num_itrs_*` must be tuned so the
  two kernels have **similar isolated runtimes** (run mode 1 for each and adjust).
- **`run_pitfall_occupancy_mps.sh`** – `achieved_occupancy` is a placeholder;
  profile with NCU first (command is in the script comment).
- **Profiling (mode 0 / NCU)** may require GPU performance-counter permissions.
  The latency experiments (modes 1–3) do **not** need NCU.

## Note for the slides

RTX 5090 (consumer) ≠ H100 (datacenter): the interference *mechanisms* reproduce,
but absolute thresholds (cache sizes, DRAM bandwidth, FP64 rates) differ. State
the hardware explicitly on any figure.
