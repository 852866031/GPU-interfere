# GPU Workload Replay Visualizer

Record what an NVIDIA GPU actually does while running a workload — every kernel launch,
device-wide resource rates, per-SM block placement, power/clocks — then **replay it** on
an architecture-shaped web GUI with a time slider.

Built for the "GPU interference one level deeper" project: the point is to show **which
specific resource** is stressed **at which scale** (GPU-wide → per-SM → per-SMSP) over
the life of a workload — not one meaningless "GPU utilization" number. It reuses the
`../profiler/` measurement primitives (`probe.cu`) as its own-kernel tracing layer.

> Scope: this tool **observes and replays one workload**. It does not run interference
> experiments or predict slowdowns — that is the `../profiler/` tool's job.

---

## Quick start

```bash
cd visualizer

# 1. record a workload  (writes recordings/<name>/ and runs/<name>.json)
python3 record.py llama3 --capture -- python3 workloads/llama_prefill.py

# 2. launch the web server
bash serve.sh                    # -> http://localhost:8000

# 3. open http://localhost:8000, pick the run, press ▶ Play
```

A synthetic `dummy_prefill` run is always served (no recording needed) so the GUI works
out of the box.

---

## How it works

Four stages. Everything the GPU did is captured in **one recording**; the GUI is a pure
offline replayer of the resulting JSON.

```
record.py  ──►  recordings/<name>/{report.nsys-rep, nvml.csv, meta.json, stdout.txt}
                     │
extract.py  ────────►  runs/<name>.json         (the "run contract" — see dummy_run.py)
                     │
server.py  ─────────►  GET /api/runs, /api/runs/<name>   +   static web/
                     │
web/ (index+app+css) ─►  replay in the browser, driven entirely client-side
```

- **`record.py <name> -- <cmd>`** wraps the workload command in Nsight Systems
  (`nsys profile`) with GPU-metrics sampling, runs an NVML poller alongside for
  power/clock/temp/VRAM, captures the workload's stdout (for block traces), and then
  calls `extract.py`.
- **`extract.py`** turns the nsys SQLite export into one tidy `runs/<name>.json`: the
  launch timeline, the resource lanes bucketed to 100 µs, NVML samples, the block→SM
  trace, and NVTX phase ranges.
- **`server.py`** is a dependency-free stdlib HTTP server. It lists `runs/*.json` (plus
  the built-in `dummy_prefill`) and serves each as JSON to the frontend.
- **`web/`** loads a run once and animates it locally (60 fps, time slider, speed
  0.05×–4×). No per-frame requests, so it survives a dropped SSH tunnel.

`dummy_run.py` is the **reference implementation of the run contract** — it documents the
exact JSON shape and generates the always-available `dummy_prefill` demo.

---

## What measures what

Each element of the replay comes from a specific measurement source, at a specific time
resolution. **The guiding principle: hardware counters count *events* (flows), so they
can report throughput and residency, but never *state* (space held).**

| GUI element | source | resolution | works on opaque kernels (PyTorch)? |
|---|---|---|---|
| launch timeline, stream queues, running box, hotspots, **memcpy H2D/D2H** | **CUPTI activity** (via nsys) | exact per launch | ✅ |
| rate lanes: SM active, warp occupancy, warp issue, tensor pipe, **L2**, DRAM BW, PCIe | **PM sampling** (`nsys --gpu-metrics`) | ~100 µs, device-wide | ✅ (unattributed) |
| power, SM/mem clock, temp, throttle, VRAM used | **NVML** poller | ~100 ms | ✅ |
| **per-SM block placement** (SM tiles, blocks/SM, occupancy fill) | **block→SM trace** (`%smid`+`%globaltimer`) | ~ns, per block | ⚠️ own/compiled kernels only |
| NVTX phase strip (prefill / decode / …) | **NVTX ranges** in the workload | exact | ✅ (if the workload annotates) |
| profiling-cost box (baseline vs profiled) | two timed runs (self-timed region) | — | ✅ |

### The rate lanes (device-wide, % of peak)

Sampled by nsys from the GPU's hardware performance monitor. Each is a **spatial average
across all units** (all 170 SMs, all L2 slices) and a **% of that unit's peak sustained
throughput**. Lane set on the RTX 5090:

| lane | counter | what 100% means |
|---|---|---|
| SM active | `sm__cycles_active` | every SM had ≥1 warp resident every cycle |
| Warp occupancy | `tpc__warps_active_shader_cs` | all 48 warp slots/SM filled |
| Warp issue | `sm__inst_executed_realtime` | 4 instructions/cycle/SM issued (1/SMSP) |
| Tensor pipe | `sm__pipe_tensor_cycles_active` | tensor pipe busy every cycle |
| **L2 BW** | `lts__t_sector_throughput` | L2 sector throughput at peak |
| DRAM BW | `dram__read/write_throughput` | GDDR7 bus at peak (~1.8 TB/s theoretical) |
| PCIe | `pcie__*_throughput` | PCIe 5.0 x16 link at peak |

> **L2 is not in NVIDIA's stock GeForce metric set.** We ship a custom set
> `gb20x-l2.config` (stock GB20x + an LTS row) that `record.py` loads automatically.
> The sampled L2 value reads ~10–15% below NCU's `lts__throughput` rollup (it's the
> sector component only). Pinned to nsys 2024.6.

### The block→SM trace (per-SM data)

The only source of **per-SM** information — sampled counters are device-wide averages and
cannot distinguish "all SMs half-full" from "half the SMs full, half idle". The trace
records, for each block, which physical SM it ran on and its entry/exit time
(`%smid` + `%globaltimer`), giving the SM-tile animation, blocks/SM, and true occupancy.

It requires code *inside* the kernel, so it works only for kernels you **compile**
(the `../profiler/` probe registry via `probe trace`).

> ### ⚠️ What the SM-tile animation is actually showing
>
> The per-SM tiles have **two** data sources, and which one is in play depends on the run:
>
> | run type | tiles are… | source |
> |---|---|---|
> | compiled kernels with a trace (`probe trace …`) | **real** — actual blocks per SM, over time | `%smid` block trace (exact) |
> | opaque kernels (PyTorch / any un-traced run) | **inferred** — every tile shows the *same* fill = device-average occupancy | the `Warp occupancy` lane |
>
> So for a PyTorch run (e.g. `llama3_prefill`, `llama3_sm50`) the tiles are **not** a real
> per-SM measurement — they are the device average spread uniformly across all 170 tiles,
> because no per-SM data exists for opaque kernels here. That average is identical whether
> the truth is "all SMs half-full" or "half the SMs full, half idle", so the animation can
> *look* like all SMs are used even when (e.g. under MPS 50%) only half really are. The GUI
> labels this in the tile tooltip; treat un-traced tiles as "device average, not per-SM".

### Why we can't get per-SM data for PyTorch kernels — NVBit

Per-SM placement of **opaque** kernels (cuBLAS/cuDNN/attention — no source) would need
**NVBit**, NVIDIA's binary-instrumentation framework, which injects the same `%smid` read
into precompiled SASS. We tested it (v1.7.5 and v1.8) and it **does not work in any
Python process on this machine.**

**What NVBit would give us** (if it worked here): a real block→SM trace of the *actual*
torch kernels — the true per-SM tile animation (which SMs a real GEMM used, wave fill/drain,
underfill shown as dark tiles, two-tenant coloring under colocation), plus, via its
memory-trace tools, a kernel's real working-set **footprint** — the one cache-capacity
signal that hardware counters structurally cannot report.

**What limits us — tested, minimally reproduced:**
- NVBit instruments our **compiled** binaries on this GPU *exactly* (validated: C
  executables, C programs that `dlopen` a CUDA `.so`, both `LD_PRELOAD` and
  `CUDA_INJECTION64_PATH`, sm_120 / driver 580 / CUDA 12.8).
- NVBit produces **zero events in a Python process** — reproduced with the *same* CUDA
  `.so` loaded from a C program (works) vs. from `python3 -c "ctypes.CDLL(...).run()"`
  (nothing). Not torch-specific; any Python host fails.
- Ruled out by direct test: NVBit version (1.7.5 ≡ 1.8), CUDA toolkit mismatch (all
  12.8), the CUPTI single-client slot (a full `libcupti` stub didn't help), lazy module
  loading, cudart/libcuda identity, conda-vs-system Python, symbol scope, process re-exec.
- Also mutually exclusive with `nsys` (same counter slot), so even if it worked, a per-SM
  trace could never share a recording with the rate lanes — it would be a separate pass.

**Suspected cause (unconfirmed):** a Python-process × driver-580 interaction of unknown
mechanism. A *simple* "driver too new" ceiling is **ruled out** — compiled binaries work
on driver 580 — so it must be something specific to how a Python process sets up CUDA
(late/lazy loading order, signal handlers vs. NVBit's channel thread, …) that we could not
isolate. Confirming it would require testing NVBit-in-Python on an older driver (not
practical on this shared box) or filing the minimal repro with NVlabs.

**The one route that *does* work:** NVBit is gated by the **host process, not the kernel**
— it instruments opaque library kernels fine *from a compiled program*. So the real
cuBLAS/attention kernels a transformer uses *are* traceable if driven from a **C++ harness**
fed the layer shapes, just not while Python is the host. See `../profiler/` for the probe
harness this would extend.

---

## Running your own workload

### 1. Any command (no code changes)

```bash
python3 record.py myrun -- <any command that uses the GPU>
```

You immediately get the launch timeline, all rate lanes, NVML, hotspots, and memcpys.
Kernel names are auto-demangled and the top 8 by time get distinct colors (the rest fold
to "other"). SM tiles show the device-average fallback (no per-SM trace).

### 2. Add capture range + phases (recommended for long apps)

For apps with a heavy setup phase (model load), wrap only the region you care about in
`cudaProfilerStart/Stop` and record with `--capture` — see
[`workloads/llama_prefill.py`](workloads/llama_prefill.py) for the pattern:

- `torch.cuda.profiler.start()/stop()` bracket the captured region;
- `torch.cuda.nvtx.range("prefill")` etc. become the **phase strip**;
- printing `CAPTURE_START <epoch>` anchors NVML alignment precisely;
- printing `REGION_MS <ms>` enables the **profiling-cost** box (baseline vs profiled).

```bash
python3 record.py llama3 --capture -- python3 workloads/llama_prefill.py
```

`record.py` runs the workload **twice** (once un-profiled for the baseline, once under
nsys) to measure profiling overhead. Pass `--no-baseline` to skip the second run.

### 3. Per-SM tiles for your own kernels

Only kernels compiled with the `%smid` trace produce per-SM data. The `../profiler/`
`probe` binary already has this (`probe trace <kernel> [antagonist]`); record it like any
command:

```bash
python3 record.py probe_fma --no-baseline -- ../profiler/build/probe trace fma
```

### 4. Limiting a workload to a fraction of the SMs (MPS)

To record a workload confined to N% of the SMs (SM partitioning), run it under CUDA MPS:

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-cuda-mps-control -d                      # start MPS (uses /tmp/nvidia-mps)
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 \
  python3 record.py llama3_sm50 --capture --no-baseline -- python3 workloads/llama_prefill.py
echo quit | nvidia-cuda-mps-control             # stop MPS
```

The rate lanes show the reduced device-average SM usage; per-SM tiles show the true
partition **only for traced (compiled) kernels** — an opaque workload still renders the
uniform-average fallback.

---

## Launching the web server

### Local

```bash
bash serve.sh            # http://localhost:8000  (idempotent: restarts a prior instance)
bash serve.sh 8080       # custom port
```

`serve.sh` runs `python3 server.py`, logs to `/tmp/replay_server.log`, and verifies the
API responds. Stop it with `pkill -f 'python3 server.py'`.

### From a Mac, GPU box is remote

One command on the Mac starts the server remotely, opens an SSH tunnel, and launches the
browser. Copy [`mac/gpu-replay.sh`](mac/gpu-replay.sh) to the Mac, edit the `REMOTE=` line
to your SSH target, then:

```bash
~/bin/gpu-replay.sh
```

Or do it by hand:

```bash
ssh -L 8000:localhost:8000 <user>@<gpu-host>   # then run serve.sh on the host
# open http://localhost:8000 on the Mac
```

---

## Repository layout

| path | what it is |
|---|---|
| `record.py` | record a workload (nsys + NVML + baseline pass) → run JSON |
| `extract.py` | nsys SQLite → `runs/<name>.json` (the run contract) |
| `dummy_run.py` | run-contract reference + the always-served `dummy_prefill` demo |
| `server.py` / `serve.sh` | stdlib HTTP backend + launcher |
| `gb20x-l2.config` | custom nsys metric set adding the L2 lane on GeForce |
| `web/` | frontend: `index.html`, `app.js`, `style.css` |
| `workloads/` | recordable workload scripts (e.g. `llama_prefill.py`) |
| `mac/gpu-replay.sh` | Mac-side start-server + tunnel + open-browser |
| `recordings/` | raw nsys reports + NVML csv per run (generated) |
| `runs/` | extracted run JSON, auto-served (generated) |

---

## What is *not* measurable (stated honestly, labeled in the UI)

- **Cache fill level** (how much L1/L2 a workload holds) — counters report throughput and
  hit rate, never bytes-resident. No per-line ownership hardware exists.
- **Per-SM counters** — all sampled lanes are spatial averages; per-SM truth comes only
  from the block trace.
- **Per-kernel attribution under concurrency** — overlapping kernels blend into one
  device-wide lane value; attribution is inference, not measurement.
- **Per-SM data for opaque kernels in Python** (cuBLAS/cuDNN via PyTorch) — no source, and
  NVBit does not attach to Python processes here. Compiled kernels are fine.
- **True GPC membership** of each SM (per-die yield harvesting, unpublished) — the GPC
  boxes are schematic; the 170 SM count and per-SM placement are real.
- **Sub-~100 µs lane detail** — the PM-sampling floor; kernels shorter than a bucket smear
  in the lanes (but still appear exactly on the timeline and SM tiles).

## Requirements & installation

Developed and tested on this exact stack (RTX 5090 / GB202, `sm_120`, 170 SMs):

| component | version | why |
|---|---|---|
| OS | Ubuntu 24.04.2 LTS | — |
| NVIDIA driver | 580.126.09 | must support GPU-metrics sampling |
| CUDA Toolkit | 12.8 (`nvcc` V12.8.93) | build the `../profiler` probe (for traces) |
| Nsight Systems (`nsys`) | 2024.6.2 | recording; `gb20x-l2.config` is pinned to this version |
| sqlite3 | 3.45.3 | nsys export → run JSON |
| Python | 3.12.9 | stdlib only for the server & extract.py |
| PyTorch (workloads only) | 2.8.0 (cu128) | needed by `workloads/llama_prefill.py` |
| transformers (workloads only) | 4.45.0 | Llama-3 loading |

Install the pieces:

```bash
# nsys + nvcc come with the CUDA Toolkit 12.8 (already on the machine at /usr/local/cuda-12.8)
sudo apt-get install -y sqlite3                       # if not present
# the server, record.py and extract.py use only the Python standard library — nothing to pip-install

# only if you run the bundled ML workloads:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.45.0
```

- **GPU performance-counter access must be enabled** for `nsys` GPU metrics. If sampling
  returns nothing, enable it (personal workstation):
  `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && sudo reboot`.
- The counter hardware is **single-client**: do not run `nsys` and `ncu` concurrently.
- MPS SM-limiting (optional) uses `nvidia-cuda-mps-control`, shipped with the driver.
- A different GPU generation needs a matching nsys metric set (replace `gb20x-l2.config`).
