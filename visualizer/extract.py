#!/usr/bin/env python3
"""Extract an nsys recording into the replay-GUI run JSON (see dummy_run.py).

  python3 extract.py recordings/<name>/report.nsys-rep --name <name> \
      [--nvml recordings/<name>/nvml.csv] [--meta recordings/<name>/meta.json]

Contract additions vs the dummy: launches carry their own `grid`;
`real: true`; lane names are what the metric set provides — the stock GB20x set
plus an L2 (LTS sector-throughput) row from our custom gb20x-l2.config.
"""
import argparse, collections, datetime, json, math, os, sqlite3, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SM_COUNT = 170            # GB202 / RTX 5090 (cudaDeviceProp.multiProcessorCount)
LANE_RES_US = 100
NVML_RES_US = 100_000     # nvidia-smi -lms 100

# lane -> GB20x metric names summed into it. "L2 BW" comes from our CUSTOM metric set
# (gb20x-l2.config): lts__t_sector_throughput — the sector component only, so it reads
# ~10-15% below NCU's lts__throughput rollup (max over LTS sub-counters).
LANES = [
    ("SM active",      ["SMs Active [Throughput %]"]),
    ("Warp occupancy", ["Compute Warps in Flight [Throughput %]"]),
    ("Warp issue",     ["SM Issue [Throughput %]"]),
    ("Tensor pipe",    ["Tensor Active [Throughput %]"]),
    ("L2 BW",          ["L2 Bandwidth [Throughput %]"]),
    ("DRAM BW",        ["DRAM Read Bandwidth [Throughput %]", "DRAM Write Bandwidth [Throughput %]"]),
    ("PCIe",           ["PCIe RX Throughput [Throughput %]", "PCIe TX Throughput [Throughput %]"]),
]
THROTTLE_BITS = [(0x4, "sw power cap"), (0x40, "sw thermal"), (0x8, "hw slowdown"), (0x80, "hw thermal")]
# probe workload name -> CUDA kernel function name (for matching TRACEBLK to CUPTI launches)
PROBE_KMAP = {"sleep": "k_sleep", "dram": "k_copy", "l2": "k_copy",
              "l1": "k_copy_tb", "fma": "k_fma32", "fp64": "k_fma64"}


def ensure_sqlite(rep):
    db = rep.rsplit(".", 1)[0] + ".sqlite"
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
                        "-o", db, rep], check=True, capture_output=True)
    return db


def clean_name(n):
    """ATen/cutlass kernel names are huge templates; keep the base identifier.
    'void at::native::vectorized_elementwise_kernel<4, ...>' -> 'vectorized_elementwise_kernel'.
    Variants merge under one display name (desirable: one color per kernel family)."""
    base = n.split("<")[0].split("(")[0].strip()
    base = base.split("::")[-1].replace("void ", "").strip() or n
    # cuBLAS ships anonymous kernels literally named Kernel/Kernel2 on Blackwell
    return f"cublas_gemm_{base}" if base.startswith("Kernel") else base


# copyKind id -> short label (from ENUM_CUDA_MEMCPY_OPER); H2D/D2H/D2D are the common ones
MEMCPY_KIND = {1: "Memcpy HtoD", 2: "Memcpy DtoH", 8: "Memcpy DtoD",
               10: "Memcpy PtoP", 11: "Memcpy HtoD", 12: "Memcpy DtoH"}


def load_launches(con):
    rows = con.execute("""
        SELECT k.start, k.end, k.streamId,
               k.gridX * k.gridY * k.gridZ, k.blockX * k.blockY * k.blockZ,
               s.value, r.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
        ORDER BY k.start""").fetchall()
    if not rows:
        sys.exit("no kernel launches in this recording")
    # memcpys as synthetic zero-grid "kernels": same host->queue->timeline path, no SM tiles
    try:
        for st, en, sid, ck, cid, sub in con.execute("""
                SELECT m.start, m.end, m.streamId, m.copyKind, m.correlationId, r.start
                FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON m.correlationId = r.correlationId"""):
            rows.append((st, en, sid, 0, 0, MEMCPY_KIND.get(ck, "Memcpy"), sub))
    except sqlite3.OperationalError:
        pass
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--name", required=True)
    ap.add_argument("--nvml"); ap.add_argument("--meta"); ap.add_argument("--stdout")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    con = sqlite3.connect(ensure_sqlite(args.report))
    rows = load_launches(con)

    # ---- time base: t=0 a little before the first submit/start (ns -> µs) ----
    t0_ns = min(r[6] if r[6] else r[0] for r in rows) - 2_000_000
    tmax_us = int((max(r[1] for r in rows) - t0_ns) / 1000) + 2000
    us = lambda ns: int((ns - t0_ns) / 1000)

    rows = [(st, en, sid, grid, blk, clean_name(name), sub)
            for st, en, sid, grid, blk, name, sub in rows]

    # ---- kernel registry: color slots 1..8 by total duration, rest slot 0 ----
    total = collections.Counter()
    for st, en, *_r, name, sub in [(r[0], r[1], r[2], r[5], r[6]) for r in rows]:
        total[name] += en - st
    order = [n for n, _ in total.most_common()]
    slot = {n: (i + 1 if i < 8 else 0) for i, n in enumerate(order)}

    stream_map = {s: i for i, s in enumerate(sorted({r[2] for r in rows}))}
    kernels, launches = {}, []
    for st, en, sid, grid, blk, name, sub in rows:
        k = kernels.setdefault(name, {"slot": slot[name], "grid": 0, "cap": 0, "block": blk})
        k["grid"] = max(k["grid"], grid)
        launches.append({"k": name, "stream": stream_map[sid], "grid": grid,
                         "submit": us(sub) if sub else us(st) - 1,
                         "start": us(st), "end": us(en)})

    # ---- lanes: pivot GPU_METRICS onto a uniform 100 µs grid.
    # A lane whose metrics are absent from this recording is DROPPED (not zero-filled)
    # so e.g. recordings made with the stock metric set simply have no L2 lane. ----
    mid = dict(con.execute("SELECT metricName, metricId FROM TARGET_INFO_GPU_METRICS"))
    lanes_def = [(n, ms) for n, ms in LANES if any(m in mid for m in ms)]
    for n, ms in LANES:
        if (n, ms) not in lanes_def:
            print(f"note: lane '{n}' absent in this recording — dropped", file=sys.stderr)
    NL_run = len(lanes_def)
    need = {}
    for li, (_lname, metrics) in enumerate(lanes_def):
        for m in metrics:
            if m in mid: need[mid[m]] = li
    nb = math.ceil(tmax_us / LANE_RES_US)
    acc = [[0.0] * NL_run for _ in range(nb)]; cnt = [[0] * NL_run for _ in range(nb)]
    per_ts = collections.defaultdict(lambda: [0.0] * NL_run)
    q = f"SELECT timestamp, metricId, value FROM GPU_METRICS WHERE metricId IN ({','.join(map(str, need))})"
    for ts, m, v in con.execute(q):
        per_ts[ts][need[m]] += v
    for ts, vals in per_ts.items():
        t = us(ts)
        if 0 <= t < tmax_us:
            b = t // LANE_RES_US
            for i in range(NL_run): acc[b][i] += vals[i]; cnt[b][i] += 1
    lanes, last = [], [0.0] * NL_run
    for b in range(nb):
        row = [min(100.0, round(acc[b][i] / cnt[b][i], 1)) if cnt[b][i] else last[i] for i in range(NL_run)]
        lanes.append(row); last = row

    # ---- NVML: map wall clock -> session time.
    # nsys spends seconds writing the report AFTER the workload, so end-alignment is
    # wrong; instead anchor the first clock-boosted NVML sample to the first kernel
    # start (the GPU boosts when work arrives), with wall_start as coarse fallback. ----
    nvml = []
    if args.nvml and os.path.exists(args.nvml) and args.meta:
        wall_start = json.load(open(args.meta))["wall_start"]
        samples = []
        for line in open(args.nvml):
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6: continue
            try:
                w = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                mask = int(p[6], 16) if len(p) > 6 and p[6].startswith("0x") else 0
                thr = next((lbl for bit, lbl in THROTTLE_BITS if mask & bit), None)
                samples.append((w, dict(power=float(p[1]), sm_clk=int(p[2]), mem_clk=int(p[3]),
                                        temp=float(p[4]), vram_gb=round(float(p[5]) / 1024, 1),
                                        throttle=thr)))
            except (ValueError, IndexError):
                continue
        prov = lambda w: int((w - wall_start - 0.4) * 1e6)   # ~0.4 s nsys startup
        # best anchor: workload printed CAPTURE_START <epoch> right before its first
        # launch (capture-range runs). Fallback: first clock-boost edge = first kernel.
        cap_wall = None
        if args.stdout and os.path.exists(args.stdout):
            for line in open(args.stdout):
                if line.startswith("CAPTURE_START"):
                    cap_wall = float(line.split()[1]); break
        first_us = min(l["start"] for l in launches)
        if cap_wall:
            shift = first_us - prov(cap_wall)
        else:
            busy = [w for w, d in samples if d["sm_clk"] >= 1500]
            shift = (first_us - prov(busy[0])) if busy else 0
        for w, d in samples:
            d["_t"] = prov(w) + shift
        nbn = math.ceil(tmax_us / NVML_RES_US)
        nvml = [None] * nbn
        for w, d in samples:
            t = d.pop("_t")
            if 0 <= t < tmax_us: nvml[t // NVML_RES_US] = d
        fill = next((d for d in nvml if d), dict(power=0, sm_clk=0, mem_clk=0, temp=0, vram_gb=0, throttle=None))
        for i in range(nbn):
            if nvml[i] is None: nvml[i] = fill
            else: fill = nvml[i]
    if not nvml:
        nvml = [dict(power=0, sm_clk=0, mem_clk=0, temp=0, vram_gb=0, throttle=None)]

    # ---- block->SM trace (TRACEBLK lines from the workload's stdout).
    # globaltimer has its own epoch; align per traced launch: the traced launch is the
    # LAST CUPTI launch of that kernel (trace runs after an untraced warm-up), and its
    # earliest block-start coincides with the launch's CUPTI start. ----
    blocks = []
    if args.stdout and os.path.exists(args.stdout):
        ev = collections.defaultdict(list)
        for line in open(args.stdout):
            if not line.startswith("TRACEBLK"): continue
            d = dict(kv.split("=") for kv in line.split()[1:])
            ev[d["kernel"]].append((int(d["smid"]), int(d["t0"]), int(d["t1"])))
        for wname, evs in ev.items():
            kn = PROBE_KMAP.get(wname, wname)
            idxs = [i for i, l in enumerate(launches) if l["k"] == kn]
            if not idxs:
                print(f"warning: TRACEBLK kernel {wname} ({kn}) not in CUPTI launches", file=sys.stderr)
                continue
            li = idxs[-1]; base = launches[li]["start"]
            mint0 = min(t0 for _s, t0, _t1 in evs)
            for smid, t0, t1 in evs:
                blocks.append(dict(k=kn, li=li, smid=smid,
                                   t0=base + (t0 - mint0) // 1000,
                                   t1=base + (t1 - mint0) // 1000))

    # ---- NVTX phase ranges (e.g. prefill / decode_N from the workload script) ----
    phases = []
    try:
        for st, en, txt in con.execute(
                "SELECT start, end, text FROM NVTX_EVENTS "
                "WHERE end IS NOT NULL AND text IS NOT NULL"):
            s, e = us(st), us(en)
            if e > 0 and s < tmax_us and txt != "warmup":
                phases.append(dict(name=txt, start=max(0, s), end=min(tmax_us, e)))
    except sqlite3.OperationalError:
        pass
    phases.sort(key=lambda p: p["start"])

    run = dict(name=args.name, real=True, sm_count=SM_COUNT, tmax_us=tmax_us,
               kernels=kernels, launches=launches,
               lanes=dict(res_us=LANE_RES_US, names=[l[0] for l in lanes_def], data=lanes),
               nvml=dict(res_us=NVML_RES_US, data=nvml))
    if blocks:
        run["blocks"] = blocks
    if phases:
        run["phases"] = phases
    if args.meta and os.path.exists(args.meta):
        cost = json.load(open(args.meta)).get("cost")
        if cost:
            run["cost"] = cost
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, args.name + ".json")
    json.dump(run, open(out, "w"))
    print(f"extracted -> {out}  ({len(launches)} launches, {len(kernels)} kernels, "
          f"{tmax_us/1000:.0f} ms, {nb} lane buckets, {len(nvml)} nvml samples)")


if __name__ == "__main__":
    main()
