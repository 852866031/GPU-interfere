#!/usr/bin/env python3
"""Parse Section 4.1 experiment logs -> CSVs + figures.
Run:  python3 scripts/parse_and_plot.py
"""
import re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

CFG = re.compile(r"@CONFIG (.+)")
def kv(line):
    return dict(tok.split("=", 1) for tok in line.strip().split() if "=" in tok)

def parse(logname):
    """Yield (config_dict, metric_name, value) tuples in order."""
    path = os.path.join(RES, logname)
    cur = None
    with open(path) as f:
        for line in f:
            m = CFG.search(line)
            if m:
                cur = kv(m.group(1)); continue
            for pat, name in [(r"Avg alone time is ([\d.]+)", "alone_ms"),
                              (r"Avg sequential time is ([\d.]+)", "seq_ms"),
                              (r"Avg colocated time is ([\d.]+)", "coloc_ms"),
                              (r"achieved bandwidth is ([\d.]+)", "bw")]:
                mm = re.search(pat, line)
                if mm and cur is not None:
                    yield dict(cur), name, float(mm.group(1))

# ---------- 4.1.1 thread-block scheduler ----------
tb = {}  # (blocks_per_sm) -> {alone,seq,coloc}
for cfg, name, val in parse("tb_scheduler.log"):
    b = int(cfg["blocks_per_sm"])
    tb.setdefault(b, {})[name] = val
with open(os.path.join(RES, "tb_scheduler.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["blocks_per_sm", "alone_ms", "seq_ms", "coloc_ms"])
    for b in sorted(tb):
        w.writerow([b, tb[b].get("alone_ms"), tb[b].get("seq_ms"), tb[b].get("coloc_ms")])

fig, ax = plt.subplots(figsize=(6.2, 4))
groups = sorted(tb); x = range(len(groups)); w = 0.26
ax.bar([i - w for i in x], [tb[b]["alone_ms"] for b in groups], w, label="alone (1 kernel)", color="#4C78A8")
ax.bar([i     for i in x], [tb[b]["seq_ms"]   for b in groups], w, label="sequential (2 kernels)", color="#B0B0B0")
ax.bar([i + w for i in x], [tb[b]["coloc_ms"] for b in groups], w, label="colocated (2 kernels)", color="#E45756")
ax.set_xticks(list(x)); ax.set_xticklabels([f"{b} block/SM" if b == 1 else f"{b} blocks/SM" for b in groups])
ax.set_ylabel("latency (ms)"); ax.set_title("4.1.1  Thread-block scheduler")
for i, b in enumerate(groups):
    ax.text(i + w, tb[b]["coloc_ms"] + 3, f'{tb[b]["coloc_ms"]:.0f}', ha="center", fontsize=8)
ax.legend(fontsize=8); ax.margins(y=0.15); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_411_tb_scheduler.png"), dpi=150); plt.close(fig)

# ---------- 4.1.2 L2 cache ----------
l2 = {}  # size_mb -> {alone,coloc}
for cfg, name, val in parse("l2_cache.log"):
    s = int(cfg["size_mb"]); l2.setdefault(s, {})[name] = val
sizes = sorted(l2)
with open(os.path.join(RES, "l2_cache.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["size_mb", "alone_ms", "coloc_ms", "coloc_over_alone"])
    for s in sizes:
        a, c = l2[s]["alone_ms"], l2[s]["coloc_ms"]
        w.writerow([s, a, c, round(c / a, 3)])

L2_MB = 96
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(sizes, [l2[s]["alone_ms"] for s in sizes], "-o", color="#4C78A8", label="alone (1 kernel)")
ax.plot(sizes, [2*l2[s]["alone_ms"] for s in sizes], "--", color="#B0B0B0", label="2x alone (ideal serial)")
ax.plot(sizes, [l2[s]["coloc_ms"] for s in sizes], "-s", color="#E45756", label="colocated (2 kernels, separate SMs)")
# combined-footprint = L2 threshold: 4*size (in+out per kernel, 2 kernels) = 96 MB -> size = 24 MB
ax.axvline(L2_MB/4, color="#666", ls=":", lw=1)
ax.text(L2_MB/4+1, ax.get_ylim()[1]*0.6, "combined footprint\n= 96 MB L2", fontsize=8, color="#666")
# second overlap: colocated converges back down to ~2x alone (both kernels now fully
# DRAM-bound -> colocation = pure serialization again, no L2 left to lose)
def excess(s): return l2[s]["coloc_ms"] / (2*l2[s]["alone_ms"]) - 1
cx = cy = None; peaked = False
for i in range(1, len(sizes)):
    if excess(sizes[i-1]) > 0.30: peaked = True
    if peaked and excess(sizes[i-1]) > 0.10 >= excess(sizes[i]):
        e0, e1 = excess(sizes[i-1]), excess(sizes[i])
        t = (e0 - 0.10) / (e0 - e1)
        cx = sizes[i-1] + t*(sizes[i]-sizes[i-1])
        cy = l2[sizes[i-1]]["coloc_ms"] + t*(l2[sizes[i]]["coloc_ms"] - l2[sizes[i-1]]["coloc_ms"])
        break
if cy is not None:
    ax.axvline(cx, color="#555", ls="-.", lw=1.2)
    ax.text(cx+1.5, 20, f"~{cx:.0f} MB", fontsize=8, color="#555")
ax.set_xlabel("copy size per kernel (MB)"); ax.set_ylabel("latency (ms)")
ax.set_title("4.1.2  L2 cache interference"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_412_l2_cache.png"), dpi=150); plt.close(fig)

# ---------- 4.1.3 memory bandwidth ----------
sat = {}
for cfg, name, val in parse("mem_bw_saturation.log"):
    if name == "bw":
        sat[int(cfg["num_tb"])] = val
tbs = sorted(sat)
with open(os.path.join(RES, "mem_bw_saturation.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["num_tb", "bw_GBs"])
    for t in tbs: w.writerow([t, sat[t]])

mps = {"alone_half": None, "coloc": []}           # bandwidth (GB/s)
mps_lat = {"alone_half": None, "coloc": []}        # latency (ms)
for cfg, name, val in parse("mem_bw_mps.log"):
    tgt = mps if name == "bw" else (mps_lat if name == "alone_ms" else None)
    if tgt is None:
        continue
    if cfg.get("case") == "alone_half":
        tgt["alone_half"] = val
    elif cfg.get("case") == "colocated_half":
        tgt["coloc"].append(val)
with open(os.path.join(RES, "mem_bw_mps.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "bw_GBs", "latency_ms"])
    w.writerow(["alone_half", mps["alone_half"], mps_lat["alone_half"]])
    for i, (b, l) in enumerate(zip(mps["coloc"], mps_lat["coloc"]), 1):
        w.writerow([f"colocated_inst{i}", b, l])
    w.writerow(["colocated_aggregate", sum(mps["coloc"]), max(mps_lat["coloc"])])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
a1.plot(tbs, [sat[t] for t in tbs], "-o", color="#4C78A8")
a1.axhline(max(sat.values()), color="#666", ls=":", lw=1)
a1.text(tbs[1], max(sat.values())*0.94, f"saturates ~{max(sat.values()):.0f} GB/s", fontsize=8, color="#666")
a1.set_xlabel("thread blocks (single kernel, full GPU)"); a1.set_ylabel("achieved bandwidth (GB/s)")
a1.set_title("4.1.3a  DRAM bandwidth saturation")

peak = max(sat.values()); alone_h = mps["alone_half"]
k1c, k2c = mps["coloc"][0], mps["coloc"][1]      # measured colocated per kernel
agg = k1c + k2c; demand = 2 * alone_h
# Two bars: baseline (1 kernel on its half) vs colocated total (kernel A + B stacked).
x = [0, 1]; w = 0.55
a2.bar(x[0], alone_h, w, color="#4C78A8", edgecolor="white")
a2.bar(x[1], k1c, w, color="#E45756", edgecolor="white", label="kernel A")
a2.bar(x[1], k2c, w, bottom=k1c, color="#F2A6A0", edgecolor="white", label="kernel B")
# value labels
a2.text(0, alone_h + 25, f"{alone_h:.0f}", ha="center", fontsize=9, weight="bold")
a2.text(1, k1c / 2, f"A\n{k1c:.0f}", ha="center", va="center", fontsize=8)
a2.text(1, k1c + k2c / 2, f"B\n{k2c:.0f}", ha="center", va="center", fontsize=8)
a2.text(1, agg + 25, f"A+B = {agg:.0f}", ha="center", fontsize=9, weight="bold", color="#8B0000")
# reference lines: demand-if-independent (impossible) and the DRAM ceiling
a2.axhline(demand, color="#4C78A8", ls="--", lw=1.2)
a2.text(1.46, demand + 20, f"if independent: 2 x {alone_h:.0f} = {demand:.0f}  (demanded)",
        fontsize=8, color="#4C78A8", ha="right")
a2.axhline(peak, color="#333", ls=":", lw=1.3)
a2.text(1.46, peak + 20, f"DRAM ceiling ~{peak:.0f} GB/s  (max possible)",
        fontsize=8, color="#333", ha="right")
# per-kernel loss arrow from the baseline bar to kernel A's segment
a2.annotate("", xy=(1, k1c), xytext=(0, alone_h),
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.4, ls="--"))
a2.text(0.5, (alone_h + k1c) / 2 + 25, f"-{100 - agg/2/alone_h*100:.0f}% per kernel",
        ha="center", fontsize=8.5, color="#555")
a2.set_ylim(0, demand * 1.16); a2.set_ylabel("achieved bandwidth (GB/s)")
a2.set_xticks(x); a2.set_xticklabels(["1 kernel alone\n(on its 50% SMs)",
                                      "2 kernels colocated\n(A + B, each on 50% SMs)"])
a2.set_title("4.1.3b  Two kernels, each on its own 50% of SMs")
a2.legend(fontsize=7.5, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_413_mem_bw.png"), dpi=150); plt.close(fig)

# ---------- 4.1.3c latency view of the same Part-B interference ----------
al = mps_lat["alone_half"]; l1, l2v = mps_lat["coloc"][0], mps_lat["coloc"][1]
figL, axL = plt.subplots(figsize=(5.2, 4.2))
labels = ["1 kernel alone\n(on its 50% SMs)", "kernel A\n(colocated)", "kernel B\n(colocated)"]
vals = [al, l1, l2v]; cols = ["#4C78A8", "#E45756", "#E45756"]
axL.bar(range(3), vals, width=0.62, color=cols, edgecolor="white")
for i, v in enumerate(vals):
    axL.text(i, v + 8, f"{v:.0f} ms", ha="center", fontsize=9, weight="bold")
axL.axhline(al, color="#4C78A8", ls=":", lw=1.2)
axL.text(2.45, al - 45, "alone baseline", fontsize=8, color="#4C78A8", ha="right")
axL.annotate("", xy=(1, l1), xytext=(0, al),
             arrowprops=dict(arrowstyle="->", color="#888", lw=1.4, ls="--"))
axL.text(0.5, (al + l1) / 2 + 20, f"+{(l1/al-1)*100:.0f}% / +{(l2v/al-1)*100:.0f}%\nlonger",
         ha="center", fontsize=8.5, color="#555")
axL.set_ylim(0, max(vals) * 1.2); axL.set_ylabel("kernel latency (ms)  — lower is better")
axL.set_xticks(range(3)); axL.set_xticklabels(labels)
axL.set_title("4.1.3c  Same interference, seen as latency")
figL.tight_layout(); figL.savefig(os.path.join(FIG, "fig_413c_latency.png"), dpi=150); plt.close(figL)

# ---------- console summary ----------
print("=== 4.1.1 thread-block scheduler ===")
for b in sorted(tb):
    t = tb[b]; print(f"  {b} block(s)/SM: alone={t['alone_ms']:.1f}  seq={t['seq_ms']:.1f}  "
                     f"coloc={t['coloc_ms']:.1f}  coloc/alone={t['coloc_ms']/t['alone_ms']:.2f}")
print("=== 4.1.2 L2 cache (coloc/alone ratio) ===")
for s in sizes:
    print(f"  {s:3d} MB: alone={l2[s]['alone_ms']:7.1f}  coloc={l2[s]['coloc_ms']:7.1f}  ratio={l2[s]['coloc_ms']/l2[s]['alone_ms']:.2f}")
print("=== 4.1.3 bandwidth ===")
print(f"  saturation peak ~{peak:.0f} GB/s")
print(f"  alone on 50% SMs = {alone_h:.0f} GB/s ; colocated aggregate = {agg:.0f} GB/s "
      f"(each ~{agg/2:.0f}, i.e. {agg/2/alone_h*100:.0f}% of alone)")
print("Figures ->", FIG)
