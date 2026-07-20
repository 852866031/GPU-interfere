#!/usr/bin/env python3
"""Parse Section 4.2 experiment logs -> CSVs + figures."""
import re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

CFG = re.compile(r"@CONFIG (.+)")
def kv(line): return dict(t.split("=", 1) for t in line.strip().split() if "=" in t)
def parse(logname):
    cur = None
    for line in open(os.path.join(RES, logname)):
        m = CFG.search(line)
        if m: cur = kv(m.group(1)); continue
        for pat, name in [(r"Avg alone time is ([\d.]+)", "alone"),
                          (r"Avg sequential time is ([\d.]+)", "seq"),
                          (r"Avg colocated time is ([\d.]+)", "coloc")]:
            mm = re.search(pat, line)
            if mm and cur is not None:
                yield dict(cur), name, float(mm.group(1))

# ---------- 4.2.1 L1 cache ----------
l1 = {}
for cfg, name, val in parse("l1_cache.log"):
    s = int(cfg["size_kb"]); l1.setdefault(s, {})[cfg["mode"]] = val
sizes = sorted(l1)
with open(os.path.join(RES, "l1_cache.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["size_kb", "alone_ms", "coloc_ms", "coloc_over_alone"])
    for s in sizes:
        a, c = l1[s]["alone"], l1[s]["colocated"]; w.writerow([s, a, c, round(c/a, 3)])

L1_KB = 128
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(sizes, [l1[s]["alone"] for s in sizes], "-o", color="#4C78A8", label="alone (1 kernel)")
ax.plot(sizes, [2*l1[s]["alone"] for s in sizes], "--", color="#B0B0B0", label="2x alone (ideal serial)")
ax.plot(sizes, [l1[s]["colocated"] for s in sizes], "-s", color="#E45756", label="colocated (2 kernels, same SM)")
ax.axvline(L1_KB/4, color="#666", ls=":", lw=1)
ax.text(L1_KB/4+0.7, ax.get_ylim()[1]*0.55, "combined footprint\n= 128 KB L1", fontsize=8, color="#666")
ax.set_xlabel("copy size per block (KB)"); ax.set_ylabel("latency (ms)")
ax.set_title("4.2.1  L1 cache interference"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_421_l1_cache.png"), dpi=150); plt.close(fig)

# ---------- 4.2.2 IPC / warp scheduler ----------
ipc = {}
for cfg, name, val in parse("ipc.log"):
    ipc[cfg["case"]] = val
copy_a, comp_a = ipc["copy_alone"], ipc["compute_alone"]
seq, coloc = ipc["sequential"], ipc["colocated"]
ideal = max(copy_a, comp_a)                 # perfect overlap
interf = (coloc - ideal) / (seq - ideal) * 100
with open(os.path.join(RES, "ipc.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "latency_ms"])
    for k in ["copy_alone", "compute_alone", "sequential", "colocated"]: w.writerow([k, ipc[k]])
    w.writerow(["interference_pct_toward_serial", round(interf, 1)])

fig, ax = plt.subplots(figsize=(6.2, 4.2))
labels = ["copy\nalone", "compute\nalone", "colocated\n(both)", "sequential\n(both)"]
vals = [copy_a, comp_a, coloc, seq]
cols = ["#4C78A8", "#72B7B2", "#E45756", "#B0B0B0"]
ax.bar(range(4), vals, width=0.62, color=cols, edgecolor="white")
for i, v in enumerate(vals): ax.text(i, v+6, f"{v:.0f}", ha="center", fontsize=9, weight="bold")
ax.axhline(ideal, color="#2A9D8F", ls=":", lw=1.3)
ax.text(2.62, ideal-22, "perfect overlap = max(alone)", fontsize=8, color="#2A9D8F", ha="center")
ax.axhline(seq, color="#888", ls="--", lw=1.1)
ax.text(1.0, seq+8, "full serialization = sequential", fontsize=8, color="#888", ha="center")
ax.annotate("", xy=(2, coloc), xytext=(2, ideal),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
ax.text(2.12, (coloc+ideal)/2, f"+{coloc-ideal:.0f} ms\n({interf:.0f}% toward\nserial)", fontsize=8, color="#555", va="center")
ax.set_ylim(0, seq*1.22)
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_ylabel("latency (ms)")
ax.set_title("4.2.2  Warp-scheduler (IPC) interference\ncopy (memory) + compute (FMA), same SM")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_422_ipc.png"), dpi=150); plt.close(fig)

# ---------- 4.2.3 pipelines ----------
pipe = {}
for cfg, name, val in parse("pipelines.log"):
    i = int(cfg["ilp"]); pipe.setdefault(i, {})[cfg["mode"]] = val
ilps = sorted(pipe)
with open(os.path.join(RES, "pipelines.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ilp", "alone_ms", "seq_ms", "coloc_ms", "coloc_over_alone"])
    for i in ilps:
        p = pipe[i]; w.writerow([i, p["alone"], p["sequential"], p["colocated"], round(p["colocated"]/p["alone"], 3)])

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = range(len(ilps)); w = 0.26
ax.bar([i-w for i in x], [pipe[i]["alone"] for i in ilps], w, label="alone (1 kernel)", color="#4C78A8")
ax.bar([i   for i in x], [pipe[i]["sequential"] for i in ilps], w, label="sequential (2 kernels)", color="#B0B0B0")
ax.bar([i+w for i in x], [pipe[i]["colocated"] for i in ilps], w, label="colocated (2 kernels)", color="#E45756")
for i in x:
    ax.text(i+w, pipe[ilps[i]]["colocated"]+4, f'{pipe[ilps[i]]["colocated"]/pipe[ilps[i]]["alone"]:.1f}x',
            ha="center", fontsize=8, color="#8B0000")
ax.set_xticks(list(x)); ax.set_xticklabels([f"ILP {i}" for i in ilps])
ax.set_ylabel("latency (ms)"); ax.set_title("4.2.3  FP64 pipeline interference (colocated = sequential)")
ax.legend(fontsize=8); ax.margins(y=0.12); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_423_pipelines.png"), dpi=150); plt.close(fig)

# ---------- console summary ----------
print("=== 4.2.1 L1 cache (coloc/alone) ===")
for s in sizes: print(f"  {s:3d} KB: alone={l1[s]['alone']:7.1f}  coloc={l1[s]['colocated']:7.1f}  ratio={l1[s]['colocated']/l1[s]['alone']:.2f}")
print("=== 4.2.2 IPC ===")
print(f"  copy={copy_a:.0f}  compute={comp_a:.0f}  sequential={seq:.0f}  colocated={coloc:.0f}")
print(f"  perfect overlap={ideal:.0f}; colocated is {interf:.0f}% of the way to full serialization")
print("=== 4.2.3 pipelines (coloc/alone) ===")
for i in ilps: print(f"  ILP{i}: alone={pipe[i]['alone']:.0f}  seq={pipe[i]['sequential']:.0f}  coloc={pipe[i]['colocated']:.0f}  ratio={pipe[i]['colocated']/pipe[i]['alone']:.2f}")
print("Figures ->", FIG)
