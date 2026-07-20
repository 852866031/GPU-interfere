#!/usr/bin/env python3
"""Parse Section 4.3 log -> CSV + figure."""
import re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# @RESULT can be glued to a stray "[FP32] ..." prefix (stdout interleave) -> use search
R = re.compile(r"@RESULT exp=mm size=(\d+) case=(\w+) (?:ms|ratio)=([\d.]+)")
data = {}   # size -> {mm_alone, mm_coloc, slowdown}
for line in open(os.path.join(RES, "mm_pytorch.log")):
    m = R.search(line)
    if m:
        size = int(m.group(1)); data.setdefault(size, {})[m.group(2)] = float(m.group(3))

sizes = sorted(data)
with open(os.path.join(RES, "mm_pytorch.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["matrix_size", "mm_alone_ms", "mm_coloc_ms", "slowdown"])
    for s in sizes:
        d = data[s]; w.writerow([s, d["mm_alone"], d["mm_coloc"], round(d["mm_coloc"]/d["mm_alone"], 3)])

alone = [data[s]["mm_alone"] for s in sizes]
coloc = [data[s]["mm_coloc"] for s in sizes]
slow  = [data[s]["mm_coloc"]/data[s]["mm_alone"] for s in sizes]
labels = [f"{s}³" for s in sizes]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))

# (a) latency alone vs colocated (log-y: values span 0.014 .. 2.3 ms)
x = range(len(sizes)); w = 0.38
a1.bar([i-w/2 for i in x], alone, w, label="matmul alone", color="#4C78A8")
a1.bar([i+w/2 for i in x], coloc, w, label="matmul + fma kernel (colocated)", color="#E45756")
a1.set_yscale("log"); a1.set_xticks(list(x)); a1.set_xticklabels(labels)
a1.set_xlabel("matmul size (N x N x N)"); a1.set_ylabel("matmul latency (ms, log scale)")
a1.set_title("4.3a  PyTorch matmul latency: alone vs colocated")
a1.legend(fontsize=8)

# (b) slowdown vs size
a2.plot(x, slow, "-o", color="#8B0000", lw=2)
a2.axhline(1.0, color="#888", ls=":", lw=1); a2.text(1.8, 0.9, "1.0x = no interference", fontsize=8, color="#888", ha="center")
for i, s in enumerate(slow):
    a2.text(i, s+0.12, f"{s:.1f}x", ha="center", fontsize=9, weight="bold", color="#8B0000")
a2.set_xticks(list(x)); a2.set_xticklabels(labels)
a2.set_xlabel("matmul size (N x N x N)"); a2.set_ylabel("slowdown (colocated / alone)")
a2.set_ylim(0.8, max(slow)*1.2); a2.set_title("4.3b  Matmul slowdown from a colocated compute kernel")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_431_mm_interference.png"), dpi=150); plt.close(fig)

print("=== 4.3 PyTorch matmul interference ===")
for s in sizes:
    d = data[s]
    print(f"  {s:5d}^3: alone={d['mm_alone']:.4f} ms  coloc={d['mm_coloc']:.4f} ms  slowdown={d['mm_coloc']/d['mm_alone']:.2f}x")
print("Figure ->", FIG)
