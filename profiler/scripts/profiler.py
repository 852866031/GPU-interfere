#!/usr/bin/env python3
"""GPU colocation interference profiler.

Given two kernels, empirically profiles each one's demand on the shared GPU
resources we studied (DRAM bandwidth, L2, L1, FMA/warp-scheduler, FP64 pipeline)
by co-running it with canonical antagonist kernels, then predicts whether the two
can run concurrently without much interference. Needs NO perf-counter permissions.

Usage:
    python3 profiler.py <kernelA> <kernelB>
    kernels are names from `probe list` (sleep dram l2 l1 fma fp64); to profile
    your own kernel, add it to code/probe.cu's make() registry and rebuild.
"""
import subprocess, re, os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = os.path.join(ROOT, "build", "probe")

# antagonist -> (resource label, which experiment, inter/intra-SM)
ANTAG = [
    ("dram", "DRAM bandwidth",            "4.1.3", "inter-SM"),
    ("l2",   "L2 cache",                  "4.1.2", "inter-SM"),
    ("l1",   "L1 cache",                  "4.2.1", "intra-SM"),
    ("fma",  "FMA pipe / warp scheduler", "4.2.2", "intra-SM"),
    ("fp64", "FP64 pipeline",             "4.2.3", "intra-SM"),
]

def run(*args):
    out = subprocess.run([PROBE, *args], capture_output=True, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}).stdout
    return out.strip()

def kv(line):
    return dict(re.findall(r"(\w+)=([\w.\-]+)", line))

def static(k):
    return kv(run("static", k))

def usage_vector(k):
    """usage[resource] = max(0, slowdown-1) when k is co-run with resource's antagonist."""
    u = {}
    for antag, label, *_ in ANTAG:
        d = kv(run("coloc", k, antag))
        u[antag] = max(0.0, float(d["slowdown"]) - 1.0)
    return u

def measured(a, b):
    return float(kv(run("coloc", a, b))["slowdown"])

def rating(score):
    # score is a usage/contention magnitude (0 = none, ~1 = saturates)
    if score < 0.10: return "·  none"
    if score < 0.30: return "▪  low"
    if score < 0.60: return "▪▪ moderate"
    return "▪▪▪ HIGH"

def bar(score):
    n = min(5, int(round(score * 5)))
    return "█" * n + "░" * (5 - n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernelA"); ap.add_argument("kernelB")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    A, B = args.kernelA, args.kernelB

    print(f"Profiling '{A}' and '{B}' (co-running each with 5 antagonists)...", file=sys.stderr)
    sA, sB = static(A), static(B)
    uA, uB = usage_vector(A), usage_vector(B)
    mAB, mBA = measured(A, B), measured(B, A)

    # ---- decision model ----
    # contention on resource r requires BOTH kernels to use it -> min of usages.
    contention = {an: min(uA[an], uB[an]) for an, *_ in ANTAG}
    worst = max(ANTAG, key=lambda t: contention[t[0]])
    worst_an, worst_label = worst[0], worst[1]
    risk = contention[worst_an]
    pred = max(mAB, mBA)   # predicted worst-direction slowdown (validation target)

    if risk < 0.10:
        verdict, advice = "SAFE TO COLOCATE", "The two kernels stress different resources; expect minimal interference."
    elif risk < 0.30:
        verdict, advice = "MOSTLY SAFE", f"Minor contention on {worst_label}; small slowdown expected."
    elif risk < 0.60:
        verdict, advice = "COLOCATE WITH CAUTION", f"Both kernels lean on {worst_label}; noticeable slowdown likely."
    else:
        verdict, advice = "DO NOT COLOCATE", f"Both kernels saturate {worst_label}; severe interference expected."

    # ---- report ----
    L = []
    L.append(f"# GPU Colocation Interference Report\n")
    L.append(f"**Kernel A:** `{A}`  **·  Kernel B:** `{B}`  ·  GPU: {sA.get('sms','?')} SMs\n")
    L.append(f"> ## Verdict: **{verdict}**\n> {advice}\n")

    L.append("## 1. Per-kernel resource demand\n")
    L.append("Measured as the slowdown each kernel suffers when co-run with an antagonist that "
             "saturates the given resource (higher = the kernel leans on that resource more).\n")
    L.append("| Shared resource | Exp | Level | " + f"`{A}` | " + f"`{B}` |")
    L.append("|---|---|---|---|---|")
    for an, label, exp, lvl in ANTAG:
        L.append(f"| {label} | {exp} | {lvl} | {bar(min(1,uA[an]))} {uA[an]:.2f} | {bar(min(1,uB[an]))} {uB[an]:.2f} |")
    L.append("")
    L.append(f"SM occupancy (static): `{A}` = {float(sA['occupancy']):.2f} "
             f"({sA['max_blocks_per_sm']} blk/SM, {sA['threads']} thr) · "
             f"`{B}` = {float(sB['occupancy']):.2f} "
             f"({sB['max_blocks_per_sm']} blk/SM, {sB['threads']} thr)\n")

    L.append("## 2. Contention analysis (both must use a resource to fight over it)\n")
    L.append("| Shared resource | both use it? | predicted contention |")
    L.append("|---|---|---|")
    for an, label, *_ in ANTAG:
        both = "yes" if contention[an] >= 0.10 else "no"
        L.append(f"| {label} | {both} | {rating(contention[an])} |")
    L.append("")
    if risk >= 0.10:
        L.append(f"**Bottleneck resource:** {worst_label} (contention score {risk:.2f}).\n")
    else:
        L.append("**No shared resource is used heavily by both kernels.**\n")

    L.append("## 3. Validation — directly measured colocation\n")
    L.append("The model predicts from single-antagonist probes; here we actually run the two kernels together:\n")
    L.append("| Direction | measured slowdown |")
    L.append("|---|---|")
    L.append(f"| `{A}` slowed by `{B}` | {mAB:.2f}x |")
    L.append(f"| `{B}` slowed by `{A}` | {mBA:.2f}x |")
    L.append("")
    L.append(f"Worst-case measured slowdown **{pred:.2f}x** — consistent with the "
             f"{'high' if risk>=0.6 else 'moderate' if risk>=0.3 else 'low' if risk>=0.1 else 'negligible'} "
             f"contention predicted above.\n")

    L.append("## 4. Recommendation\n")
    L.append(f"**{verdict}.** {advice}")
    if risk >= 0.30:
        L.append(f"\nIf you must colocate, isolate the bottleneck: e.g. restrict each kernel to a "
                 f"disjoint set of SMs (MPS) only helps for *intra-SM* resources; for *inter-SM* "
                 f"resources (L2, DRAM) even separate SMs contend (see §4.1).")

    report = "\n".join(L)
    out = args.out or os.path.join(ROOT, "reports", f"report_{A}_vs_{B}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(report + "\n")
    print(report)
    print(f"\n[saved -> {out}]", file=sys.stderr)

if __name__ == "__main__":
    main()
