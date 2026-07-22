#!/usr/bin/env python3
"""Record a workload for the replay GUI (one run, no NCU).

  python3 record.py <run-name> -- <command ...>
  e.g.  python3 record.py dram_alone -- ../profiler/build/probe alone dram

Produces recordings/<name>/{report.nsys-rep, nvml.csv, meta.json} and then
calls extract.py to drop the run JSON into runs/<name>.json (auto-served).
"""
import argparse, json, os, subprocess, sys, time, signal

HERE = os.path.dirname(os.path.abspath(__file__))


def nvml_fields():
    """Newer drivers renamed throttle reasons; probe which spelling works."""
    base = "timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used"
    for throttle in ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None):
        q = base + ("," + throttle if throttle else "")
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return q
    sys.exit("nvidia-smi query failed:\n" + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("cmd", nargs="+", help="workload command (put it after --)")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="record only between cudaProfilerStart/Stop (workload must call them)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the un-profiled baseline run (no profiling-cost measurement)")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "recordings", args.name)
    os.makedirs(outdir, exist_ok=True)
    rep = os.path.join(outdir, "report")
    # PCI_BUS_ID ordering makes CUDA device 0 == nvidia-smi GPU 0 (this box has 2 GPUs;
    # the NVML poller and the workload must agree on which card they mean)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def region_ms(out):
        for line in out.splitlines():
            if line.startswith("REGION_MS"):
                return float(line.split()[1])
        return None

    # baseline pass: same command with NO profiler attached -> profiling-cost reference
    base_region = base_wall = None
    if not args.no_baseline:
        print("baseline pass (no profiler) ...", file=sys.stderr)
        t0 = time.time()
        rb = subprocess.run(args.cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
        base_wall = (time.time() - t0) * 1e3
        base_region = region_ms(rb.stdout)

    # NVML side-poller: 100 ms cadence into nvml.csv (wall-clock timestamps)
    q = nvml_fields()
    nvml_f = open(os.path.join(outdir, "nvml.csv"), "w")
    poller = subprocess.Popen(["nvidia-smi", "-i", "0", f"--query-gpu={q}",
                               "--format=csv,noheader,nounits", "-lms", "100"],
                              stdout=nvml_f, stderr=subprocess.DEVNULL)

    wall_start = time.time()
    # custom metric set = stock gb20x + an LTS row (L2 is absent from NVIDIA's stock set)
    mset = os.path.join(HERE, "gb20x-l2.config")
    nsys_cmd = ["nsys", "profile", "--gpu-metrics-devices=0",
                f"--gpu-metrics-set=file:{mset}", "--force-overwrite=true", "-o", rep]
    if args.capture:
        nsys_cmd += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
    r = subprocess.run([*nsys_cmd, *args.cmd], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall_end = time.time()
    open(os.path.join(outdir, "stdout.txt"), "w").write(r.stdout)   # TRACEBLK lines live here
    print("\n".join(l for l in r.stdout.splitlines() if not l.startswith("TRACEBLK"))[-2000:])

    poller.send_signal(signal.SIGINT); poller.wait(); nvml_f.close()

    # profiling cost: prefer the workload's self-timed region (excludes model load &
    # nsys report-gen); fall back to whole-process wall time.
    prof_region = region_ms(r.stdout)
    cost = None
    if base_region and prof_region:
        cost = dict(baseline_ms=base_region, profiled_ms=prof_region,
                    overhead=prof_region / base_region, source="compute region")
    elif base_wall:
        cost = dict(baseline_ms=base_wall, profiled_ms=(wall_end - wall_start) * 1e3,
                    overhead=(wall_end - wall_start) * 1e3 / base_wall, source="process wall time")
    if cost:
        print(f"profiling cost: {cost['baseline_ms']:.1f} ms -> {cost['profiled_ms']:.1f} ms "
              f"({cost['overhead']:.2f}x, {cost['source']})", file=sys.stderr)

    json.dump({"wall_start": wall_start, "wall_end": wall_end, "cmd": args.cmd, "cost": cost},
              open(os.path.join(outdir, "meta.json"), "w"))
    if r.returncode != 0:
        sys.exit(f"workload/nsys failed (exit {r.returncode})")
    print(f"recorded -> {outdir}")

    if not args.no_extract:
        subprocess.run([sys.executable, os.path.join(HERE, "extract.py"),
                        rep + ".nsys-rep", "--name", args.name,
                        "--nvml", os.path.join(outdir, "nvml.csv"),
                        "--meta", os.path.join(outdir, "meta.json"),
                        "--stdout", os.path.join(outdir, "stdout.txt")], check=True)


if __name__ == "__main__":
    main()
