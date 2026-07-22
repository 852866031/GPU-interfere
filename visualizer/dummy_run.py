"""Dummy run generator — produces the SAME JSON shape that extract.py will
eventually produce from a real recording (the data contract).

Shape:
{
  name, sm_count, tmax_us,
  kernels:  {name: {slot, grid, cap}},
  launches: [{k, stream, submit, start, end}],           # µs
  lanes:    {res_us, names, data: [[6 floats], ...]},    # % of peak per bucket
  nvml:     {res_us, data: [{power, sm_clk, mem_clk, temp, vram_gb, throttle}, ...]}
}
"""
import math

SM_COUNT = 170
SCALE = 40           # stretch the toy schedule to a realistic ~118 ms total
LANE_RES_US = 100    # PM-sampling cadence
NVML_RES_US = 10_000 # NVML cadence

#                 name          slot grid cap  thr/blk [smact issue tensor l2 dram pcie]
KERNELS = {
    "memcpy_h2d":  dict(slot=0, grid=0,   cap=0, block=0,    lanes=[2,  0,  0,  5,  8, 92]),
    "embedding":   dict(slot=1, grid=340, cap=2, block=512,  lanes=[62, 12,  0, 30, 76,  0]),
    "rmsnorm":     dict(slot=2, grid=170, cap=1, block=1024, lanes=[85, 18,  0, 46, 66,  0]),
    "qkv_gemm":    dict(slot=3, grid=512, cap=2, block=128,  lanes=[95, 55, 88, 40, 26,  0]),
    "attention":   dict(slot=4, grid=96,  cap=1, block=128,  lanes=[54, 45, 60, 66, 15,  0]),  # underfilled
    "mlp_gemm":    dict(slot=5, grid=680, cap=2, block=256,  lanes=[97, 60, 92, 34, 30,  0]),
    "logits_gemm": dict(slot=6, grid=850, cap=2, block=256,  lanes=[96, 55, 85, 44, 36,  0]),
    "probe_l1":    dict(slot=7, grid=170, cap=1, block=64,   lanes=[95,  8,  0, 46,  2,  0]),
    "probe_fma":   dict(slot=8, grid=170, cap=1, block=128,  lanes=[96, 83,  0,  1,  0,  0]),
}
LANE_NAMES = ["SM active", "Warp issue", "Tensor pipe", "L2 thruput", "DRAM BW", "PCIe"]


def _schedule():
    launches, t = [], 0

    def add(k, stream, start, dur):
        s, d = start * SCALE, dur * SCALE
        launches.append(dict(k=k, stream=stream,
                             submit=max(0, s - 380 * SCALE), start=s, end=s + d))
        return start + dur

    t = add("memcpy_h2d", 0, 10, 150)
    t = add("embedding", 0, t + 12, 120)
    for _ in range(2):                                   # 2 transformer layers
        t = add("rmsnorm",   0, t + 8, 70)
        t = add("qkv_gemm",  0, t + 8, 200)
        t = add("attention", 0, t + 8, 160)
        t = add("mlp_gemm",  0, t + 8, 280)
    t = add("logits_gemm", 0, t + 10, 240)
    add("probe_l1",  1, t + 120, 520)                    # colocation demo
    add("probe_fma", 2, t + 180, 520)
    tmax = (t + 120 + 520 + 60) * SCALE
    return launches, tmax


def _lanes_at(launches, tt):
    v = [0.0] * 6
    for l in launches:
        if l["start"] <= tt < l["end"]:
            for i, x in enumerate(KERNELS[l["k"]]["lanes"]):
                v[i] += x
    return [max(0.0, min(100.0, x + (2.5 * math.sin(tt * 0.0028 + i * 2.7) if x > 1 else 0)))
            for i, x in enumerate(v)]


def make_dummy_run():
    launches, tmax = _schedule()

    lanes = [_lanes_at(launches, t) for t in range(0, tmax, LANE_RES_US)]

    nvml, vram = [], 2.1
    for t in range(0, tmax, NVML_RES_US):
        smact = _lanes_at(launches, t)[0]
        if t <= 160 * SCALE:                              # H2D fills VRAM
            vram = min(18.2, 2.1 + 16.1 * t / (160 * SCALE))
        power = round(70 + 4.9 * smact + 2.5 * math.sin(t * 0.0028 + 9), 1)
        throttled = power > 540
        nvml.append(dict(power=power,
                         sm_clk=2410 if throttled else 2620, mem_clk=1750,
                         temp=round(42 + 0.06 * smact + t / 8800.0, 1),
                         vram_gb=round(vram, 1),
                         throttle="power cap" if throttled else None))

    return dict(name="dummy_prefill", sm_count=SM_COUNT, tmax_us=tmax,
                kernels={k: {kk: v for kk, v in d.items() if kk != "lanes"}
                         for k, d in KERNELS.items()},
                launches=launches,
                lanes=dict(res_us=LANE_RES_US, names=LANE_NAMES, data=lanes),
                nvml=dict(res_us=NVML_RES_US, data=nvml))


if __name__ == "__main__":
    import json
    r = make_dummy_run()
    print(f"tmax = {r['tmax_us']/1000:.1f} ms, {len(r['launches'])} launches, "
          f"{len(r['lanes']['data'])} lane buckets, {len(r['nvml']['data'])} nvml samples, "
          f"json = {len(json.dumps(r))/1024:.0f} KB")
