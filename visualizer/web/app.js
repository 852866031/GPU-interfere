"use strict";
/* Replay GUI frontend — pure consumer of /api/runs/<name> (see dummy_run.py contract). */

const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SLOT = [null, "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
/* per-resource colors by NAME (lane sets differ between dummy and real runs) */
const LANE_COLOR = [["SM active", "--s1"], ["Warp occup", "--s2"], ["Warp issue", "--s6"],
                    ["Tensor", "--s3"], ["L2", "--s5"], ["DRAM", "--s4"], ["PCIe", "--s7"]];
const laneColor = name => (LANE_COLOR.find(([p]) => name.startsWith(p)) ?? [0, "--ink-3"])[1];

let RUN = null, L = [], TMAX = 1, STREAMS = [];   // current run data
const warpsPer = k => Math.ceil((k.block || 128) / 32);   // warps per block (threads/32)
let blocksByLaunch = {}, SMIDX = {}, REVSMID = null;   // real block-trace indexes
const kcolor = k => k.slot ? css(SLOT[k.slot]) : css("--ink-3");

/* ---------------- data access (bucketed, from the API payload) ---------------- */
function active(tt) { return L.filter(l => l.start <= tt && tt < l.end); }
function queued(tt) { return L.filter(l => l.submit <= tt && tt < l.start); }
function lanesAt(tt) {
  const d = RUN.lanes.data;
  return d[Math.min(d.length - 1, Math.floor(tt / RUN.lanes.res_us))];
}
function nvmlAt(tt) {
  const d = RUN.nvml.data;
  return d[Math.min(d.length - 1, Math.floor(tt / RUN.nvml.res_us))];
}
/* block->SM placement: REAL trace data when present, dummy model otherwise */
function placement(tt) {
  const per = Array.from({length: RUN.sm_count}, () => []);
  if (RUN.blocks) {
    for (const b of RUN.blocks) {
      if (b.t0 > tt || tt >= b.t1) continue;
      const idx = SMIDX[b.smid] ?? -1;
      if (idx < 0 || idx >= per.length) continue;
      const e = per[idx].find(x => x.k.name === b.k);
      if (e) e.c++; else per[idx].push({k: RUN.kernels[b.k], c: 1});
    }
    return per;
  }
  for (const l of active(tt)) {
    if (!l.k.grid) continue;
    const off = l.stream * 37;
    const n = Math.min(l.k.grid, RUN.sm_count);
    for (let i = 0; i < n; i++) {
      const c = Math.min(l.k.cap, Math.ceil((l.k.grid - i) / RUN.sm_count));
      per[(i + off) % RUN.sm_count].push({k: l.k, c});
    }
  }
  return per;
}

/* ---------------- die DOM (rebuilt per run) ---------------- */
let smEls = [];
function buildDie() {
  smEls = []; $("bank0").innerHTML = ""; $("bank1").innerHTML = "";
  const per = Math.floor(RUN.sm_count / 12), extra = RUN.sm_count - per * 12;
  const GPCS = Array.from({length: 12}, (_, g) => per + (g < extra ? 1 : 0));
  let smIdx = 0;
  GPCS.forEach((n, g) => {
    const box = document.createElement("div"); box.className = "gpc";
    box.innerHTML = `<div class="glabel">GPC${g}</div>`;
    const grid = document.createElement("div"); grid.className = "smgrid";
    for (let i = 0; i < n; i++) {
      const d = document.createElement("div"); d.className = "sm"; d.dataset.idx = smIdx++;
      grid.appendChild(d); smEls.push(d);
    }
    box.appendChild(grid);
    $(g < 6 ? "bank0" : "bank1").appendChild(box);
  });
}

/* block accounting for a running launch: done / executing / pending.
   Dummy model only — real runs need the block trace (returns null until then). */
function blockStats(l, tt) {
  if (RUN.blocks) {   // real: count trace events for THIS launch (untraced launches -> null)
    const evs = blocksByLaunch[l._i];
    if (!evs) return null;
    const G = l.grid ?? l.k.grid;
    let started = 0, exec = 0;
    for (const b of evs) { if (b.t0 <= tt) { started++; if (tt < b.t1) exec++; } }
    return {G, done: started - exec, exec, pend: Math.max(0, G - started)};
  }
  if (RUN.real) return null;   // real run without trace: unknowable
  const grid = l.grid ?? l.k.grid;
  if (!grid) return null;
  const G = grid, resident = Math.min(G, RUN.sm_count * l.k.cap);
  const done = Math.floor(G * (tt - l.start) / (l.end - l.start));
  const exec = Math.min(resident, G - done);
  return {G, done, exec, pend: Math.max(0, G - done - exec)};
}

/* ---------------- render one frame ---------------- */
const chip = (k, extra = "") =>
  `<span class="chip"><span class="sw" style="background:${kcolor(k)}"></span>${k.name}${extra}</span>`;

/* run-length group consecutive same-kernel launches into "name ×N" chips,
   capped at `max` groups with a "… +N more" tail (queues can hold thousands) */
function groupChips(list, max) {
  const groups = [];
  for (const l of list) {
    const g = groups[groups.length - 1];
    if (g && g.k === l.k) g.n++;
    else groups.push({k: l.k, n: 1});
  }
  let html = groups.slice(0, max).map(g =>
    chip(g.k, g.n > 1 ? ` <span class="sub">×${g.n}</span>` : "")).join("");
  if (groups.length > max) {
    const rest = groups.slice(max).reduce((a, g) => a + g.n, 0);
    html += `<span class="chip">… +${rest} more</span>`;
  }
  return html;
}

function render(tt) {
  $("tlabel").textContent = `t = ${(tt / 1000).toFixed(2)} ms`;
  $("scrub").value = tt / TMAX;
  const act = active(tt), q = queued(tt), v = lanesAt(tt), nv = nvmlAt(tt);
  /* lane lookup by name — dummy and real runs have different lane sets */
  const lv = name => { const i = RUN.lanes.names.findIndex(n => n.startsWith(name)); return i < 0 ? null : v[i]; };

  /* host row */
  const recent = L.filter(l => tt - 2400 < l.submit && l.submit <= tt).at(-1);
  const apiCall = k => k.name.startsWith("Memcpy") ? `cudaMemcpyAsync(${k.name.replace("Memcpy ", "")})` : `cudaLaunchKernel(${k.name})`;
  $("launchfeed").innerHTML = recent
    ? `<span class="chip"><span class="sw" style="background:${kcolor(recent.k)}"></span>${apiCall(recent.k)}</span>`
    : `<span class="empty">idle</span>`;
  $("queues").innerHTML = STREAMS.map(s => {
    const qs = q.filter(l => l.stream === s);
    return `<div class="chipline" style="margin:2px 0"><span class="sub" style="width:80px">stream ${s}${qs.length ? ` (${qs.length})` : ""}</span>
      ${qs.length ? groupChips(qs, 5) : '<span class="empty">empty</span>'}</div>`;
  }).join("");
  const heads = q.filter(l => l.start - tt < 1600);
  $("gigathread").innerHTML = heads.length ? groupChips(heads, 3)
    : `<span class="empty">${act.length ? "dispatching" : "idle"}</span>`;
  $("running").innerHTML = act.length ? act.map(l => {
    const p = Math.round(100 * (tt - l.start) / (l.end - l.start));
    const s = blockStats(l, tt);
    const blk = l.k.name.startsWith("Memcpy") ? `<span class="sub">copy engine</span>`
              : s ? `<span class="sub">exec <b>${s.exec}</b> · pend <b>${s.pend}</b> / ${s.G}</span>`
              : `<span class="sub">grid ${l.grid ?? l.k.grid}</span>`;
    return `<span class="chip run"><span class="sw" style="background:${kcolor(l.k)}"></span>${l.k.name}
            ${blk}<span class="prog"><div style="width:${p}%"></div></span></span>`;
  }).join("") : `<span class="empty">idle</span>`;

  /* pending-blocks box (in the brace funnel, under Running): always present */
  const pendGroups = act.map(l => {
    const s = blockStats(l, tt);
    if (!s || !s.pend) return "";
    const n = Math.min(12, Math.ceil(s.pend / 64));      // 1 square ≈ 64 blocks
    const sq = Array.from({length: n}, () => `<i style="background:${kcolor(l.k)}"></i>`).join("");
    return `<span class="pendgrp">${sq}<span class="cnt">${l.k.name} · ${s.pend}</span></span>`;
  }).filter(Boolean).join("");
  $("pendrow").innerHTML =
    `<span class="pendbox"><span class="plabel">Pending blocks</span>${pendGroups ||
      `<span class="empty">${RUN.real && !RUN.blocks ? "n/a — needs block trace" : "none"}</span>`}</span>`;

  /* PCIe */
  const pcie = lv("PCIe") ?? 0;
  $("pciebar").style.width = pcie + "%";
  $("pciebar").style.background = css("--s7");
  $("pcieval").innerHTML = `<b style="color:${css("--s7")}">${(pcie * 0.63).toFixed(0)} GB/s</b> · ${pcie.toFixed(0)}% of peak`;

  /* SM tiles as warp-slot gauges: fill bottom-up by resident warps / 48, stacked per kernel.
     Real run without trace: honest uniform tint (device avg) instead. */
  const WSLOTS = 48;
  let slotsUsed = 0;
  if (RUN.real && !RUN.blocks) {
    /* no per-SM data: every tile shows the SAME fill height = device-avg occupancy.
       Uniformity is the honest signal that only the average is known. */
    const occ = lv("Warp occup") ?? 0;
    slotsUsed = Math.round(occ / 100 * WSLOTS * RUN.sm_count);
    const h = Math.max(occ, 2).toFixed(1);
    smEls.forEach(el => {
      el.style.background = `linear-gradient(to top, ${css("--s1")} 0% ${h}%, ${css("--grid")} ${h}% 100%)`;
      el._res = null;   // tooltip: no per-SM data
    });
  } else {
    const per = placement(tt);
    smEls.forEach((el, i) => {
      const res = per[i];
      if (!res.length) { el.style.background = "var(--grid)"; el._res = res; return; }
      const stops = []; let acc = 0;
      for (const r of res) {
        const wp = r.c * warpsPer(r.k);
        slotsUsed += wp;
        const h = Math.max(100 * wp / WSLOTS, 9);   // min sliver so tiny tenants stay visible
        stops.push(`${kcolor(r.k)} ${acc.toFixed(0)}% ${Math.min(100, acc + h).toFixed(0)}%`);
        acc = Math.min(100, acc + h);
      }
      el.style.background = `linear-gradient(to top, ${stops.join(", ")}, ${css("--grid")} ${acc.toFixed(0)}% 100%)`;
      el._res = res;
    });
  }
  const totSlots = WSLOTS * RUN.sm_count;
  $("warpbar").style.width = (100 * slotsUsed / totSlots) + "%";
  $("warpbar").style.background = css("--s2");
  $("warpval").innerHTML = `<b style="color:${css("--s2")}">${slotsUsed}</b> / ${totSlots} (${(100 * slotsUsed / totSlots).toFixed(1)}%)`;

  /* L2 + VRAM */
  const l2v = lv("L2");
  if (l2v === null) {
    $("l2slab").innerHTML = `L2 cache · 96 MB — <span class="na">no L2 lane in this recording (re-record with the gb20x-l2 metric set)</span>`;
    $("l2slab").style.background = "";
  } else {
    const l2t = Math.round(l2v);
    $("l2slab").innerHTML = `L2 cache · 96 MB — throughput <b style="color:${l2t > 45 ? css("--ink-1") : css("--s5")}">${l2t}%</b> of peak (rate, not fill)`;
    $("l2slab").style.background = `color-mix(in oklab, ${css("--s5")} ${Math.round(l2t * 0.7)}%, ${css("--surface-1")})`;
  }
  const DRAM_PEAK = 1792;   // GB/s, GDDR7 spec peak on RTX 5090 (512-bit @ 28 Gbps)
  const dram = lv("DRAM") ?? 0;
  const vramPct = 100 * nv.vram_gb / 32;
  $("vramval").innerHTML = `<b style="color:${css("--s1")}">${nv.vram_gb.toFixed(1)}</b> / 32 GB (${vramPct.toFixed(0)}%)`;
  $("vrambar").style.width = vramPct + "%";
  $("vrambar").style.background = css("--s1");
  $("dramval").innerHTML = `BW <b style="color:${css("--s4")}">${(dram / 100 * DRAM_PEAK).toFixed(0)}</b> / ${DRAM_PEAK} GB/s (${dram.toFixed(0)}%)`;
  $("drambar").style.width = dram + "%";
  $("drambar").style.background = css("--s4");

  /* NVML strip */
  const item = (label, val, color) =>
    `<span><span class="dot" style="background:${css(color)};display:inline-block;vertical-align:-1px;margin-right:4px"></span>${label} <b style="color:${css(color)}">${val}</b></span>`;
  $("nvml").innerHTML =
    item("Power", nv.power.toFixed(0) + " W / 575", "--s8") +
    item("SM clock", nv.sm_clk + " MHz", "--s2") +
    item("Mem clock", nv.mem_clk + " MHz", "--q250") +
    item("Temp", nv.temp.toFixed(0) + " °C", "--s6") +
    item("VRAM", nv.vram_gb.toFixed(1) + " / 32 GB", "--s1") +
    `<span class="throttle"><span class="dot" style="background:${nv.throttle ? "var(--st-warn)" : "var(--st-good)"}"></span>
      ${nv.throttle ? "⚠ throttle: " + nv.throttle : "no throttle"}</span>`;

  drawLanes(tt);
  drawKtl(tt);
  drawHots(tt);
}

/* ---------------- lanes canvas ---------------- */
const cvs = $("lanes"), ctx = cvs.getContext("2d");
function drawLanes(tt) {
  const n = RUN.lanes.names.length;
  const lh = 30, gap = 6, H = 8 + n * (lh + gap) + 4;
  const W = cvs.clientWidth;
  cvs.style.height = H + "px";
  cvs.width = W * devicePixelRatio; cvs.height = H * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  const x0 = 92, vw = 48, w = W - x0 - vw;   // plot ends early; values live outside
  ctx.clearRect(0, 0, W, H);
  ctx.font = "10px system-ui";
  for (let li = 0; li < n; li++) {
    const y0 = 8 + li * (lh + gap), col = css(laneColor(RUN.lanes.names[li]));
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(RUN.lanes.names[li], x0 - 8, y0 + lh / 2);
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, w, lh);
    ctx.fillStyle = col; ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.moveTo(x0, y0 + lh);
    for (let px = 0; px <= w; px++) {
      const val = lanesAt(px / w * TMAX)[li];
      ctx.lineTo(x0 + px, y0 + lh - lh * val / 100);
    }
    ctx.lineTo(x0 + w, y0 + lh); ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = col; ctx.font = "bold 11px system-ui"; ctx.textAlign = "left";
    ctx.fillText(lanesAt(tt)[li].toFixed(0) + "%", x0 + w + 6, y0 + lh / 2);
    ctx.font = "10px system-ui";
  }
  const cx = x0 + (tt / TMAX) * w;
  ctx.strokeStyle = css("--ink-1"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 4); ctx.lineTo(cx, H - 4); ctx.stroke();
}

/* ---------------- kernel launch timeline ---------------- */
/* overlap slots: concurrent launches get different thirds of the big Running row */
function assignSlots() {
  const slotEnd = [-Infinity, -Infinity, -Infinity];
  for (const l of [...L].sort((a, b) => a.start - b.start)) {
    let s = slotEnd.findIndex(e => e <= l.start);
    if (s < 0) s = slotEnd.indexOf(Math.min(...slotEnd));   // >3 concurrent: reuse least-busy slot
    l._slot = s; slotEnd[s] = l.end;
  }
}

function drawKtl(tt) {
  const c = $("ktl"), g = c.getContext("2d");
  const hasPh = !!RUN.phases;
  const py = 4, ph = 14;                                    // phase strip (NVTX ranges)
  const ry = hasPh ? py + ph + 6 : 6, rh = 62, sub = rh / 3;
  const H = ry + rh + 8;
  const W = c.clientWidth;
  c.style.height = H + "px";
  c.width = W * devicePixelRatio; c.height = H * devicePixelRatio;
  g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, W, H);
  const x0 = 92, vw = 48, w = W - x0 - vw;                  // same margins as the lanes
  const X = t => x0 + w * t / TMAX;

  g.font = "10px system-ui"; g.textBaseline = "middle";
  if (hasPh) {                                              // alternating translucent bands
    g.fillStyle = css("--ink-2"); g.textAlign = "right";
    g.fillText("Phase", x0 - 8, py + ph / 2);
    RUN.phases.forEach((p, i) => {
      const a = X(p.start), b = X(p.end);
      g.fillStyle = css("--ink-3"); g.globalAlpha = i % 2 ? 0.32 : 0.18;
      g.fillRect(a, py, Math.max(1, b - a), ph);
      g.globalAlpha = 1;
      if (b - a > 46) {
        g.fillStyle = css("--ink-1"); g.textAlign = "left";
        g.fillText(p.name, a + 4, py + ph / 2);
      }
    });
  }
  g.fillStyle = css("--ink-2"); g.textAlign = "right";
  g.fillText("Running", x0 - 8, ry + rh / 2);
  g.strokeStyle = css("--grid"); g.lineWidth = 1;
  g.strokeRect(x0, ry, w, rh);

  for (const l of L) {
    const y = ry + l._slot * sub, bw = Math.max(1, X(l.end) - X(l.start));
    g.fillStyle = kcolor(l.k);
    g.beginPath(); g.roundRect(X(l.start), y + 2, bw, sub - 4, 2); g.fill();
  }
  /* sweeping time cursor, like the rate lanes */
  const cx = X(tt);
  g.strokeStyle = css("--ink-1"); g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx, 2); g.lineTo(cx, H - 2); g.stroke();
}

/* ---------------- hotspot charts (cumulative up to the cursor) ---------------- */
function drawHots(tt) {
  const c = $("hots"), g = c.getContext("2d");
  const W = c.clientWidth, H = 210;
  c.width = W * devicePixelRatio; c.height = H * devicePixelRatio;
  g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, W, H);

  const stats = {};
  for (const l of L) if (tt > l.start) {
    const s = stats[l.k.name] ??= {k: l.k, dur: 0, n: 0};
    s.dur += Math.min(tt, l.end) - l.start;
    s.n++;
  }
  let rows = Object.values(stats).sort((a, b) => b.dur - a.dur);
  if (rows.length > 8) {   // fold the tail: real workloads have dozens of kernel names
    const other = {k: {name: `(other · ${rows.length - 7})`, slot: 0}, dur: 0, n: 0};
    for (const s of rows.slice(7)) { other.dur += s.dur; other.n += s.n; }
    rows = [...rows.slice(0, 7), other];
  }

  const charts = [
    {x0: 0,          w: W / 2 - 14, title: "Hotspots: total duration", val: s => s.dur / 1000, fmt: v => v.toFixed(2) + " ms"},
    {x0: W / 2 + 14, w: W / 2 - 14, title: "Hotspots: launch count",   val: s => s.n,          fmt: v => v.toFixed(0) + "×"},
  ];
  for (const ch of charts) {
    g.fillStyle = css("--ink-2"); g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.font = "600 11px system-ui";
    g.fillText(ch.title, ch.x0 + 128, 13);
    g.font = "10px system-ui";
    if (!rows.length) { g.fillStyle = css("--ink-3"); g.fillText("(nothing launched yet)", ch.x0 + 128, 34); continue; }
    const lab = 128, valw = 64, bw = ch.w - lab - valw;
    const max = Math.max(...rows.map(ch.val)) || 1;
    const rh = Math.min(21, (H - 24) / rows.length);
    rows.forEach((s, i) => {
      const y = 20 + i * rh;
      g.fillStyle = css("--ink-2"); g.textAlign = "right"; g.textBaseline = "middle";
      g.fillText(s.k.name, ch.x0 + lab - 8, y + rh / 2);
      const w2 = Math.max(1, bw * ch.val(s) / max);
      g.fillStyle = kcolor(s.k);
      g.beginPath(); g.roundRect(ch.x0 + lab, y + 2, w2, rh - 6, 2); g.fill();
      g.fillStyle = css("--ink-2"); g.textAlign = "left";
      g.fillText(ch.fmt(ch.val(s)), ch.x0 + lab + w2 + 6, y + rh / 2);
    });
  }
}

/* ---------------- static panels ---------------- */
function buildLegend() {
  $("legend").innerHTML = Object.values(RUN.kernels).map(k =>
    `<span class="chip"><span class="sw" style="background:${kcolor(k)}"></span>${k.name}
     <span class="sub">grid ${k.grid || "—"}${k.block ? ` · ${warpsPer(k)}w/blk` : ""}${k.cap ? ` · ≤${k.cap} blk/SM` : ""}</span></span>`).join("");
  /* compact name legend under the launch timeline */
  $("ktllegend").innerHTML = Object.values(RUN.kernels).map(k =>
    `<span class="chip"><span class="sw" style="background:${kcolor(k)}"></span>${k.name}</span>`).join("");

  const sw = c => `<span class="sw" style="background:${css(c)}"></span>`;
  const item = (c, l) => `<span class="chip">${sw(c)}${l}</span>`;
  const CLASSES = [
    ["~ns", "block→SM trace (%smid + globaltimer)",
      `<span class="chip"><span class="sw" style="background:linear-gradient(45deg,${css("--s3")},${css("--s7")})"></span>SM tiles / block squares (kernel colors)</span>`],
    ["exact per launch", "CUPTI activity timeline",
      item("--ink-3", "launch feed · stream queues · GigaThread · running + progress")],
    [`~${RUN.lanes.res_us} µs`, "PM sampling (nsys gpu-metrics)",
      RUN.lanes.names.map(n => item(laneColor(n), n)).join("")],
    [`~${RUN.nvml.res_us / 1000} ms`, "NVML poller",
      item("--s8", "Power") + item("--s2", "SM clock") + item("--q250", "Mem clock") +
      item("--s6", "Temp") + item("--s1", "VRAM capacity") + item("--st-warn", "throttle status")],
    ["static / N-A", "—", item("--ink-3", "NVLink (absent on GeForce) · GPU spec labels")],
  ];
  $("resmap").innerHTML = CLASSES.map(([res, src, chips]) =>
    `<div class="resrow"><b>${res}</b><span class="sub src">${src}</span>
     <span class="chipline">${chips}</span></div>`).join("");
}

/* ---------------- fan brace: anchor top ends under the Running box ---------------- */
function alignFan() {
  const fan = document.querySelector(".fan");
  const box = $("running").closest(".panel");
  if (!fan || !box) return;
  const f = fan.getBoundingClientRect(), b = box.getBoundingClientRect();
  const pct = x => Math.max(0, Math.min(100, 100 * (x - f.left) / f.width));
  fan.children[0].setAttribute("x1", pct(b.left + 12));
  fan.children[1].setAttribute("x1", pct(b.right - 12));
}
window.addEventListener("resize", alignFan);

/* ---------------- tooltip + playback ---------------- */
const tip = $("tip");
document.addEventListener("mousemove", e => {
  const sm = e.target.closest(".sm");
  if (!sm) { tip.style.display = "none"; return; }
  const res = sm._res;
  const i = +sm.dataset.idx;
  const slots = res ? res.reduce((a, r) => a + r.c * warpsPer(r.k), 0) : 0;
  tip.textContent = `SM ${String(REVSMID ? (REVSMID[i] ?? i) : i).padStart(3, "0")}` +
    (res ? ` — ${slots}/48 warp slots\n` : "\n") +
    (res === null ? "uniform tint (device avg) —\nper-SM data needs the block trace"
     : res.length ? res.map(r => `${r.k.name}: ${r.c} blk × ${warpsPer(r.k)} warps`).join("\n") : "idle");
  tip.style.display = "block";
  tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 10) + "px";
});

let cur = 0, playing = false, last = 0;
function tick(ts) {
  if (playing && RUN) {
    cur += (ts - last) * (TMAX / 20000) * parseFloat($("speed").value);  // 1x = ~20 s replay
    if (cur >= TMAX) { cur = TMAX; playing = false; $("play").textContent = "▶ Play"; }
    render(cur);
  }
  last = ts; requestAnimationFrame(tick);
}
$("play").onclick = () => {
  if (!playing && cur >= TMAX) cur = 0;
  playing = !playing; $("play").textContent = playing ? "⏸ Pause" : "▶ Play";
};
$("scrub").oninput = e => { playing = false; $("play").textContent = "▶ Play"; cur = e.target.value * TMAX; if (RUN) render(cur); };

/* ---------------- load ---------------- */
async function loadRun(name) {
  RUN = await (await fetch("/api/runs/" + name)).json();
  Object.entries(RUN.kernels).forEach(([n, k]) => k.name = n);
  L = RUN.launches.map(l => ({...l, k: RUN.kernels[l.k]}));
  TMAX = RUN.tmax_us;
  STREAMS = [...new Set(L.map(l => l.stream))].sort((a, b) => a - b);   // only streams the run actually uses
  L.forEach((l, i) => l._i = i);
  blocksByLaunch = {}; SMIDX = {}; REVSMID = null;
  if (RUN.blocks) {   // %smid values need not be contiguous: map observed ids -> tiles
    REVSMID = [...new Set(RUN.blocks.map(b => b.smid))].sort((a, b) => a - b);
    REVSMID.forEach((s, i) => SMIDX[s] = i);
    for (const b of RUN.blocks) (blocksByLaunch[b.li] ??= []).push(b);
  }
  assignSlots();
  $("subtitle").textContent = `— ${RUN.name} · ${RUN.sm_count} SMs · ${(TMAX / 1000).toFixed(0)} ms`;

  const c = RUN.cost;
  if (c) {
    const fmt = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(1) + " ms";
    $("cost-base").textContent = fmt(c.baseline_ms);
    $("cost-prof").textContent = fmt(c.profiled_ms);
    const x = $("cost-x");
    x.textContent = c.overhead.toFixed(2) + "× slower";
    x.style.background = c.overhead >= 3 ? css("--st-crit") : c.overhead >= 1.5 ? css("--st-warn") : css("--st-good");
    x.style.color = css("--page");
    $("cost-src").textContent = `(nsys gpu-metrics; measured on ${c.source})`;
    $("costbox").style.display = "flex";
  } else {
    $("costbox").style.display = "none";
  }

  buildDie(); buildLegend();
  cur = 0; playing = false; $("play").textContent = "▶ Play";
  render(0); alignFan();
}
(async () => {
  const runs = await (await fetch("/api/runs")).json();
  $("runsel").innerHTML = runs.map(r => `<option>${r}</option>`).join("");
  $("runsel").onchange = e => loadRun(e.target.value);
  await loadRun(runs[0]);
  requestAnimationFrame(tick);
})();
