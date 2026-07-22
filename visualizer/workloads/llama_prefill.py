#!/usr/bin/env python3
"""Llama-3-8B prefill + short decode, trimmed for replay recording (M6a).

Based on util_examples/cupti/llm_app/run_llm.py. Model load and a warm-up
forward happen OUTSIDE the nsys capture window (cudaProfilerApi), so the
recording spans only the compute we want to replay. Prints CAPTURE_START
<epoch> so extract.py can anchor NVML wall-clock samples precisely.

Run under the recorder:
  python3 record.py llama3_prefill --capture -- python3 workloads/llama_prefill.py
"""
import json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Meta-Llama-3-8B"
PROMPTS = Path("/home/jiaxuan/Documents/Projects/util_examples/cupti/llm_app/prompts.jsonl")
BATCH, DECODE_TOKENS = 8, 4

prompts = [json.loads(l)["prompt"] for l in PROMPTS.open()][:BATCH]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
print(f"prompt batch: {tuple(enc.input_ids.shape)}", flush=True)

# warm-up outside the capture (cuBLAS heuristics, lazy inits), then let clocks idle
with torch.no_grad(), torch.cuda.nvtx.range("warmup"):
    model(input_ids=enc.input_ids[:1, :16])
torch.cuda.synchronize()
time.sleep(3)

torch.cuda.synchronize()
print(f"CAPTURE_START {time.time():.6f}", flush=True)
torch.cuda.profiler.start()
torch.cuda.synchronize()
_region_t0 = time.perf_counter()   # time COMPUTE only — not profiler.start/stop (stop blocks on nsys buffer flush)
with torch.no_grad():
    with torch.cuda.nvtx.range("prefill"):
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=True)
        torch.cuda.synchronize()
    past = out.past_key_values
    ids = out.logits[:, -1].argmax(-1, keepdim=True)
    mask = enc.attention_mask
    for i in range(DECODE_TOKENS):
        with torch.cuda.nvtx.range(f"decode_{i}"):
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            out = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            ids = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
    torch.cuda.synchronize()
_region_ms = (time.perf_counter() - _region_t0) * 1e3
torch.cuda.profiler.stop()   # after timing: this blocks on nsys's activity-buffer flush (teardown, not runtime)
print(f"REGION_MS {_region_ms:.2f}", flush=True)
print(f"done: prefill {tuple(enc.input_ids.shape)} + {DECODE_TOKENS} decode steps", flush=True)
