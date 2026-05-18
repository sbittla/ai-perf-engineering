#!/usr/bin/env python3
"""
torch_profiler_trace.py  —  Project 1, Step 4: PyTorch Built-in Profiler
==========================================================================
PURPOSE:
    Use PyTorch's built-in profiler to find slow ops inside the model
    WITHOUT needing Nsight tools. The profiler records timing for every
    PyTorch operator (aten ops) and CUDA kernel, then exports:

    1. A printed table sorted by CUDA time (quick terminal read)
    2. A Chrome trace JSON (open in chrome://tracing for visual timeline)
    3. A TensorBoard trace (open with TensorBoard for interactive view)

WHY USE THIS INSTEAD OF nsys?
    torch.profiler understands PyTorch's operator graph. It can tell you
    "aten::linear took 45ms" whereas nsys only shows raw CUDA kernels.
    Use BOTH: torch.profiler to find slow ops, nsys for system-wide view.

WHAT TO LOOK FOR IN THE TABLE:
    - Ops with high "Self CUDA %" are the primary GPU time consumers
    - Ops with high "CPU total" but low "CUDA total" = CPU-bound ops
    - "aten::copy_" with large size = data transfers to/from GPU

HOW TO RUN:
    python torch_profiler_trace.py --model gpt2 --tokens 50
    
    Then open trace: 
    1. Chrome trace: open chrome://tracing → load reports/chrome_trace.json
    2. TensorBoard: tensorboard --logdir reports/tb_trace

NVTX MARKERS (bonus):
    Adding torch.cuda.nvtx.range_push/pop() labels sections so they
    appear as named bands in Nsight Systems timeline — very useful.
"""

import argparse
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="gpt2")
parser.add_argument("--tokens", type=int, default=50)
parser.add_argument("--outdir", default="reports")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

prompt = "The key challenge in machine learning is"
inputs  = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# ── WARMUP — run once outside profiler ───────────────────────────────────────
# CRITICAL: Always warm up before profiling.
# First run triggers CUDA JIT compilation (cuBLAS auto-tuning, Triton compilation).
# These one-time costs would inflate profiling results if included.
print("Warming up (1 run outside profiler)...")
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
torch.cuda.synchronize()
print("Warmup done.\n")

# ── PROFILE with torch.profiler ──────────────────────────────────────────────
print("Running profiler (3 steps)...")

# schedule controls when the profiler is active:
#   wait=1   : skip the first 1 step (additional warmup inside profiler)
#   warmup=1 : 1 step in warmup mode (profiler active but results discarded)
#   active=3 : collect data for 3 steps (these are what we analyse)
#   repeat=1 : do this cycle once
# This prevents the profiler startup overhead from contaminating results.
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

# on_trace_ready=tensorboard_trace_handler saves TensorBoard-compatible traces
tb_dir = os.path.join(args.outdir, "tb_trace")

with profile(
    activities=[
        ProfilerActivity.CPU,   # Record CPU ops: Python, C++ ATen ops
        ProfilerActivity.CUDA,  # Record CUDA kernels on the GPU
    ],
    schedule=schedule,
    on_trace_ready=tensorboard_trace_handler(tb_dir),
    record_shapes=True,    # Record tensor shapes — helps identify large ops
    profile_memory=True,   # Track memory allocation/deallocation per op
    with_stack=True,       # Include Python call stack (larger file, more context)
    with_flops=True,       # Estimate FLOPs for matmul and conv ops
) as prof:
    
    for step in range(5):  # wait(1) + warmup(1) + active(3) = 5 total
        
        # NVTX range markers: these appear as coloured bands in Nsight Systems.
        # Even without nsys, they add structure to the TensorBoard trace.
        torch.cuda.nvtx.range_push(f"step_{step}_generate")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        torch.cuda.nvtx.range_pop()
        
        # prof.step() tells the profiler to advance the schedule
        # IMPORTANT: must be called at the end of each profiling step
        prof.step()

# ── PRINT TABLE — top ops by CUDA time ───────────────────────────────────────
print("\n" + "="*70)
print("  TOP 20 OPS BY CUDA TIME")
print("="*70)

# key_averages() aggregates repeated calls to the same op
# sort_by='cuda_time_total' puts the biggest GPU time consumers first
# row_limit=20 shows top 20 rows
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# ── PRINT TABLE — top ops by CPU time ────────────────────────────────────────
print("\n" + "="*70)
print("  TOP 10 OPS BY CPU TIME (these may be blocking GPU)")
print("="*70)
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=10
))

# ── PRINT TABLE — memory allocations ────────────────────────────────────────
print("\n" + "="*70)
print("  TOP 10 OPS BY MEMORY ALLOCATION")
print("="*70)
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))

# ── EXPORT Chrome trace ───────────────────────────────────────────────────────
chrome_path = os.path.join(args.outdir, "chrome_trace.json")
prof.export_chrome_trace(chrome_path)
print(f"\n✓ Chrome trace saved: {chrome_path}")
print(f"  → Open chrome://tracing in Chrome browser")
print(f"  → Click 'Load' and select this file")
print(f"  → Zoom in on GPU rows to see kernel durations")

# ── EXPORT stacks for flamegraph ──────────────────────────────────────────────
stacks_path = os.path.join(args.outdir, "profiler_stacks.txt")
prof.export_stacks(stacks_path, metric="self_cuda_time_total")
print(f"✓ Stack traces saved: {stacks_path}")
print(f"  → Generate flamegraph: flamegraph.pl {stacks_path} > gpu_flamegraph.svg")

print(f"\n✓ TensorBoard traces saved: {tb_dir}/")
print(f"  → Run: tensorboard --logdir {tb_dir}")
print(f"  → Open: http://localhost:6006 → PyTorch Profiler tab")

# ── Quick torch.utils.bottleneck hint ────────────────────────────────────────
print("\n" + "="*70)
print("  TIP: For a quick one-command bottleneck report, run:")
print(f"  python -m torch.utils.bottleneck baseline_inference.py --model {args.model}")
print("  This combines cProfile (Python) + autograd profiler (CUDA) automatically.")
print("="*70)
