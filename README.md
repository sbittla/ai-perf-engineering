# AI Systems Performance Engineering

**A hands-on learning repository for GPU profiling, LLM inference optimisation, and Linux systems performance.**

Aligned with the OpenAI Workload Porting & Performance Engineer role profile.

---

## Who This Is For

Engineers transitioning from **Enterprise / Cloud Performance Engineering** into **AI Systems & Infrastructure Performance Engineering**. You already know distributed systems, observability, and performance methodology. This repository adds GPU systems depth.

## What You Will Be Able to Do After Completion

- Profile GPU workloads with `nsys`, `ncu`, and `torch.profiler`
- Diagnose CPU↔GPU bottlenecks using `perf`, `eBPF`, and flamegraphs
- Optimise LLM inference: batch size, KV cache, quantization, `torch.compile`
- Benchmark AI workloads end-to-end with repeatable methodology
- Understand and navigate distributed inference systems (NCCL, FSDP, tensor parallelism)
- Discuss hardware/software tradeoffs like a systems performance engineer

---

## Repository Structure

```
ai-perf-engineering/
│
├── 00_pytorch_basics/          ← Start here if new to PyTorch
│   ├── exercises/              ← Fill in the TODOs
│   └── solutions/              ← Reference answers
│
├── 01_phase1_foundations/      ← CPU architecture & Linux internals
│   ├── cpu_arch/
│   └── linux_perf/
│
├── 02_phase2_gpu/              ← GPU programming & CUDA profiling (most critical)
│   ├── module3_cuda_fundamentals/
│   ├── module4_profiling/
│   └── module5_pytorch_perf/
│
├── 03_phase3_systems/          ← Linux low-level profiling
│   ├── module6_linux_profiling/
│   └── module7_memory_numa/
│
├── 04_phase4_inference/        ← LLM inference systems
│   ├── module8_llm_systems/
│   └── module9_distributed/
│
├── 05_phase5_workload/         ← Workload porting & benchmarking
│   ├── module10_porting/
│   └── module11_benchmarking/
│
├── 06_capstone_projects/       ← End-to-end portfolio projects
│   ├── project1_llm_opt/       ← LLM Inference Optimisation Lab
│   ├── project2_dataloader/    ← DataLoader I/O Bottleneck Hunt
│   ├── project3_kv_cache/      ← KV Cache Memory Pressure Experiment
│   └── project4_flamegraph/    ← CPU-to-GPU Pipeline Flamegraph Challenge
│
├── shared/
│   ├── models/model.py         ← TinyTransformer used across all scripts
│   └── utils/results_table.py  ← Before/after comparison table printer
│
├── scripts/
│   └── setup_env.sh            ← Install all dependencies
│
└── docs/
    ├── ROADMAP.md              ← 6-phase learning plan with timelines
    ├── COMMANDS.md             ← Every profiling command in one place
    ├── INTERVIEW_PREP.md       ← Questions to answer before interviewing
    └── HARDWARE_SETUP.md       ← RTX 4060 environment setup guide
```

---

## Getting Started

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/ai-perf-engineering.git
cd ai-perf-engineering

# Install all dependencies (creates ~/capstone_venv)
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh

source ~/capstone_venv/bin/activate
```

### 2. PyTorch Basics (if new to PyTorch — 2–3 days)

```bash
cd 00_pytorch_basics/exercises

python exercise_01_tensors.py           # tensors, shapes, devices
python exercise_02_autograd.py          # gradients, backward, no_grad
python exercise_03_nn_modules.py        # building models
python exercise_04_training_loop.py     # complete training loop + AMP
python exercise_05_performance_basics.py # CUDA timing, profiler, memory
```

Each exercise has `TODO` blocks with hints at the bottom.

### 3. Phase 2 — GPU Profiling (most important phase)

```bash
cd 02_phase2_gpu/module3_cuda_fundamentals

# Learn the GPU execution model
python vector_add.py
python occupancy_experiment.py
python matmul_bench.py          # builds your roofline intuition

# Profile a training run
cd ../module4_profiling
nsys profile --stats=true python train.py --task lm --steps 50
python profile_pytorch_infer.py # all profiling tools in one script

# Precision & optimisation
cd ../module5_pytorch_perf
python fp16_bf16_bench.py
python llama_infer_optimize.py  # 5-step optimisation ladder
```

### 4. Capstone Projects (portfolio artifacts)

```bash
# Project 1: LLM Inference Optimisation
cd 06_capstone_projects/project1_llm_opt
python baseline_inference.py --model gpt2
bash profile_nsys.sh gpt2
python torch_compile_bench.py --model gpt2

# Project 2: DataLoader I/O Bottleneck
cd ../project2_dataloader
python slow_dataloader.py &
bash diagnose_io.sh            # watch it in another terminal
python fast_dataloader.py      # compare results
```

---

## Learning Path

| Phase | Duration | Key Outcome |
|-------|----------|-------------|
| **00 PyTorch Basics** | 2–3 days | Write & time any PyTorch code |
| **Phase 1** Foundation | 3 weeks | Hardware + Linux mental model |
| **Phase 2** GPU Profiling | 5 weeks | Profile any GPU workload |
| **Phase 3** Systems | 4 weeks | Flamegraph, perf, eBPF fluency |
| **Phase 4** Inference | 4 weeks | LLM serving + distributed systems |
| **Phase 5** Benchmarking | 4 weeks | Characterise and compare workloads |
| **Capstone** Projects | 4 weeks | Portfolio-ready artifacts |

**Total: ~5–6 months part-time**

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed plan.

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 4060 (8GB) or equivalent CUDA-capable GPU
- **OS:** Ubuntu 22.04 (recommended) or WSL2 on Windows
- **RAM:** 16GB+ recommended
- **Storage:** 50GB+ for models and datasets
- **CUDA:** 12.x (installed by `setup_env.sh`)

See [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) for detailed setup.

---

## Key Tools Reference

| Tool | Purpose | Install |
|------|---------|---------|
| `nsys` | System-wide GPU timeline | CUDA Toolkit |
| `ncu` | Per-kernel hardware counters | CUDA Toolkit |
| `torch.profiler` | PyTorch op-level timing | `pip install torch` |
| `py-spy` | Python CPU flamegraphs | `pip install py-spy` |
| `perf` | Linux hardware counters | `apt install linux-tools-generic` |
| `bpftrace` | eBPF kernel tracing | `apt install bpftrace` |
| `nvitop` | Rich GPU process monitor | `pip install nvitop` |
| `vLLM` | High-throughput LLM serving | `pip install vllm` |

See [`docs/COMMANDS.md`](docs/COMMANDS.md) for all commands in one place.

---

## Contributing

This is a personal learning repository. If you find bugs or have improvements, PRs are welcome.

## License

MIT
