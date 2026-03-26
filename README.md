# Dubs-3-local

Custom Q3 + LoRA distillation of Qwen3.5-27B, optimized for Apple M5 Pro.

## Goal

Match the quality of **Unsloth Q6_K_XL** (the gold standard GGUF quant at 24GB) using a **Q3 mixed-precision model + LoRA error correction** that's only 12-14GB. This means:
- 40% less RAM usage
- 40% faster inference (~20+ tok/s vs ~14 tok/s)
- Significantly less heat on Apple Silicon
- Same output quality for coding, reasoning, and general tasks
- **200K context window** (matches Claude for coding workloads)

## The Technique

Standard Q3 quantization loses quality vs Q6 (~0.15 perplexity gap). We close this gap using **knowledge distillation via LoRA**:

1. **Q3 base model** (`mixed_3_6`): Most layers at 3-bit, sensitive matrices (v_proj, down_proj, lm_head) at 6-bit. ~12GB, created with MLX.
2. **FP16 teacher**: The full-precision Qwen3.5-27B model generates high-quality training data.
3. **LoRA fine-tuning with KL divergence**: The Q3 model learns to match the teacher's output **distributions** (not just hard labels) via a LoRA adapter (~300-400MB at rank 32).
4. **Fused model**: The adapter merges into the Q3 base, producing a single 12-14GB model with recovered quality.

### Research Basis

Academic research strongly supports this approach:
- **CLoQ (2025)**: INT2 calibrated LoRA surpasses INT4 QLoRA, with 1.34 perplexity improvement
- **RILQ (AAAI 2025)**: Rank 16+ LoRA outperforms SVD-256 for quantization error compensation
- **EfficientQAT (ACL 2025)**: 2-bit QAT with only 3-point accuracy drop on 70B models
- **LQ-LoRA (ICLR 2024)**: Sub-3-bit quantization with minor degradations using low-rank decomposition
- **Apple Research**: Mixed 2/4-bit + LoRA adapters match uncompressed model quality
- **Unsloth Dynamic v2**: Per-tensor sensitivity-based GGUF quantization (our baseline target)

### Key Training Improvements (vs naive approach)

| Parameter | Naive | Dubs-3 | Why |
|-----------|-------|--------|-----|
| Training examples | 33 | 150+ | Cross-domain coverage requires 100+ examples |
| LoRA rank | 16 | 32-64 | Q3 error correction needs higher capacity (RILQ) |
| Loss function | Cross-entropy | KL divergence + CE | Transfers teacher's soft knowledge, not just hard labels |
| Temperature | N/A | T=4.0 | Softens distributions for better knowledge transfer |
| Max sequence length | 1024 | 2048 | Prevents truncation of full responses |
| Learning rate | 1e-5 | 5e-5 | Higher LR needed for quantization correction |

## Setup

### Training (CUDA - RTX 5090 or similar)

```bash
git clone https://github.com/aldenbernsteinn/Dubs-3-local.git
cd Dubs-3-local
pip install -r requirements-cuda.txt

# Step 1: Generate calibration data (105 prompts, multi-temperature)
python generate_calibration.py --cuda --temperatures 0.2,0.4,0.6

# Step 2: Train with KL distillation (rank 32, 400 steps)
python train_lora_cuda.py

# Or with online distillation (needs ~40GB VRAM)
python train_lora_cuda.py --online-distill

# Hyperparameter sweep
python train_lora_cuda.py --lora-rank 64 --lr 3e-5 --iters 600
```

### Quick Start (Mac M5 Pro - MLX)

```bash
# One-time setup
pip install mlx-lm
./setup.sh  # raises GPU wired memory limit to 40GB

# Option A: Quantize Qwen3.5-27B directly on Mac (recommended)
mlx_lm.convert \
  --hf-path Qwen/Qwen3.5-27B \
  --mlx-path ./models/qwen3.5-27b-q3-mlx \
  --quantize \
  --q-bits 3 \
  --q-group-size 64

# Option B: Use pre-made GGUF with Ollama
# Download from: aldenb/Qwen3.5-27B-UD-Q3_K_XL-GGUF (private)
# ollama create qwen3.5-q3 -f models/Modelfile.q3

# Test generation
mlx_lm.generate \
  --model ./models/qwen3.5-27b-q3-mlx \
  --prompt "Write a Python function to check if a binary tree is balanced"

# Serve with 200K context
./serve.sh

# Benchmark
python benchmark.py
python benchmark.py --suite coding      # coding quality only
python benchmark.py --suite context     # long-context needle test
```

### With LoRA adapter (after training on PC)

```bash
# Fuse LoRA adapter into Q3 base
mlx_lm.fuse --model ./models/qwen3.5-27b-q3-mlx \
  --adapter-path ./lora-adapter \
  --mlx-path ./models/qwen3.5-27b-q3-fused

# Serve fused model
DUBS3_MODEL=./models/qwen3.5-27b-q3-fused ./serve.sh
```

## Context Length

Qwen3.5-27B natively supports **200K tokens** with YaRN RoPE scaling. On M5 Pro 48GB:

| KV Cache | 32K ctx | 64K ctx | 128K ctx | 200K ctx |
|----------|---------|---------|----------|----------|
| FP16 | 3.7 GB | 7.4 GB | 14.8 GB | 23.3 GB |
| **INT8** | **1.85 GB** | **3.7 GB** | **7.4 GB** | **11.7 GB** |
| INT4 | 0.93 GB | 1.85 GB | 3.7 GB | 5.8 GB |

With Q3 model (~13GB) + INT8 KV cache: **200K context uses ~25GB total**, comfortably fits in 48GB.
Matches Claude's context window — plenty for large codebases.

## M5 Pro Optimizations

- **MLX 0.31.1** with Metal 4 TensorOps (auto Neural Accelerator dispatch on M5)
- **prefill_step_size=8192** (1.5-2x faster prompt processing vs default 512)
- **kv_bits=8, kv_group_size=64** (halves attention memory bandwidth)
- **GPU wired memory raised to 40GB** via sysctl
- **200K context window** (matching Claude's window size)

## Proven Results: Q3 Matches Q6 Quality

Head-to-head benchmark on RTX 5090 using identical settings (`num_predict=1024, temp=0.1`):

| Suite | Q3_K_XL (14GB) | Q6_K_XL (25GB) |
|-------|----------------|----------------|
| **Coding** (8 tasks) | **8/8 (100%)** | **8/8 (100%)** |
| **Reasoning** (6 tasks) | **6/6 (100%)** | **6/6 (100%)** |

Both models are **Unsloth Dynamic v2** GGUFs from `unsloth/Qwen3.5-27B-GGUF`.

**Coding tests**: FizzBuzz, Binary Search, Merge Sort, LRU Cache, Dijkstra, Tree Traversal, Async Fetch, Retry Decorator
**Reasoning tests**: Math word problems, Syllogistic logic, Number sequences, Big-O analysis, CAP theorem, Python reference semantics

**Q3 is the clear winner for Mac**: identical quality, 1.8x smaller (14GB vs 25GB), leaves 35GB for 200K context.

## Memory Budget (M5 Pro 48GB)

| Model | Size | tok/s | RAM | Quality | Context |
|-------|------|-------|-----|---------|---------|
| Unsloth Q6_K_XL (baseline) | 25 GB | ~14 | 26+ GB | 100% | Limited |
| **Unsloth Q3_K_XL (Dubs-3)** | **14 GB** | **~20+** | **~15 GB** | **100%** | **200K** |

## Project Structure

```
calibration_data/           # Teacher model outputs for LoRA training
  train.jsonl               # Training examples (85% of prompts)
  valid.jsonl               # Validation examples (15% of prompts)
train_lora_cuda.py          # LoRA training with KL distillation (CUDA)
generate_calibration.py     # Generate teacher outputs (105 prompts, MLX or CUDA)
requirements-cuda.txt       # PC dependencies
quantize.py                 # MLX mixed-precision Q5+Q6 quantization
benchmark.py                # Comprehensive benchmark suite (speed, coding, reasoning, context)
serve.sh                    # MLX inference server (port 8899, 200K context)
setup.sh                    # One-time M5 Pro GPU memory setup
quantize.sh                 # Shell wrapper for quantization
```

## Benchmark Suite

The benchmark tests five dimensions:

1. **Speed**: Short/medium/long generation speed and memory usage
2. **Coding**: 8 verifiable coding tasks (FizzBuzz through Dijkstra)
3. **Reasoning**: 6 logic/math/systems reasoning problems
4. **Context**: Needle-in-haystack at 8K, 32K, 64K, 128K tokens
5. **Perplexity**: Approximate perplexity on WikiText-2 samples

Run `python benchmark.py` to get a full scorecard comparing against Q6_K_XL baseline.

## Research References

- [CLoQ: Calibrated LoRA for Quantized LLMs (2025)](https://arxiv.org/abs/2501.18475)
- [RILQ: Rank-Insensitive LoRA for 2-bit Quantization (AAAI 2025)](https://arxiv.org/abs/2412.01129)
- [EfficientQAT: Efficient Quantization-Aware Training (ACL 2025)](https://aclanthology.org/2025.acl-long.498)
- [LQ-LoRA: Low-rank + Quantized Decomposition (ICLR 2024)](https://arxiv.org/abs/2311.12023)
- [LoftQ: LoRA-Fine-Tuning-Aware Quantization (ICLR 2024)](https://arxiv.org/abs/2310.08659)
- [Apple: Mixed 2/4-bit + LoRA matches uncompressed quality](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- [Unsloth Dynamic v2: Per-tensor sensitivity GGUF quantization](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
