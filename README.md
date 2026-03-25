# Dubs-3-local

Custom Q3 + LoRA distillation of Qwen3.5-27B, optimized for Apple M5 Pro.

## Goal

Match the quality of **Unsloth Q6_K_XL** (the gold standard GGUF quant at 24GB) using a **Q3 mixed-precision model + LoRA error correction** that's only 12-14GB. This means:
- 40% less RAM usage
- 40% faster inference (~20+ tok/s vs ~14 tok/s)
- Significantly less heat on Apple Silicon
- Same output quality for coding, reasoning, and general tasks

## The Technique

Standard Q3 quantization loses quality vs Q6 (~0.15 perplexity gap). We close this gap using **knowledge distillation via LoRA**:

1. **Q3 base model** (`mixed_3_6`): Most layers at 3-bit, sensitive matrices (v_proj, down_proj, lm_head) at 6-bit. ~12GB, created with MLX.
2. **FP16 teacher**: The full-precision Qwen3.5-27B model generates high-quality training data.
3. **LoRA fine-tuning**: The Q3 model learns to match the teacher's outputs via a small LoRA adapter (~200MB).
4. **Fused model**: The adapter merges into the Q3 base, producing a single 12-14GB model with recovered quality.

Research basis: Apple showed mixed 2/4-bit + LoRA adapters match uncompressed model quality. We apply this to match Q6_K_XL specifically.

## Setup

### Training (CUDA - RTX 5090 or similar)

```bash
git clone https://github.com/aldenbernsteinn/Dubs-3-local.git
cd Dubs-3-local
pip install -r requirements-cuda.txt

# Train with FP16 teacher (regenerates calibration data from full model)
python train_lora_cuda.py --regenerate-data

# Or use existing calibration data from Mac
python train_lora_cuda.py
```

### Inference (Mac M5 Pro - MLX)

```bash
# One-time setup
pip install mlx-lm
./setup.sh  # raises GPU wired memory limit

# Create Q3 base model (if not already done)
python quantize.py

# Fuse LoRA adapter (after training on PC)
mlx_lm.fuse --model ~/QWEN-M5/model-q3-base \
  --adapter-path ~/QWEN-M5/lora-adapter \
  --mlx-path ~/QWEN-M5/model-q3-fused

# Serve
./serve.sh

# Benchmark
source ~/mlx-env/bin/activate
python benchmark.py
```

## M5 Pro Optimizations

All inference scripts include these Apple Silicon optimizations:
- **MLX 0.31.1** with Metal 4 TensorOps (auto Neural Accelerator dispatch on M5)
- **prefill_step_size=8192** (1.5-2x faster prompt processing vs default 512)
- **kv_bits=8, kv_group_size=64** (halves attention memory bandwidth)
- **GPU wired memory raised to 40GB** via sysctl
- **32k context window** support

## Benchmarks (M5 Pro 48GB)

| Model | Size | tok/s | RAM | Quality |
|-------|------|-------|-----|---------|
| llama.cpp Q6_K_XL (original) | 24 GB | 9.8 | 25.5 GB | Baseline |
| MLX mixed Q5+Q6 | 19 GB | 14.4 | 20.5 GB | Matches Q6 |
| MLX Q3 base (no LoRA) | 12 GB | ~20+ | ~15 GB | Lower |
| **MLX Q3 + LoRA (target)** | **~13 GB** | **~20+** | **~16 GB** | **Matches Q6** |

## Project Structure

```
calibration_data/       # Teacher model outputs for LoRA training
  train.jsonl           # 33 training examples
  valid.jsonl           # 5 validation examples
train_lora_cuda.py      # LoRA training script for CUDA GPU
requirements-cuda.txt   # PC dependencies
quantize.py             # MLX mixed-precision Q5+Q6 quantization
quantize_q3.py          # MLX mixed_3_6 Q3 quantization (via mlx_lm.convert)
generate_calibration.py # Generate teacher outputs on Mac
benchmark.py            # M5 Pro optimized benchmark
serve.sh                # MLX inference server (port 8899)
setup.sh                # One-time M5 Pro GPU memory setup
quantize.sh             # Shell wrapper for quantization
```

## Research References

- Apple: Mixed 2/4-bit + LoRA matches uncompressed quality
- EfficientQAT (ACL 2025): 2-bit QAT with <3 point accuracy loss
- LQ-LoRA (ICLR 2024): Low-rank + quantized decomposition at 2.85 BPW
- CLoQ (2025): INT2 calibrated LoRA surpasses INT4 QLoRA
- Unsloth Dynamic v2: Per-tensor sensitivity-based GGUF quantization
