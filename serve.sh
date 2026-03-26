#!/bin/bash
# Start MLX server with M5 Pro optimizations
# OpenAI-compatible API on port 8899
# Supports up to 200K context with INT8 KV cache (~12GB on 48GB M5 Pro)

source ~/mlx-env/bin/activate

MODEL_PATH="${DUBS3_MODEL:-$HOME/QWEN-M5/model}"
CONTEXT_LENGTH="${DUBS3_CONTEXT:-200000}"
PORT="${DUBS3_PORT:-8899}"

echo "Starting Qwen3.5-27B Q3+LoRA server on port $PORT..."
echo "  Model: $MODEL_PATH"
echo "  Context: $CONTEXT_LENGTH tokens ($(echo "$CONTEXT_LENGTH / 1024" | bc)K)"
echo "  KV cache: INT8 (kv_bits=8, kv_group_size=64)"
echo "  Prefill: 8192 tokens/step"
echo ""

mlx_lm.server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --context-length "$CONTEXT_LENGTH"
