#!/bin/bash
# Start MLX server with M5 Pro optimizations
# OpenAI-compatible API on port 8899

source ~/mlx-env/bin/activate

echo "Starting Qwen3.5-27B mixed-precision server on port 8899..."
echo "Context: 32768 tokens"
echo ""

mlx_lm.server \
  --model ~/QWEN-M5/model \
  --port 8899 \
  --host 0.0.0.0
