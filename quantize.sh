#!/bin/bash
# Creates a mixed-precision Q5+Q6 Qwen3.5-27B model optimized for M5 Pro
# Uses research-based sensitivity rules since dynamic_quant doesn't support
# Qwen3.5's custom kernels (Gated DeltaNet).

set -e
source ~/mlx-env/bin/activate

python3 ~/QWEN-M5/quantize.py
