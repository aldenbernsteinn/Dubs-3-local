#!/usr/bin/env python3
"""
Mixed-precision Q5+Q6 quantization for Qwen3.5-27B on M5 Pro.

Uses mlx_lm.convert internals to properly save quantized model with config.
"""

import os
import json
from pathlib import Path
from mlx_lm.utils import load_model, save_model
from mlx_lm.convert import convert as mlx_convert
import mlx.nn as nn

MODEL_ID = "Qwen/Qwen3.5-27B"
OUTPUT_PATH = Path(os.path.expanduser("~/QWEN-M5/model"))
NUM_LAYERS = 64
FIRST_N = 3
LAST_N = 3


def mixed_5_6_predicate(path, module):
    """Q6 for sensitive matrices, Q5 for everything else."""
    if not isinstance(module, nn.Linear):
        return False

    layer_num = None
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_num = int(parts[i + 1])
            break

    is_first = layer_num is not None and layer_num < FIRST_N
    is_last = layer_num is not None and layer_num >= (NUM_LAYERS - LAST_N)

    # Most sensitive matrices get 6-bit
    if "k_proj" in path or "v_proj" in path:
        return {"bits": 6, "group_size": 64}
    if "down_proj" in path and (is_first or is_last):
        return {"bits": 6, "group_size": 64}
    if "o_proj" in path and is_last:
        return {"bits": 6, "group_size": 64}

    # Everything else gets 5-bit
    return {"bits": 5, "group_size": 64}


def main():
    print("Converting Qwen3.5-27B with mixed Q5+Q6 precision...")
    print("  6-bit: k_proj (all), v_proj (all), down_proj (first/last 3), o_proj (last 3)")
    print("  5-bit: everything else")

    # Use mlx_lm.convert which handles everything properly
    mlx_convert(
        hf_path=MODEL_ID,
        mlx_path=str(OUTPUT_PATH),
        quantize=True,
        q_bits=5,  # base bits
        q_group_size=64,
        quant_predicate=mixed_5_6_predicate,
    )

    print(f"\nDone! Model saved to {OUTPUT_PATH}")
    import subprocess
    result = subprocess.run(["du", "-sh", str(OUTPUT_PATH)], capture_output=True, text=True)
    print(f"Size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
