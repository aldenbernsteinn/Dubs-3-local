#!/usr/bin/env python3
"""Benchmark Qwen3.5-27B mixed-precision on M5 Pro with all optimizations."""

import time
import os
import subprocess
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = os.path.expanduser("~/QWEN-M5/model")

def get_ram_mb():
    pid = os.getpid()
    result = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True)
    return int(result.stdout.strip()) // 1024

def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    print(f"RAM after load: {get_ram_mb()} MB")

    prompt = (
        "Implement a lock-free concurrent hash map in Rust that supports "
        "insert, get, and delete operations. Use atomic operations and handle "
        "memory reclamation with hazard pointers. Provide the full implementation "
        "with proper error handling."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.2)

    print("Generating with M5 Pro optimizations...")
    print("  prefill_step_size=8192, kv_bits=8, kv_group_size=64")
    print()

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=2048,
        sampler=sampler,
        verbose=True,
        prefill_step_size=8192,
        kv_bits=8,
        kv_group_size=64,
    )
    elapsed = time.time() - start

    print(f"\nRAM after gen: {get_ram_mb()} MB")
    print(f"Wall time: {elapsed:.1f}s")
    print("---RESPONSE---")
    print(response)

if __name__ == "__main__":
    main()
