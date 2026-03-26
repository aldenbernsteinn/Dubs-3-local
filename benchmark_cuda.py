#!/usr/bin/env python3
"""
CUDA benchmark for Dubs-3 Q3+LoRA quality evaluation.

Tests coding quality, reasoning, and perplexity on CUDA hardware.
Quality is hardware-agnostic — results here predict Mac MLX quality.

Usage:
    python benchmark_cuda.py                          # Benchmark base NF4 model
    python benchmark_cuda.py --adapter ./lora-adapter  # Benchmark with LoRA adapter
    python benchmark_cuda.py --suite coding            # Coding tests only
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3.5-27B"


def load_model(adapter_path=None):
    """Load model in NF4, optionally with LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {MODEL_ID} in NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=1024, temp=0.1):
    """Generate a response."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temp, 0.01),
            do_sample=temp > 0,
            top_p=0.9,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tok_count = outputs.shape[1] - inputs["input_ids"].shape[1]
    return response, elapsed, tok_count


# ── Benchmark Suites ─────────────────────────────────────────────────────────

def bench_speed(model, tokenizer):
    """Benchmark inference speed."""
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)

    prompts = [
        ("Short", "What is a hash map?", 256),
        ("Medium", "Implement a lock-free concurrent hash map in Rust with insert, get, and delete. Use atomics.", 1024),
        ("Long", "Write a comprehensive guide to building a production REST API in Go. Cover structure, middleware, DB, auth, testing, deploy, monitoring.", 2048),
    ]

    results = []
    for name, prompt, max_tok in prompts:
        response, elapsed, tok_count = generate(model, tokenizer, prompt, max_tok)
        tps = tok_count / elapsed if elapsed > 0 else 0
        results.append((name, tok_count, elapsed, tps))
        print(f"  {name:12s}: {tok_count:4d} tokens in {elapsed:5.1f}s = {tps:5.1f} tok/s")

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"\n  VRAM used: {vram:.1f} GB")
    return results


def bench_coding(model, tokenizer):
    """Benchmark coding quality with verifiable problems."""
    print("\n" + "=" * 60)
    print("CODING QUALITY BENCHMARK")
    print("=" * 60)

    tests = [
        {
            "name": "FizzBuzz",
            "prompt": "Write a Python function fizzbuzz(n) that returns a list of strings from 1 to n. For multiples of 3 return 'Fizz', multiples of 5 return 'Buzz', multiples of both return 'FizzBuzz', otherwise the number as a string. Just the function, no explanation.",
            "check": lambda r: "def fizzbuzz" in r and "Fizz" in r and "Buzz" in r,
        },
        {
            "name": "Binary Search",
            "prompt": "Write a Python function binary_search(arr, target) that returns the index of target in a sorted array, or -1 if not found. Just the function, no explanation.",
            "check": lambda r: "def binary_search" in r and ("mid" in r or "lo" in r or "low" in r),
        },
        {
            "name": "Merge Sort",
            "prompt": "Write a Python function merge_sort(arr) that sorts a list using merge sort. Return the sorted list. Just the function, no explanation.",
            "check": lambda r: "def merge_sort" in r and "merge" in r.lower(),
        },
        {
            "name": "LRU Cache",
            "prompt": "Write a Python class LRUCache with get(key) and put(key, value) methods with O(1) time complexity. Use OrderedDict. Just the class, no explanation.",
            "check": lambda r: "class LRU" in r and "get" in r and "put" in r,
        },
        {
            "name": "Dijkstra",
            "prompt": "Write a Python function dijkstra(graph, start) where graph is a dict of {node: [(neighbor, weight), ...]}. Return a dict of shortest distances. Use heapq. Just the function, no explanation.",
            "check": lambda r: "def dijkstra" in r and "heapq" in r,
        },
        {
            "name": "Tree Traversal",
            "prompt": "Write Python functions for inorder, preorder, and postorder traversal of a binary tree. Each node has val, left, right attributes. Return lists of values. Just the functions, no explanation.",
            "check": lambda r: "inorder" in r and "preorder" in r and "postorder" in r,
        },
        {
            "name": "Async Fetch",
            "prompt": "Write a Python async function fetch_all(urls) using aiohttp that fetches multiple URLs concurrently and returns a list of response texts. Handle errors gracefully. Just the function with imports.",
            "check": lambda r: "async" in r and ("aiohttp" in r or "asyncio" in r),
        },
        {
            "name": "SQL Parser",
            "prompt": "Write a Python function parse_select(sql) that takes a simple SELECT statement string and returns a dict with keys 'columns', 'table', and 'where' (optional). Handle 'SELECT col1, col2 FROM table WHERE condition'. Just the function.",
            "check": lambda r: "def parse_select" in r and ("columns" in r or "table" in r),
        },
    ]

    passed = 0
    for test in tests:
        response, elapsed, tok_count = generate(model, tokenizer, test["prompt"], 1024, temp=0.1)
        ok = test["check"](response)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {test['name']:15s} ({tok_count} tok, {elapsed:.1f}s)")
        if not ok:
            print(f"         First 200 chars: {response[:200]}")

    score = passed / len(tests) * 100
    print(f"\n  Coding score: {passed}/{len(tests)} ({score:.0f}%)")
    return score


def bench_reasoning(model, tokenizer):
    """Benchmark reasoning quality."""
    print("\n" + "=" * 60)
    print("REASONING QUALITY BENCHMARK")
    print("=" * 60)

    tests = [
        {
            "name": "Math word problem",
            "prompt": "A store sells apples for $2 each and oranges for $3 each. If I buy 5 apples and 3 oranges, how much do I spend total? Answer with just the number.",
            "check": lambda r: "19" in r,
        },
        {
            "name": "Logic puzzle",
            "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain briefly.",
            "check": lambda r: "no" in r.lower(),
        },
        {
            "name": "Sequence",
            "prompt": "What is the next number in the sequence: 2, 6, 12, 20, 30, ? Answer with just the number.",
            "check": lambda r: "42" in r,
        },
        {
            "name": "Big-O analysis",
            "prompt": "What is the time complexity of finding an element in a balanced BST with n nodes? Answer in Big-O notation only.",
            "check": lambda r: "log" in r.lower() and "n" in r.lower(),
        },
        {
            "name": "CAP theorem",
            "prompt": "In a distributed system, if we have a network partition, can we maintain both consistency and availability according to the CAP theorem? Answer Yes or No and explain in one sentence.",
            "check": lambda r: "no" in r.lower(),
        },
        {
            "name": "Code analysis",
            "prompt": "What does this Python code print?\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))\n```\nAnswer with just the number.",
            "check": lambda r: "4" in r,
        },
    ]

    passed = 0
    for test in tests:
        response, elapsed, tok_count = generate(model, tokenizer, test["prompt"], 512, temp=0.1)
        ok = test["check"](response)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {test['name']:20s}")
        if not ok:
            print(f"         Response: {response[:150]}")

    score = passed / len(tests) * 100
    print(f"\n  Reasoning score: {passed}/{len(tests)} ({score:.0f}%)")
    return score


def bench_perplexity(model, tokenizer):
    """Measure perplexity on sample texts."""
    print("\n" + "=" * 60)
    print("PERPLEXITY ESTIMATE")
    print("=" * 60)

    samples = [
        "The game was played on a neutral site at the Pontiac Silverdome in Pontiac, Michigan. The Broncos were the designated home team for Super Bowl XII.",
        "In computer science, a linked list is a linear collection of data elements whose order is not given by their physical placement in memory. Instead, each element points to the next.",
        "The Python programming language was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
        "Quantum computing is the exploitation of collective properties of quantum states, such as superposition and entanglement, to perform computation.",
        "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups.",
    ]

    total_loss = 0.0
    total_tokens = 0

    for text in samples:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()

        n_tokens = input_ids.shape[1] - 1
        total_loss += loss * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"  Samples: {len(samples)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"\n  Target: close to Q6_K_XL baseline (~7.2-7.4)")
    return perplexity


def bench_quality_comparison(model, tokenizer):
    """Side-by-side quality test with detailed outputs."""
    print("\n" + "=" * 60)
    print("QUALITY SAMPLE OUTPUTS")
    print("=" * 60)

    prompts = [
        "Implement a Python function that checks if a binary tree is balanced. A balanced tree is one where the height of the two subtrees of any node never differs by more than one.",
        "Explain the difference between optimistic and pessimistic locking in databases. When would you use each?",
        "Write a Python function to find the longest common subsequence of two strings using dynamic programming.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt[:80]}...")
        response, elapsed, tok_count = generate(model, tokenizer, prompt, 1024, temp=0.1)
        print(f"Response ({tok_count} tok, {elapsed:.1f}s):")
        print(response[:500])
        if len(response) > 500:
            print(f"  ... [{len(response) - 500} more chars]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dubs-3 CUDA benchmark")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (omit for base model benchmark)")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "speed", "coding", "reasoning", "perplexity", "quality"])
    args = parser.parse_args()

    label = "base NF4" if not args.adapter else f"NF4 + LoRA ({args.adapter})"
    print(f"Benchmarking: {label}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model, tokenizer = load_model(args.adapter)

    results = {}

    if args.suite in ("all", "speed"):
        results["speed"] = bench_speed(model, tokenizer)

    if args.suite in ("all", "coding"):
        results["coding"] = bench_coding(model, tokenizer)

    if args.suite in ("all", "reasoning"):
        results["reasoning"] = bench_reasoning(model, tokenizer)

    if args.suite in ("all", "perplexity"):
        results["perplexity"] = bench_perplexity(model, tokenizer)

    if args.suite in ("all", "quality"):
        bench_quality_comparison(model, tokenizer)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({label})")
    print("=" * 60)
    if "coding" in results:
        print(f"  Coding:     {results['coding']:.0f}%")
    if "reasoning" in results:
        print(f"  Reasoning:  {results['reasoning']:.0f}%")
    if "perplexity" in results:
        print(f"  Perplexity: {results['perplexity']:.2f}")
    if "speed" in results:
        avg_tps = sum(r[3] for r in results["speed"]) / len(results["speed"])
        print(f"  Avg speed:  {avg_tps:.1f} tok/s")

    # Save
    output_file = f"benchmark_results_{'lora' if args.adapter else 'base'}.json"
    with open(output_file, "w") as f:
        serializable = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
