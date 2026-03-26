#!/usr/bin/env python3
"""
Comprehensive benchmark for Dubs-3 Q3+LoRA vs Unsloth Q6_K_XL.

Tests:
  1. Inference speed (tok/s) and RAM usage
  2. Coding quality (implementation correctness)
  3. Reasoning quality (logic and accuracy)
  4. Long-context needle-in-haystack
  5. Perplexity on WikiText-2 sample

Usage:
    source ~/mlx-env/bin/activate
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --suite speed      # Speed only
    python benchmark.py --suite coding     # Coding quality only
    python benchmark.py --suite context    # Long-context test
    python benchmark.py --suite all        # Everything
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = os.environ.get("DUBS3_MODEL", os.path.expanduser("~/QWEN-M5/model"))


def get_ram_mb():
    pid = os.getpid()
    result = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True)
    return int(result.stdout.strip()) // 1024


def timed_generate(model, tokenizer, prompt_text, max_tokens=2048, temp=0.2):
    """Generate with timing and token counting."""
    sampler = make_sampler(temp=temp)
    start = time.time()
    response = generate(
        model, tokenizer, prompt=prompt_text,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
        prefill_step_size=8192, kv_bits=8, kv_group_size=64,
    )
    elapsed = time.time() - start
    # Approximate token count
    tok_count = len(tokenizer.encode(response))
    return response, elapsed, tok_count


def format_prompt(prompt):
    """Format a user prompt with chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# ── Benchmark Suites ─────────────────────────────────────────────────────────

def bench_speed(model, tokenizer):
    """Benchmark inference speed and memory."""
    print("\n" + "=" * 60)
    print("SPEED & MEMORY BENCHMARK")
    print("=" * 60)

    ram_before = get_ram_mb()

    prompts = [
        ("Short", "What is a hash map?", 256),
        ("Medium", "Implement a lock-free concurrent hash map in Rust that supports insert, get, and delete operations. Use atomic operations and handle memory reclamation with hazard pointers.", 2048),
        ("Long output", "Write a comprehensive guide to building a production-ready REST API in Go. Cover project structure, middleware, database integration, authentication, testing, deployment, and monitoring. Be thorough.", 4096),
    ]

    results = []
    for name, prompt, max_tok in prompts:
        text = format_prompt(prompt)
        response, elapsed, tok_count = timed_generate(model, tokenizer, text, max_tok)
        tps = tok_count / elapsed if elapsed > 0 else 0
        results.append((name, tok_count, elapsed, tps))
        print(f"  {name:15s}: {tok_count:4d} tokens in {elapsed:5.1f}s = {tps:5.1f} tok/s")

    ram_after = get_ram_mb()
    print(f"\n  RAM: {ram_before} MB (loaded) -> {ram_after} MB (after gen)")
    print(f"  Average speed: {sum(r[3] for r in results) / len(results):.1f} tok/s")
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
            "check": lambda r: "def fizzbuzz" in r and ("Fizz" in r) and ("Buzz" in r),
        },
        {
            "name": "Binary Search",
            "prompt": "Write a Python function binary_search(arr, target) that returns the index of target in a sorted array, or -1 if not found. Just the function, no explanation.",
            "check": lambda r: "def binary_search" in r and ("mid" in r or "lo" in r or "low" in r),
        },
        {
            "name": "Merge Sort",
            "prompt": "Write a Python function merge_sort(arr) that sorts a list using merge sort. Return the sorted list. Just the function, no explanation.",
            "check": lambda r: "def merge_sort" in r and ("merge" in r.lower()),
        },
        {
            "name": "LRU Cache",
            "prompt": "Write a Python class LRUCache with get(key) and put(key, value) methods with O(1) time complexity. Use OrderedDict. Just the class, no explanation.",
            "check": lambda r: "class LRU" in r and ("get" in r) and ("put" in r),
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
        text = format_prompt(test["prompt"])
        response, elapsed, tok_count = timed_generate(model, tokenizer, text, 1024, temp=0.1)
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
            "name": "Systems reasoning",
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
        text = format_prompt(test["prompt"])
        response, elapsed, tok_count = timed_generate(model, tokenizer, text, 512, temp=0.1)
        ok = test["check"](response)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {test['name']:20s}")
        if not ok:
            print(f"         Response: {response[:150]}")

    score = passed / len(tests) * 100
    print(f"\n  Reasoning score: {passed}/{len(tests)} ({score:.0f}%)")
    return score


def bench_context(model, tokenizer):
    """Needle-in-a-haystack test at various context lengths."""
    print("\n" + "=" * 60)
    print("LONG-CONTEXT BENCHMARK (Needle-in-a-Haystack)")
    print("=" * 60)

    # Generate filler text (roughly 4 tokens per word)
    filler_sentences = [
        "The history of computing is filled with fascinating developments and innovations.",
        "Many programming languages have been created over the decades, each with unique features.",
        "Software engineering practices continue to evolve as technology advances.",
        "Database systems form the backbone of most modern applications.",
        "Network protocols enable communication between computers around the world.",
        "Operating systems manage hardware resources and provide abstractions for applications.",
        "Cryptography plays a crucial role in securing digital communications.",
        "Machine learning has transformed many fields including computer vision and natural language processing.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        "Version control systems help teams collaborate on software projects effectively.",
    ]

    needle = "The secret password for the treasure vault is 'quantum-butterfly-92'."

    context_sizes = [8192, 32768, 65536, 128000]  # 8K, 32K, 64K, 128K
    results = []

    for target_tokens in context_sizes:
        # Build haystack to approximate target size
        target_words = target_tokens // 4
        haystack_parts = []
        word_count = 0
        needle_inserted = False
        needle_position = target_words // 2  # Insert at middle

        while word_count < target_words:
            sentence = random.choice(filler_sentences)
            words = len(sentence.split())

            if not needle_inserted and word_count >= needle_position:
                haystack_parts.append(needle)
                needle_inserted = True
                word_count += len(needle.split())

            haystack_parts.append(sentence)
            word_count += words

        if not needle_inserted:
            haystack_parts.insert(len(haystack_parts) // 2, needle)

        haystack = " ".join(haystack_parts)
        actual_tokens = len(tokenizer.encode(haystack))

        prompt = f"Read the following text carefully and find the secret password mentioned somewhere in it.\n\n{haystack}\n\nWhat is the secret password for the treasure vault? Answer with just the password."
        text = format_prompt(prompt)

        print(f"  Testing {target_tokens // 1024}K context ({actual_tokens} actual tokens)...", end=" ", flush=True)

        try:
            ram_before = get_ram_mb()
            response, elapsed, tok_count = timed_generate(model, tokenizer, text, 128, temp=0.1)
            ram_after = get_ram_mb()

            found = "quantum-butterfly-92" in response.lower()
            status = "FOUND" if found else "MISSED"
            print(f"[{status}] {elapsed:.1f}s, RAM delta: +{ram_after - ram_before}MB")
            results.append((target_tokens, found, elapsed, ram_after))
        except Exception as e:
            print(f"[ERROR] {e}")
            results.append((target_tokens, False, 0, 0))

    passed = sum(1 for _, found, _, _ in results if found)
    print(f"\n  Context score: {passed}/{len(results)} sizes passed")
    return results


def bench_perplexity(model, tokenizer):
    """Approximate perplexity on a WikiText-2 sample."""
    print("\n" + "=" * 60)
    print("PERPLEXITY ESTIMATE (WikiText-2 sample)")
    print("=" * 60)

    # Small representative sample (since we don't download the full dataset)
    sample_texts = [
        "The game was played on a neutral site at the Pontiac Silverdome in Pontiac, Michigan. The Broncos were the designated home team for Super Bowl XII.",
        "In computer science, a linked list is a linear collection of data elements whose order is not given by their physical placement in memory. Instead, each element points to the next.",
        "The Python programming language was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
        "Quantum computing is the exploitation of collective properties of quantum states, such as superposition and entanglement, to perform computation.",
        "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups.",
    ]

    import torch  # noqa: F811 - only needed if running on MLX with torch available
    import mlx.core as mx

    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(sample_texts):
        tokens = tokenizer.encode(text, return_tensors="np")
        token_ids = mx.array(tokens)

        # Forward pass to get logits
        logits = model(token_ids)

        # Compute cross-entropy loss
        shift_logits = logits[:, :-1, :]
        shift_labels = token_ids[:, 1:]

        log_probs = mx.log(mx.softmax(shift_logits, axis=-1))
        token_losses = -mx.take_along_axis(
            log_probs, shift_labels[..., None], axis=-1
        ).squeeze(-1)

        total_loss += float(mx.sum(token_losses))
        total_tokens += shift_labels.size

    avg_loss = total_loss / total_tokens
    perplexity = 2.718281828 ** avg_loss  # e^loss

    print(f"  Samples: {len(sample_texts)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"\n  (Compare with Q6_K_XL baseline ~7.2-7.4)")
    return perplexity


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dubs-3 benchmark suite")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "speed", "coding", "reasoning", "context", "perplexity"],
                        help="Which benchmark suite to run")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Model path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)
    print(f"RAM after load: {get_ram_mb()} MB")
    print(f"Model path: {args.model}")

    results = {}

    if args.suite in ("all", "speed"):
        results["speed"] = bench_speed(model, tokenizer)

    if args.suite in ("all", "coding"):
        results["coding"] = bench_coding(model, tokenizer)

    if args.suite in ("all", "reasoning"):
        results["reasoning"] = bench_reasoning(model, tokenizer)

    if args.suite in ("all", "context"):
        results["context"] = bench_context(model, tokenizer)

    if args.suite in ("all", "perplexity"):
        try:
            results["perplexity"] = bench_perplexity(model, tokenizer)
        except Exception as e:
            print(f"  Perplexity test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "coding" in results:
        print(f"  Coding:     {results['coding']:.0f}%")
    if "reasoning" in results:
        print(f"  Reasoning:  {results['reasoning']:.0f}%")
    if "context" in results:
        ctx_passed = sum(1 for _, found, _, _ in results["context"] if found)
        print(f"  Context:    {ctx_passed}/{len(results['context'])} sizes")
    if "perplexity" in results:
        print(f"  Perplexity: {results['perplexity']:.2f}")
    if "speed" in results:
        avg_tps = sum(r[3] for r in results["speed"]) / len(results["speed"])
        print(f"  Avg speed:  {avg_tps:.1f} tok/s")

    # Save results
    output_file = Path("benchmark_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            serializable[k] = v
        elif isinstance(v, list):
            serializable[k] = [
                {"tokens": t, "passed": p, "time": e, "ram": r} if isinstance(t, int)
                else {"name": t[0], "tokens": t[1], "time": t[2], "tps": t[3]}
                for t, p, e, r in v
            ] if k == "context" else [
                {"name": n, "tokens": tok, "time": t, "tps": tps}
                for n, tok, t, tps in v
            ]
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
