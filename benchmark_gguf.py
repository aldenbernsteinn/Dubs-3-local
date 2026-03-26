#!/usr/bin/env python3
"""
Head-to-head benchmark: Q3_K_XL vs Q6_K_XL using llama.cpp.

Compares quality, speed, and correctness on identical prompts.
This directly answers: can Q3 match Q6 quality?

Usage:
    python benchmark_gguf.py                    # Run all benchmarks
    python benchmark_gguf.py --suite coding     # Coding only
    python benchmark_gguf.py --q3-only          # Only benchmark Q3
    python benchmark_gguf.py --q6-only          # Only benchmark Q6
"""

import argparse
import json
import time
from pathlib import Path

from llama_cpp import Llama

Q3_PATH = "models/Qwen3.5-27B-UD-Q3_K_XL.gguf"
Q6_PATH = "models/Qwen3.5-27B-UD-Q6_K_XL.gguf"


def load_model(path, n_gpu_layers=-1, n_ctx=4096):
    """Load a GGUF model."""
    print(f"Loading {Path(path).name}...")
    model = Llama(
        model_path=path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    return model


def generate(model, prompt, max_tokens=1024, temp=0.1):
    """Generate using chat completion format."""
    start = time.time()
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=max(temp, 0.01),
    )
    elapsed = time.time() - start

    text = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text.split()) * 1.3)

    return text, elapsed, int(completion_tokens)


# ── Test Suites ──────────────────────────────────────────────────────────────

CODING_TESTS = [
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
        "prompt": "Write Python functions for inorder, preorder, and postorder traversal of a binary tree. Each node has val, left, right attributes. Return lists of values. Just the functions.",
        "check": lambda r: "inorder" in r and "preorder" in r and "postorder" in r,
    },
    {
        "name": "Async Fetch",
        "prompt": "Write a Python async function fetch_all(urls) using aiohttp that fetches multiple URLs concurrently and returns a list of response texts. Handle errors. Just the function with imports.",
        "check": lambda r: "async" in r and ("aiohttp" in r or "asyncio" in r),
    },
    {
        "name": "Decorator",
        "prompt": "Write a Python decorator @retry(max_attempts=3, delay=1.0) that retries a function on exception with exponential backoff. Just the decorator, no explanation.",
        "check": lambda r: "def retry" in r and ("attempt" in r or "tries" in r or "except" in r),
    },
]

REASONING_TESTS = [
    {
        "name": "Math",
        "prompt": "A store sells apples for $2 each and oranges for $3 each. If I buy 5 apples and 3 oranges, how much do I spend total? Answer with just the number.",
        "check": lambda r: "19" in r,
    },
    {
        "name": "Logic",
        "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain briefly.",
        "check": lambda r: "no" in r.lower()[:50],
    },
    {
        "name": "Sequence",
        "prompt": "What is the next number in the sequence: 2, 6, 12, 20, 30, ? Answer with just the number.",
        "check": lambda r: "42" in r,
    },
    {
        "name": "Big-O",
        "prompt": "What is the time complexity of finding an element in a balanced BST with n nodes? Answer in Big-O notation only.",
        "check": lambda r: "log" in r.lower() and "n" in r.lower(),
    },
    {
        "name": "CAP",
        "prompt": "In a distributed system, if we have a network partition, can we maintain both consistency and availability according to the CAP theorem? Answer Yes or No in one sentence.",
        "check": lambda r: "no" in r.lower()[:50],
    },
    {
        "name": "Python ref",
        "prompt": "What does this code print?\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))\n```\nAnswer with just the number.",
        "check": lambda r: "4" in r,
    },
]

QUALITY_PROMPTS = [
    "Implement a Python function that checks if a binary tree is balanced. A balanced tree is one where the height of two subtrees never differs by more than one.",
    "Explain the difference between optimistic and pessimistic locking in databases. When would you use each?",
    "Write a Python function to find the longest common subsequence of two strings using dynamic programming.",
]


def run_test_suite(model, model_name, tests, suite_name):
    """Run a test suite and return results."""
    print(f"\n{'=' * 60}")
    print(f"{suite_name} - {model_name}")
    print(f"{'=' * 60}")

    passed = 0
    total_tokens = 0
    total_time = 0

    for test in tests:
        response, elapsed, tok_count = generate(model, test["prompt"], 1024, temp=0.1)
        ok = test["check"](response)
        passed += int(ok)
        total_tokens += tok_count
        total_time += elapsed

        status = "PASS" if ok else "FAIL"
        tps = tok_count / elapsed if elapsed > 0 else 0
        print(f"  [{status}] {test['name']:15s} ({tok_count:3d} tok, {tps:.1f} tok/s)")
        if not ok:
            # Show first 150 chars of response for debugging
            print(f"         -> {response[:150]}")

    score = passed / len(tests) * 100
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"\n  Score: {passed}/{len(tests)} ({score:.0f}%)")
    print(f"  Avg speed: {avg_tps:.1f} tok/s")

    return {"score": score, "passed": passed, "total": len(tests), "avg_tps": avg_tps}


def run_quality_samples(model, model_name):
    """Generate sample outputs for qualitative comparison."""
    print(f"\n{'=' * 60}")
    print(f"QUALITY SAMPLES - {model_name}")
    print(f"{'=' * 60}")

    outputs = []
    for i, prompt in enumerate(QUALITY_PROMPTS):
        response, elapsed, tok_count = generate(model, prompt, 1024, temp=0.1)
        tps = tok_count / elapsed if elapsed > 0 else 0
        print(f"\n--- Sample {i+1} ({tok_count} tok, {tps:.1f} tok/s) ---")
        print(f"Q: {prompt[:80]}...")
        print(f"A: {response[:400]}")
        if len(response) > 400:
            print(f"   ... [{len(response) - 400} more chars]")
        outputs.append(response)

    return outputs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q3 vs Q6 head-to-head benchmark")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "coding", "reasoning", "quality"])
    parser.add_argument("--q3-only", action="store_true", help="Only benchmark Q3")
    parser.add_argument("--q6-only", action="store_true", help="Only benchmark Q6")
    parser.add_argument("--q3-path", type=str, default=Q3_PATH)
    parser.add_argument("--q6-path", type=str, default=Q6_PATH)
    parser.add_argument("--n-gpu", type=int, default=-1, help="GPU layers (-1 = all)")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    args = parser.parse_args()

    models_to_test = []
    if not args.q6_only:
        models_to_test.append(("Q3_K_XL", args.q3_path))
    if not args.q3_only:
        models_to_test.append(("Q6_K_XL", args.q6_path))

    all_results = {}

    for model_name, model_path in models_to_test:
        if not Path(model_path).exists():
            print(f"SKIP {model_name}: {model_path} not found")
            continue

        model = load_model(model_path, n_gpu_layers=args.n_gpu, n_ctx=args.ctx)
        results = {}

        if args.suite in ("all", "coding"):
            results["coding"] = run_test_suite(model, model_name, CODING_TESTS, "CODING")

        if args.suite in ("all", "reasoning"):
            results["reasoning"] = run_test_suite(model, model_name, REASONING_TESTS, "REASONING")

        if args.suite in ("all", "quality"):
            results["quality_samples"] = run_quality_samples(model, model_name)

        all_results[model_name] = results

        # Free model memory before loading next
        del model

    # ── Comparison Summary ────────────────────────────────────────────────
    if len(all_results) >= 2:
        print("\n" + "=" * 60)
        print("HEAD-TO-HEAD COMPARISON: Q3_K_XL vs Q6_K_XL")
        print("=" * 60)

        for suite in ["coding", "reasoning"]:
            if suite in all_results.get("Q3_K_XL", {}) and suite in all_results.get("Q6_K_XL", {}):
                q3 = all_results["Q3_K_XL"][suite]
                q6 = all_results["Q6_K_XL"][suite]
                gap = q3["score"] - q6["score"]
                speed_ratio = q3["avg_tps"] / q6["avg_tps"] if q6["avg_tps"] > 0 else 0

                print(f"\n  {suite.upper()}:")
                print(f"    Q3_K_XL: {q3['passed']}/{q3['total']} ({q3['score']:.0f}%) @ {q3['avg_tps']:.1f} tok/s")
                print(f"    Q6_K_XL: {q6['passed']}/{q6['total']} ({q6['score']:.0f}%) @ {q6['avg_tps']:.1f} tok/s")
                print(f"    Gap: {gap:+.0f}% quality, {speed_ratio:.1f}x speed")

        print(f"\n  Q3_K_XL is {Path(args.q3_path).stat().st_size/1024**3:.1f}GB vs Q6_K_XL at {Path(args.q6_path).stat().st_size/1024**3:.1f}GB")

    elif len(all_results) == 1:
        name = list(all_results.keys())[0]
        r = all_results[name]
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {name}")
        print(f"{'=' * 60}")
        for suite, data in r.items():
            if isinstance(data, dict) and "score" in data:
                print(f"  {suite}: {data['passed']}/{data['total']} ({data['score']:.0f}%) @ {data['avg_tps']:.1f} tok/s")

    # Save results
    output_file = "benchmark_q3_vs_q6.json"
    serializable = {}
    for name, r in all_results.items():
        serializable[name] = {k: v for k, v in r.items() if isinstance(v, dict)}
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
