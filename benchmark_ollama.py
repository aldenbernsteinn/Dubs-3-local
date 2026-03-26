#!/usr/bin/env python3
"""
Head-to-head benchmark: Q3_K_XL vs Q6 (qwen3.5-opencode) via Ollama.

Directly answers: can Q3 match Q6 quality on coding, reasoning, and general tasks?

Usage:
    python benchmark_ollama.py                     # Full comparison
    python benchmark_ollama.py --suite coding      # Coding only
    python benchmark_ollama.py --q3-only           # Only Q3
"""

import argparse
import json
import time

import ollama

Q3_MODEL = "qwen3.5-q3"
Q6_MODEL = "qwen3.5-q6"


def strip_think(text):
    """Strip <think>...</think> blocks from Qwen3.5 responses."""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def generate(model_name, prompt, max_tokens=1024, temp=0.1):
    """Generate a response via Ollama."""
    start = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp, "num_predict": max_tokens, "num_ctx": 2048},
    )
    elapsed = time.time() - start

    text = response["message"]["content"]
    eval_count = response.get("eval_count") or len(text.split())
    eval_duration = response.get("eval_duration") or 1
    tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    return text, elapsed, eval_count, tps


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
        "check": lambda r: "no" in r.lower()[:100],
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
        "prompt": "In a distributed system with a network partition, can we maintain both consistency and availability per the CAP theorem? Answer Yes or No in one sentence.",
        "check": lambda r: "no" in r.lower()[:100],
    },
    {
        "name": "Python ref",
        "prompt": "What does this code print?\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))\n```\nAnswer with just the number.",
        "check": lambda r: "4" in r,
    },
]


def run_suite(model_name, tests, suite_name):
    """Run a benchmark suite on a model."""
    print(f"\n{'=' * 60}")
    print(f"{suite_name} - {model_name}")
    print("=" * 60)

    passed = 0
    total_tps = []

    for test in tests:
        text, elapsed, tok_count, tps = generate(model_name, test["prompt"], 1024, 0.1)
        clean_text = strip_think(text)
        ok = test["check"](clean_text)
        passed += int(ok)
        total_tps.append(tps)

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {test['name']:15s} ({tok_count:3d} tok, {tps:.1f} tok/s, {elapsed:.1f}s)")
        if not ok:
            print(f"         -> {text[:150]}")

    score = passed / len(tests) * 100
    avg_tps = sum(total_tps) / len(total_tps) if total_tps else 0
    print(f"\n  Score: {passed}/{len(tests)} ({score:.0f}%)")
    print(f"  Avg generation speed: {avg_tps:.1f} tok/s")

    return {"score": score, "passed": passed, "total": len(tests), "avg_tps": avg_tps}


def run_quality_samples(model_name):
    """Generate detailed outputs for qualitative comparison."""
    print(f"\n{'=' * 60}")
    print(f"QUALITY SAMPLES - {model_name}")
    print("=" * 60)

    prompts = [
        "Implement a Python function that checks if a binary tree is balanced (heights differ by at most 1).",
        "Explain optimistic vs pessimistic locking. When would you use each?",
        "Write a Python LCS function using dynamic programming.",
    ]

    outputs = []
    for i, prompt in enumerate(prompts):
        text, elapsed, tok_count, tps = generate(model_name, prompt, 1024, 0.1)
        print(f"\n--- Sample {i+1} ({tok_count} tok, {tps:.1f} tok/s) ---")
        print(f"Q: {prompt}")
        print(f"A: {text[:500]}")
        if len(text) > 500:
            print(f"   ... [{len(text) - 500} more chars]")
        outputs.append(text)

    return outputs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q3 vs Q6 Ollama benchmark")
    parser.add_argument("--suite", default="all", choices=["all", "coding", "reasoning", "quality"])
    parser.add_argument("--q3-only", action="store_true")
    parser.add_argument("--q6-only", action="store_true")
    parser.add_argument("--q3-model", default=Q3_MODEL)
    parser.add_argument("--q6-model", default=Q6_MODEL)
    args = parser.parse_args()

    models = []
    if not args.q6_only:
        models.append(("Q3_K_XL", args.q3_model))
    if not args.q3_only:
        models.append(("Q6_K_XL", args.q6_model))

    all_results = {}

    for label, model_name in models:
        results = {}

        if args.suite in ("all", "coding"):
            results["coding"] = run_suite(model_name, CODING_TESTS, "CODING")

        if args.suite in ("all", "reasoning"):
            results["reasoning"] = run_suite(model_name, REASONING_TESTS, "REASONING")

        if args.suite in ("all", "quality"):
            results["quality"] = run_quality_samples(model_name)

        all_results[label] = results

    # ── Comparison ────────────────────────────────────────────────────────
    if len(all_results) >= 2:
        print("\n" + "=" * 60)
        print("HEAD-TO-HEAD: Q3_K_XL vs Q6_K_XL")
        print("=" * 60)

        for suite in ["coding", "reasoning"]:
            q3_key = [k for k in all_results if "Q3" in k][0]
            q6_key = [k for k in all_results if "Q6" in k][0]

            if suite in all_results[q3_key] and suite in all_results[q6_key]:
                q3 = all_results[q3_key][suite]
                q6 = all_results[q6_key][suite]
                gap = q3["score"] - q6["score"]
                speed = q3["avg_tps"] / q6["avg_tps"] if q6["avg_tps"] > 0 else 0

                print(f"\n  {suite.upper()}:")
                print(f"    Q3_K_XL:  {q3['passed']}/{q3['total']} ({q3['score']:.0f}%) @ {q3['avg_tps']:.1f} tok/s")
                print(f"    Q6 (ref): {q6['passed']}/{q6['total']} ({q6['score']:.0f}%) @ {q6['avg_tps']:.1f} tok/s")
                print(f"    Quality gap: {gap:+.0f}%  |  Speed: {speed:.2f}x")

    elif len(all_results) == 1:
        label = list(all_results.keys())[0]
        r = all_results[label]
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {label}")
        print("=" * 60)
        for s, d in r.items():
            if isinstance(d, dict) and "score" in d:
                print(f"  {s}: {d['passed']}/{d['total']} ({d['score']:.0f}%) @ {d['avg_tps']:.1f} tok/s")

    # Save
    output = "benchmark_q3_vs_q6.json"
    save = {k: {s: v for s, v in r.items() if isinstance(v, dict)} for k, r in all_results.items()}
    with open(output, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved to {output}")


if __name__ == "__main__":
    main()
