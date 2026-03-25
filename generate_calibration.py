#!/usr/bin/env python3
"""
Generate calibration data from the Q5+Q6 teacher model.
Runs diverse prompts and saves input/output pairs for LoRA distillation.
"""

import json
import os
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

TEACHER_PATH = os.path.expanduser("~/QWEN-M5/model")
OUTPUT_PATH = os.path.expanduser("~/QWEN-M5/calibration_data")

# Diverse prompts covering coding, reasoning, knowledge
PROMPTS = [
    # Coding - algorithms
    "Write a Python function that implements Dijkstra's shortest path algorithm using a priority queue. Include type hints and handle edge cases.",
    "Implement a trie data structure in Rust with insert, search, and prefix_search methods. Use proper Rust idioms.",
    "Write a Go function that performs a topological sort on a directed acyclic graph represented as an adjacency list.",
    "Implement a B-tree with order 5 in C++. Include insert and search operations with proper node splitting.",
    "Write a TypeScript class for an LRU cache with O(1) get and put operations using a doubly-linked list and hash map.",
    # Coding - systems
    "Write a Python async web scraper that respects rate limits, handles retries with exponential backoff, and saves results to SQLite.",
    "Implement a simple thread pool in Rust with a fixed number of worker threads and a job queue.",
    "Write a C function that implements a memory allocator using a free list with first-fit allocation strategy.",
    "Create a Go HTTP middleware that implements JWT authentication with token refresh.",
    "Write a Python decorator that implements memoization with TTL (time-to-live) expiry and max cache size.",
    # Coding - data structures
    "Implement a skip list in Python with insert, delete, and search operations. Include probabilistic level generation.",
    "Write a Rust implementation of a concurrent queue using atomics, supporting multiple producers and consumers.",
    "Implement a segment tree in C++ that supports range sum queries and point updates.",
    "Write a Python class for a Bloom filter with configurable false positive rate.",
    "Implement a red-black tree in Java with insert, delete, and iterator support.",
    # Reasoning
    "Explain the CAP theorem and give a concrete example of how a distributed database like Cassandra makes tradeoffs between consistency and availability.",
    "What are the tradeoffs between using microservices vs a monolith? When would you choose each? Give specific examples.",
    "Explain how garbage collection works in Go vs Rust's ownership model. Compare their approaches to memory safety.",
    "Describe the difference between optimistic and pessimistic concurrency control. When would you use each?",
    "Explain how a B+ tree index works in a database and why it's preferred over a hash index for range queries.",
    # General knowledge
    "Explain the difference between TCP and UDP. When would you use each? Give real-world examples.",
    "What is the difference between a process and a thread? Explain context switching.",
    "Explain how HTTPS/TLS handshake works step by step.",
    "What is eventual consistency and how does it differ from strong consistency?",
    "Explain the concept of backpressure in streaming systems.",
    # Math / logic
    "Prove that the square root of 2 is irrational.",
    "Explain the time complexity of quicksort in best, average, and worst cases. Why is the worst case O(n^2)?",
    "Derive the master theorem for recurrence relations and give three examples.",
    "Explain the halting problem and why it's undecidable.",
    "What is the pigeonhole principle? Give three non-trivial applications in computer science.",
    # Long-form
    "Design a URL shortening service like bit.ly. Cover the API design, database schema, hash generation strategy, and how to handle high traffic.",
    "Design a distributed message queue system. Cover partitioning, replication, ordering guarantees, and consumer group management.",
    "Write a comprehensive guide to implementing OAuth 2.0 with PKCE flow for a mobile application.",
    "Explain how a modern CPU pipeline works, covering instruction fetch, decode, execute, memory access, and write-back. Include branch prediction.",
    "Design a real-time collaborative text editor. Cover CRDTs vs OT, conflict resolution, and network architecture.",
    # Code review / debugging
    "Here's a Python function with a subtle bug. Find and fix it:\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n```",
    "What's wrong with this Rust code and how would you fix it?\n```rust\nfn longest<'a>(x: &str, y: &str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}\n```",
    "Review this SQL query for performance issues:\n```sql\nSELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > '2024-01-01'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;\n```",
]


def main():
    print(f"Loading teacher model from {TEACHER_PATH}...")
    model, tokenizer = load(TEACHER_PATH)
    sampler = make_sampler(temp=0.3)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Generate train.jsonl and valid.jsonl
    train_data = []
    valid_data = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] Generating response...")
        print(f"  Prompt: {prompt[:80]}...")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = generate(
            model, tokenizer, prompt=text,
            max_tokens=1024, sampler=sampler,
            prefill_step_size=8192, kv_bits=8, kv_group_size=64,
        )

        # Format for mlx_lm.lora training (expects "text" field)
        entry = {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"}

        if i < len(PROMPTS) - 5:
            train_data.append(entry)
        else:
            valid_data.append(entry)

    # Save
    train_path = os.path.join(OUTPUT_PATH, "train.jsonl")
    valid_path = os.path.join(OUTPUT_PATH, "valid.jsonl")

    with open(train_path, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    with open(valid_path, "w") as f:
        for entry in valid_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(valid_data)} validation examples to {valid_path}")
    print("Done!")


if __name__ == "__main__":
    main()
