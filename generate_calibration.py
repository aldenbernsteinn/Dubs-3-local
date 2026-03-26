#!/usr/bin/env python3
"""
Generate calibration data from the teacher model for LoRA distillation.
Runs diverse prompts at multiple temperatures and saves input/output pairs.

Supports both MLX (Mac) and CUDA (PC) backends:
    Mac:  python generate_calibration.py
    CUDA: python generate_calibration.py --cuda
"""

import argparse
import json
import os
import random
from pathlib import Path

# ── Prompt bank: 105 diverse prompts ──────────────────────────────────────────

PROMPTS = [
    # ═══ CODING: Algorithms (15) ═══
    "Write a Python function that implements Dijkstra's shortest path algorithm using a priority queue. Include type hints and handle edge cases.",
    "Implement a trie data structure in Rust with insert, search, and prefix_search methods. Use proper Rust idioms.",
    "Write a Go function that performs a topological sort on a directed acyclic graph represented as an adjacency list.",
    "Implement a B-tree with order 5 in C++. Include insert and search operations with proper node splitting.",
    "Write a TypeScript class for an LRU cache with O(1) get and put operations using a doubly-linked list and hash map.",
    "Implement a skip list in Python with insert, delete, and search operations. Include probabilistic level generation.",
    "Write a Rust implementation of a concurrent queue using atomics, supporting multiple producers and consumers.",
    "Implement a segment tree in C++ that supports range sum queries and point updates.",
    "Write a Python class for a Bloom filter with configurable false positive rate.",
    "Implement a red-black tree in Java with insert, delete, and iterator support.",
    "Write a Python function for A* pathfinding on a 2D grid with diagonal movement and weighted terrain costs.",
    "Implement a Fenwick tree (Binary Indexed Tree) in C++ supporting range updates and point queries.",
    "Write a Go implementation of consistent hashing with virtual nodes for distributed key-value storage.",
    "Implement a suffix array with LCP array construction in Python for efficient substring search.",
    "Write a Rust function that solves the N-Queens problem using backtracking with pruning optimizations.",

    # ═══ CODING: Systems Programming (15) ═══
    "Write a Python async web scraper that respects rate limits, handles retries with exponential backoff, and saves results to SQLite.",
    "Implement a simple thread pool in Rust with a fixed number of worker threads and a job queue.",
    "Write a C function that implements a memory allocator using a free list with first-fit allocation strategy.",
    "Create a Go HTTP middleware that implements JWT authentication with token refresh.",
    "Write a Python decorator that implements memoization with TTL (time-to-live) expiry and max cache size.",
    "Implement a lock-free concurrent hash map in Rust using atomic operations. Handle memory reclamation with epoch-based reclamation.",
    "Write a Go implementation of a circuit breaker pattern with configurable failure thresholds, timeout, and half-open state.",
    "Implement a simple TCP chat server in Python using asyncio that supports multiple rooms, private messages, and user nicknames.",
    "Write a C++ RAII wrapper for file descriptors that supports move semantics and automatic cleanup.",
    "Create a Python context manager that implements distributed locking using Redis with automatic renewal and deadlock detection.",
    "Write a Rust async runtime-agnostic rate limiter using the token bucket algorithm with burst support.",
    "Implement a write-ahead log (WAL) in Go with fsync guarantees, log compaction, and crash recovery.",
    "Write a Python implementation of a connection pool with health checks, max idle time, and overflow handling.",
    "Implement a simple LSM-tree key-value store in C++ with memtable, SSTables, and basic compaction.",
    "Write a Go gRPC interceptor that implements request tracing with OpenTelemetry span propagation.",

    # ═══ CODING: Web & API (10) ═══
    "Write a TypeScript Express middleware that validates request bodies against JSON Schema with detailed error messages.",
    "Implement a GraphQL resolver in Python (using Strawberry) for a paginated, filterable list of items with cursor-based pagination.",
    "Write a React hook in TypeScript that implements optimistic updates with rollback on failure for a todo list API.",
    "Create a FastAPI endpoint in Python that handles file uploads with streaming, progress tracking, and virus scanning integration.",
    "Write a Next.js API route that implements server-sent events (SSE) for real-time notifications with reconnection handling.",
    "Implement a WebSocket server in Go that handles authentication, heartbeats, and graceful disconnection for a collaborative editor.",
    "Write a Python Flask blueprint that implements OAuth 2.0 authorization code flow with PKCE, including token storage and refresh.",
    "Create a TypeScript Zod schema for a complex nested form with conditional validation rules and custom error messages.",
    "Write a Rust Axum handler that implements multipart form upload with size limits, type validation, and S3 streaming upload.",
    "Implement a Redis-backed session store in Go with serialization, encryption, and sliding expiration.",

    # ═══ CODING: Bug Fixes & Code Review (15) ═══
    "Here's a Python function with a subtle bug. Find and fix it:\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n```",
    "What's wrong with this Rust code and how would you fix it?\n```rust\nfn longest<'a>(x: &str, y: &str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}\n```",
    "Review this SQL query for performance issues:\n```sql\nSELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > '2024-01-01'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;\n```",
    "Find the race condition in this Go code and fix it:\n```go\nvar cache = make(map[string]string)\n\nfunc Get(key string) string {\n    if v, ok := cache[key]; ok {\n        return v\n    }\n    v := fetchFromDB(key)\n    cache[key] = v\n    return v\n}\n```",
    "This Python async code has a subtle deadlock. Identify and fix it:\n```python\nimport asyncio\n\nlock_a = asyncio.Lock()\nlock_b = asyncio.Lock()\n\nasync def task1():\n    async with lock_a:\n        await asyncio.sleep(0.1)\n        async with lock_b:\n            print('task1 done')\n\nasync def task2():\n    async with lock_b:\n        await asyncio.sleep(0.1)\n        async with lock_a:\n            print('task2 done')\n\nasync def main():\n    await asyncio.gather(task1(), task2())\n```",
    "Find the memory leak in this C++ code:\n```cpp\nclass EventManager {\n    std::vector<std::function<void()>> listeners;\npublic:\n    void subscribe(std::function<void()> fn) {\n        listeners.push_back(fn);\n    }\n    void emit() {\n        for (auto& fn : listeners) fn();\n    }\n};\n// Usage:\nauto mgr = std::make_shared<EventManager>();\nauto widget = std::make_shared<Widget>();\nmgr->subscribe([widget]() { widget->update(); });\n```",
    "This TypeScript code has a type safety issue that TypeScript won't catch. Find it:\n```typescript\ninterface User { id: number; name: string; role: 'admin' | 'user'; }\nfunction updateUser(user: User, updates: Partial<User>) {\n    return { ...user, ...updates };\n}\nconst admin = updateUser(getUser(1), { role: 'admin' as any, id: undefined as any });\n```",
    "Fix the off-by-one error in this binary search:\n```python\ndef find_first_ge(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return lo\n```",
    "This Rust code compiles but panics at runtime. Why?\n```rust\nuse std::collections::HashMap;\nfn main() {\n    let mut map: HashMap<String, Vec<i32>> = HashMap::new();\n    let key = \"hello\".to_string();\n    let vec = map.entry(key).or_insert_with(Vec::new);\n    vec.push(1);\n    println!(\"{}\", map[&\"hello\".to_string()].len());\n    println!(\"{}\", map[&key].len()); // key was moved!\n}\n```",
    "Review this React component for performance issues and bugs:\n```jsx\nfunction UserList({ users }) {\n    const [search, setSearch] = useState('');\n    const filtered = users.filter(u => u.name.includes(search));\n    return (\n        <div>\n            <input onChange={e => setSearch(e.target.value)} />\n            {filtered.map(u => (\n                <UserCard key={u.id} user={u} onClick={() => alert(u.name)} />\n            ))}\n        </div>\n    );\n}\n```",
    "Find the security vulnerability in this Express.js route:\n```javascript\napp.get('/api/user/:id', async (req, res) => {\n    const query = `SELECT * FROM users WHERE id = ${req.params.id}`;\n    const user = await db.query(query);\n    res.json(user);\n});\n```",
    "This Python generator has a subtle resource leak. Find and fix it:\n```python\ndef read_large_file(path):\n    f = open(path)\n    for line in f:\n        if line.startswith('#'):\n            continue\n        yield line.strip()\n# Usage: for line in read_large_file('data.txt'): process(line)\n```",
    "What's wrong with this Go error handling pattern?\n```go\nfunc processFile(path string) error {\n    f, err := os.Open(path)\n    if err != nil {\n        return fmt.Errorf(\"open: %w\", err)\n    }\n    defer f.Close()\n    data, err := io.ReadAll(f)\n    if err != nil {\n        return fmt.Errorf(\"read: %w\", err)\n    }\n    return processData(data)\n}\n```\nHint: Consider what happens with the deferred Close() and the error from processData.",
    "This Python dataclass has a mutable default argument bug. Fix it properly:\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass Config:\n    name: str\n    tags: list = []\n    metadata: dict = {}\n```",
    "Find the concurrency bug in this Java code:\n```java\npublic class Counter {\n    private int count = 0;\n    public synchronized void increment() { count++; }\n    public int getCount() { return count; }\n    public synchronized void addAndCheck(int threshold) {\n        increment();\n        if (getCount() >= threshold) { reset(); }\n    }\n    private synchronized void reset() { count = 0; }\n}\n```",

    # ═══ REASONING: Systems & Architecture (10) ═══
    "Explain the CAP theorem and give a concrete example of how a distributed database like Cassandra makes tradeoffs between consistency and availability.",
    "What are the tradeoffs between using microservices vs a monolith? When would you choose each? Give specific examples.",
    "Explain how garbage collection works in Go vs Rust's ownership model. Compare their approaches to memory safety.",
    "Describe the difference between optimistic and pessimistic concurrency control. When would you use each?",
    "Explain how a B+ tree index works in a database and why it's preferred over a hash index for range queries.",
    "Compare event sourcing vs traditional CRUD for a banking application. Cover consistency, auditability, and operational complexity.",
    "Explain the Raft consensus algorithm step by step. How does leader election work? How are log entries committed?",
    "Compare gRPC vs REST vs GraphQL for a microservices architecture serving both mobile and web clients. When would you choose each?",
    "Explain how connection pooling works in database drivers. What happens when the pool is exhausted? How do you size it correctly?",
    "Describe how a CDN works end-to-end. Cover DNS resolution, cache hierarchies, origin shielding, and cache invalidation strategies.",

    # ═══ GENERAL KNOWLEDGE: CS Fundamentals (10) ═══
    "Explain the difference between TCP and UDP. When would you use each? Give real-world examples.",
    "What is the difference between a process and a thread? Explain context switching.",
    "Explain how HTTPS/TLS handshake works step by step.",
    "What is eventual consistency and how does it differ from strong consistency?",
    "Explain the concept of backpressure in streaming systems.",
    "How does DNS resolution work end to end? Walk through what happens when you type google.com in a browser.",
    "Explain how virtual memory works. Cover page tables, TLBs, page faults, and the role of the MMU.",
    "What is the difference between horizontal and vertical scaling? When would you prefer each?",
    "Explain how a compiler transforms source code to machine code. Cover lexing, parsing, AST, IR, optimization, and code generation.",
    "What are ACID properties in databases? Give an example of a transaction that violates each property if not enforced.",

    # ═══ MATH & LOGIC (10) ═══
    "Prove that the square root of 2 is irrational.",
    "Explain the time complexity of quicksort in best, average, and worst cases. Why is the worst case O(n^2)?",
    "Derive the master theorem for recurrence relations and give three examples.",
    "Explain the halting problem and why it's undecidable.",
    "What is the pigeonhole principle? Give three non-trivial applications in computer science.",
    "Prove that the number of edges in a tree with n vertices is exactly n-1.",
    "Explain amortized analysis using the dynamic array (vector) push_back operation as an example. Why is it O(1) amortized?",
    "Prove that every graph with n vertices and more than n-1 edges contains a cycle.",
    "Explain the difference between P, NP, NP-hard, and NP-complete with examples of problems in each class.",
    "Derive the expected number of comparisons in randomized quicksort and show it's O(n log n).",

    # ═══ SYSTEM DESIGN (10) ═══
    "Design a URL shortening service like bit.ly. Cover the API design, database schema, hash generation strategy, and how to handle high traffic.",
    "Design a distributed message queue system. Cover partitioning, replication, ordering guarantees, and consumer group management.",
    "Write a comprehensive guide to implementing OAuth 2.0 with PKCE flow for a mobile application.",
    "Explain how a modern CPU pipeline works, covering instruction fetch, decode, execute, memory access, and write-back. Include branch prediction.",
    "Design a real-time collaborative text editor. Cover CRDTs vs OT, conflict resolution, and network architecture.",
    "Design a rate limiter service that supports multiple strategies (fixed window, sliding window, token bucket). Cover distributed deployment and edge cases.",
    "Design a distributed caching system like Memcached. Cover consistent hashing, cache eviction, thundering herd problem, and cache warming.",
    "Design a notification system that supports push, email, SMS, and in-app notifications with user preferences, batching, and rate limiting.",
    "Design a search autocomplete system. Cover trie-based suggestions, ranking by popularity, personalization, and handling typos.",
    "Design a file storage service like Dropbox. Cover chunking, deduplication, sync conflicts, and bandwidth optimization.",

    # ═══ MULTI-TURN CONVERSATIONS (10) ═══
    "I'm building a REST API in Go for a task management app. What's the best way to structure the project? Start with the directory layout and explain each package's responsibility.",
    "I have a Python Django app that's getting slow. The main page takes 3 seconds to load. It makes 15 database queries per request. Walk me through how to diagnose and fix this step by step.",
    "I'm choosing between PostgreSQL and MongoDB for a new e-commerce platform. Help me decide by analyzing: product catalog (variable attributes), order processing (transactions), inventory management, and analytics queries.",
    "I need to implement authentication for a React + Node.js app. Walk me through the full flow: registration, login, token management, protected routes, and password reset. Use JWTs with refresh tokens.",
    "I'm debugging a production issue: our Node.js service suddenly started returning 502 errors after deploying a new version. CPU is at 100%, memory is normal. Walk me through your debugging approach.",
    "I want to migrate a monolithic Rails app to microservices. The app has users, orders, products, payments, and notifications. Help me plan the decomposition: which services to extract first, how to handle data, and what communication patterns to use.",
    "I'm implementing a CI/CD pipeline for a Python monorepo with 5 services. Walk me through: test strategy, build caching, deployment ordering (some services depend on others), rollback strategy, and monitoring.",
    "I need to add full-text search to my PostgreSQL-based app. Should I use pg_trgm, tsvector, or Elasticsearch? Walk me through the tradeoffs for my use case: 10M documents, 100 queries/sec, need fuzzy matching and faceted search.",
    "I'm writing a custom Kubernetes operator in Go for managing database clusters. Walk me through the controller pattern: reconciliation loop, status management, handling failures, and leader election.",
    "My team is adopting TypeScript for a large JavaScript codebase (200k LOC). Help me plan the migration: strict vs gradual, type generation for APIs, handling third-party libs without types, and CI enforcement.",
]


def load_mlx():
    """Load teacher model using MLX (Mac)."""
    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    teacher_path = os.path.expanduser("~/QWEN-M5/model")
    print(f"Loading MLX teacher model from {teacher_path}...")
    model, tokenizer = load(teacher_path)

    def generate_fn(text, temp, max_tokens):
        sampler = make_sampler(temp=temp)
        return mlx_generate(
            model, tokenizer, prompt=text,
            max_tokens=max_tokens, sampler=sampler,
            prefill_step_size=8192, kv_bits=8, kv_group_size=64,
        )

    return tokenizer, generate_fn


def load_cuda():
    """Load teacher model using CUDA (PC)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "Qwen/Qwen3.5-27B"
    print(f"Loading CUDA teacher model {model_id} in NF4...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    def generate_fn(text, temp, max_tokens):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temp, 0.01),
                do_sample=temp > 0,
                top_p=0.9,
            )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    return tokenizer, generate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Use CUDA backend instead of MLX")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ~/QWEN-M5/calibration_data or ./calibration_data)")
    parser.add_argument("--temperatures", type=str, default="0.3",
                        help="Comma-separated temperatures for response diversity (e.g. '0.2,0.4,0.6')")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens per response")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                        help="Fraction of data for training (rest for validation)")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

    if args.output:
        output_path = Path(args.output)
    elif args.cuda:
        output_path = Path("./calibration_data")
    else:
        output_path = Path(os.path.expanduser("~/QWEN-M5/calibration_data"))

    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.cuda:
        tokenizer, generate_fn = load_cuda()
    else:
        tokenizer, generate_fn = load_mlx()

    # Shuffle prompts for better train/valid split diversity
    prompts = list(PROMPTS)
    random.seed(42)
    random.shuffle(prompts)

    split_idx = int(len(prompts) * args.train_ratio)

    all_data = []
    total = len(prompts) * len(temperatures)
    count = 0

    for i, prompt in enumerate(prompts):
        for temp in temperatures:
            count += 1
            print(f"\n[{count}/{total}] temp={temp:.1f}")
            print(f"  Prompt: {prompt[:80]}...")

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            response = generate_fn(text, temp, args.max_tokens)

            entry = {
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
                "prompt_idx": i,
                "temperature": temp,
            }
            all_data.append(entry)

    # Split into train/valid based on prompt index (not response index)
    train_data = [e for e in all_data if e["prompt_idx"] < split_idx]
    valid_data = [e for e in all_data if e["prompt_idx"] >= split_idx]

    # Remove metadata before saving
    for entry in train_data + valid_data:
        del entry["prompt_idx"]
        del entry["temperature"]

    train_path = output_path / "train.jsonl"
    valid_path = output_path / "valid.jsonl"

    with open(train_path, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    with open(valid_path, "w") as f:
        for entry in valid_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(valid_data)} validation examples to {valid_path}")
    print(f"Total prompts: {len(prompts)}, temperatures: {temperatures}")
    print("Done!")


if __name__ == "__main__":
    main()
