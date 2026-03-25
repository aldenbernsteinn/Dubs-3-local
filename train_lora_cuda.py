#!/usr/bin/env python3
"""
LoRA distillation training for Qwen3.5-27B on CUDA (RTX 5090).

Goal: Train a LoRA adapter on a Q3 base model using FP16 teacher outputs,
so the Q3+LoRA model matches Unsloth Q6_K_XL quality.

The teacher (Qwen3.5-27B FP16) is loaded in NF4 for inference on the 5090.
Calibration data was generated on the Mac from a Q5+Q6 teacher, but this
script can regenerate from the full FP16 model for better quality.

Usage:
    pip install -r requirements-cuda.txt
    python train_lora_cuda.py [--regenerate-data]
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_ID = "Qwen/Qwen3.5-27B"
OUTPUT_DIR = Path("./lora-adapter")
DATA_DIR = Path("./calibration_data")

# Calibration prompts (same as Mac generate_calibration.py)
PROMPTS = [
    "Write a Python function that implements Dijkstra's shortest path algorithm using a priority queue. Include type hints and handle edge cases.",
    "Implement a trie data structure in Rust with insert, search, and prefix_search methods. Use proper Rust idioms.",
    "Write a Go function that performs a topological sort on a directed acyclic graph represented as an adjacency list.",
    "Implement a B-tree with order 5 in C++. Include insert and search operations with proper node splitting.",
    "Write a TypeScript class for an LRU cache with O(1) get and put operations using a doubly-linked list and hash map.",
    "Write a Python async web scraper that respects rate limits, handles retries with exponential backoff, and saves results to SQLite.",
    "Implement a simple thread pool in Rust with a fixed number of worker threads and a job queue.",
    "Write a C function that implements a memory allocator using a free list with first-fit allocation strategy.",
    "Create a Go HTTP middleware that implements JWT authentication with token refresh.",
    "Write a Python decorator that implements memoization with TTL (time-to-live) expiry and max cache size.",
    "Implement a skip list in Python with insert, delete, and search operations. Include probabilistic level generation.",
    "Write a Rust implementation of a concurrent queue using atomics, supporting multiple producers and consumers.",
    "Implement a segment tree in C++ that supports range sum queries and point updates.",
    "Write a Python class for a Bloom filter with configurable false positive rate.",
    "Implement a red-black tree in Java with insert, delete, and iterator support.",
    "Explain the CAP theorem and give a concrete example of how a distributed database like Cassandra makes tradeoffs between consistency and availability.",
    "What are the tradeoffs between using microservices vs a monolith? When would you choose each? Give specific examples.",
    "Explain how garbage collection works in Go vs Rust's ownership model. Compare their approaches to memory safety.",
    "Describe the difference between optimistic and pessimistic concurrency control. When would you use each?",
    "Explain how a B+ tree index works in a database and why it's preferred over a hash index for range queries.",
    "Explain the difference between TCP and UDP. When would you use each? Give real-world examples.",
    "What is the difference between a process and a thread? Explain context switching.",
    "Explain how HTTPS/TLS handshake works step by step.",
    "What is eventual consistency and how does it differ from strong consistency?",
    "Explain the concept of backpressure in streaming systems.",
    "Prove that the square root of 2 is irrational.",
    "Explain the time complexity of quicksort in best, average, and worst cases. Why is the worst case O(n^2)?",
    "Derive the master theorem for recurrence relations and give three examples.",
    "Explain the halting problem and why it's undecidable.",
    "What is the pigeonhole principle? Give three non-trivial applications in computer science.",
    "Design a URL shortening service like bit.ly. Cover the API design, database schema, hash generation strategy, and how to handle high traffic.",
    "Design a distributed message queue system. Cover partitioning, replication, ordering guarantees, and consumer group management.",
    "Write a comprehensive guide to implementing OAuth 2.0 with PKCE flow for a mobile application.",
    "Explain how a modern CPU pipeline works, covering instruction fetch, decode, execute, memory access, and write-back. Include branch prediction.",
    "Design a real-time collaborative text editor. Cover CRDTs vs OT, conflict resolution, and network architecture.",
]


def generate_teacher_data(model, tokenizer):
    """Generate calibration data from FP16 teacher model."""
    print("Generating teacher outputs from FP16 model...")
    DATA_DIR.mkdir(exist_ok=True)

    train_data = []
    valid_data = []

    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:60]}...")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        entry = {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"}

        if i < len(PROMPTS) - 5:
            train_data.append(entry)
        else:
            valid_data.append(entry)

    with open(DATA_DIR / "train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    with open(DATA_DIR / "valid.jsonl", "w") as f:
        for entry in valid_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(train_data)} train + {len(valid_data)} valid examples")


def load_data():
    """Load calibration data from JSONL files."""
    train_entries = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            train_entries.append(json.loads(line))

    valid_entries = []
    with open(DATA_DIR / "valid.jsonl") as f:
        for line in f:
            valid_entries.append(json.loads(line))

    return train_entries, valid_entries


def tokenize_data(entries, tokenizer, max_length=1024):
    """Tokenize entries for training."""
    texts = [e["text"] for e in entries]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return Dataset.from_dict({k: v.tolist() for k, v in tokenized.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Regenerate calibration data from FP16 teacher (better quality)")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load model in NF4 (4-bit) for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {MODEL_ID} in NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Generate or load calibration data
    if args.regenerate_data or not (DATA_DIR / "train.jsonl").exists():
        generate_teacher_data(model, tokenizer)

    train_entries, valid_entries = load_data()
    print(f"Loaded {len(train_entries)} train + {len(valid_entries)} valid examples")

    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize data
    train_dataset = tokenize_data(train_entries, tokenizer)
    valid_dataset = tokenize_data(valid_entries, tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\nStarting LoRA training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}")
    print("Transfer this folder back to your Mac and run mlx_lm.fuse to merge.")


if __name__ == "__main__":
    main()
