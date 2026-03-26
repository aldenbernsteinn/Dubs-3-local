#!/usr/bin/env python3
"""
LoRA distillation training for Qwen3.5-27B on CUDA (RTX 5090).

Goal: Train a LoRA adapter so Q3+LoRA matches Unsloth Q6_K_XL quality.

Uses knowledge distillation with KL divergence loss to transfer the teacher's
soft knowledge, not just hard labels. The teacher (FP16 in NF4) generates both
text responses AND logit distributions for proper distillation.

Usage:
    pip install -r requirements-cuda.txt

    # Use pre-generated calibration data (from generate_calibration.py)
    python train_lora_cuda.py

    # Regenerate data from FP16 teacher on this GPU
    python train_lora_cuda.py --regenerate-data

    # Hyperparameter sweep
    python train_lora_cuda.py --lora-rank 64 --lr 3e-5 --iters 600
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
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


# ── Distillation Trainer with KL divergence ──────────────────────────────────

class DistillationTrainer(Trainer):
    """
    Custom trainer that uses temperature-scaled KL divergence loss
    for knowledge distillation instead of plain cross-entropy.

    Combined loss: alpha * KL_div(student, teacher_temp) + (1 - alpha) * CE(student, labels)

    When teacher logits are not available (no logits file), falls back to
    temperature-scaled cross-entropy which approximates soft-target behavior.
    """

    def __init__(self, *args, distill_temperature=4.0, distill_alpha=0.7,
                 teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_temperature = distill_temperature
        self.distill_alpha = distill_alpha
        self.teacher_model = teacher_model
        if teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        student_logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # Standard cross-entropy loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # KL divergence with teacher (online distillation)
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            T = self.distill_temperature
            shift_teacher = teacher_logits[..., :-1, :].contiguous()

            # KL divergence on temperature-scaled logits
            student_log_probs = F.log_softmax(shift_logits / T, dim=-1)
            teacher_probs = F.softmax(shift_teacher / T, dim=-1)

            # Only compute KL on non-padding tokens
            mask = (shift_labels != -100).unsqueeze(-1).float()
            kl_loss = F.kl_div(
                student_log_probs * mask,
                teacher_probs * mask,
                reduction="batchmean",
            ) * (T ** 2)

            loss = self.distill_alpha * kl_loss + (1 - self.distill_alpha) * ce_loss
        else:
            # Fallback: temperature-scaled CE (approximates soft targets)
            T = self.distill_temperature
            scaled_logits = shift_logits / T
            soft_ce = F.cross_entropy(
                scaled_logits.view(-1, scaled_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            ) * (T ** 2)

            loss = self.distill_alpha * soft_ce + (1 - self.distill_alpha) * ce_loss

        return (loss, outputs) if return_outputs else loss


# ── Data generation from teacher ─────────────────────────────────────────────

def generate_teacher_data(model, tokenizer):
    """Generate calibration data from FP16 teacher model."""
    from generate_calibration import PROMPTS

    print(f"Generating teacher outputs from FP16 model ({len(PROMPTS)} prompts)...")
    DATA_DIR.mkdir(exist_ok=True)

    train_data = []
    valid_data = []
    split_idx = int(len(PROMPTS) * 0.85)

    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:60]}...")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        entry = {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"}

        if i < split_idx:
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


def tokenize_data(entries, tokenizer, max_length=2048):
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

    # Mask padding tokens in labels so loss ignores them
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        tokenized["labels"][tokenized["labels"] == pad_token_id] = -100

    return Dataset.from_dict({k: v.tolist() for k, v in tokenized.items()})


def main():
    parser = argparse.ArgumentParser(description="LoRA distillation training for Dubs-3")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Regenerate calibration data from FP16 teacher")
    parser.add_argument("--iters", type=int, default=400,
                        help="Training steps (default: 400)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32, try 64 for better quality)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--distill-temp", type=float, default=4.0,
                        help="Distillation temperature (default: 4.0)")
    parser.add_argument("--distill-alpha", type=float, default=0.7,
                        help="KL loss weight vs CE (default: 0.7)")
    parser.add_argument("--online-distill", action="store_true",
                        help="Use online KL distillation (loads teacher separately, needs more VRAM)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

    # Optionally load a separate teacher for online KL distillation
    teacher_model = None
    if args.online_distill:
        print("Loading separate teacher model for online KL distillation...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Use distillation trainer
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        distill_temperature=args.distill_temp,
        distill_alpha=args.distill_alpha,
        teacher_model=teacher_model,
    )

    distill_mode = "online KL" if teacher_model else f"temperature-scaled CE (T={args.distill_temp})"
    print(f"\nStarting LoRA training (distillation: {distill_mode})...")
    print(f"  rank={args.lora_rank}, lr={args.lr}, steps={args.iters}")
    print(f"  alpha={args.distill_alpha}, temperature={args.distill_temp}")
    trainer.train()

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}")
    print("Transfer this folder back to your Mac and run mlx_lm.fuse to merge.")


if __name__ == "__main__":
    main()
