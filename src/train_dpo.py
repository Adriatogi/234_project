"""
DPO fine-tuning of LLaMA 3.1 8B to resist sycophancy using QLoRA.

Trains on preference pairs where:
  - chosen = non-sycophantic response (from Qwen)
  - rejected = sycophantic response (from LLaMA)

Usage:
    python src/train_dpo.py \
        --train-file data/dpo_train.jsonl \
        --val-file data/dpo_val.jsonl \
        --output-dir checkpoints/dpo-llama-anti-syco

    python src/train_dpo.py \
        --train-file data/dpo_train.jsonl \
        --val-file data/dpo_val.jsonl \
        --output-dir checkpoints/dpo-llama-anti-syco \
        --wandb-project dpo-anti-sycophancy \
        --epochs 3
"""

import argparse
import json
import os

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_jsonl_dataset(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for anti-sycophancy")
    parser.add_argument("--train-file", required=True, help="Path to DPO training JSONL")
    parser.add_argument("--val-file", required=True, help="Path to DPO validation JSONL")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="checkpoints/dpo-llama-anti-syco")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 = use epochs)")
    parser.add_argument("--wandb-project", default="dpo-anti-sycophancy")
    parser.add_argument("--run-name", default=None, help="W&B run name (defaults to output_dir basename)")
    args = parser.parse_args()

    run_name = args.run_name or os.path.basename(args.output_dir)

    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Loading datasets...")
    train_dataset = load_jsonl_dataset(args.train_file)
    val_dataset = load_jsonl_dataset(args.val_file)
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val:   {len(val_dataset)} pairs")

    print(f"Loading model: {args.model} (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=1 if args.max_steps > 0 else 10,
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        save_strategy="no" if args.max_steps > 0 else "epoch",
        save_total_limit=2,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    print("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(f"Final eval metrics: {metrics}")

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
