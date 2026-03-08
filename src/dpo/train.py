"""
From-scratch DPO training loop for LLaMA 3.1 8B with QLoRA.

Inspired by the TRL DPOTrainer (src/train_dpo.py) and adapted 
from Assignment 3 (run_dpo.py). Uses an explicit 
training loop whose DPO loss. .

Instead of computing log-probs over continuous action sequences 
like A3, we compute per-token log-probs over generated text and sum 
over response tokens only.

Usage:
    python src/dpo/train.py \
        --train-file data/dpo_train.jsonl \
        --val-file data/dpo_val.jsonl \
        --output-dir checkpoints/dpo-from-scratch

    python src/dpo/train.py \
        --train-file data/dpo_train.jsonl \
        --val-file data/dpo_val.jsonl \
        --output-dir checkpoints/dpo-from-scratch \
        --wandb-project dpo-scratch \
        --epochs 1 --beta 0.1
"""

import argparse
import copy
import os
import sys
from functools import partial

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpo.data import DPODataset, collate_dpo
from dpo.loss import dpo_loss, sequence_log_probs


def compute_logps(model, input_ids, mask):
    """Forward pass and compute summed response-token log-probs."""
    outputs = model(input_ids=input_ids, attention_mask=(input_ids != 0).long())
    return sequence_log_probs(outputs.logits, input_ids, mask)


@torch.no_grad()
def evaluate(model, ref_model, dataloader, beta, device):
    """Compute mean DPO loss and accuracy on a validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)

        policy_chosen = compute_logps(model, chosen_ids, chosen_mask)
        policy_rejected = compute_logps(model, rejected_ids, rejected_mask)
        ref_chosen = compute_logps(ref_model, chosen_ids, chosen_mask)
        ref_rejected = compute_logps(ref_model, rejected_ids, rejected_mask)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        total_loss += loss.item() * chosen_ids.size(0)

        rewards_chosen = beta * (policy_chosen - ref_chosen)
        rewards_rejected = beta * (policy_rejected - ref_rejected)
        total_correct += (rewards_chosen > rewards_rejected).sum().item()
        total_count += chosen_ids.size(0)

    model.train()
    return total_loss / total_count, total_correct / total_count


def train(
    model,
    ref_model,
    optimizer,
    train_loader,
    val_loader,
    epochs: int,
    beta: float,
    grad_accum_steps: int,
    max_grad_norm: float,
    device,
    output_dir: str,
    use_wandb: bool = False,
):
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            policy_chosen = compute_logps(model, chosen_ids, chosen_mask)
            policy_rejected = compute_logps(model, rejected_ids, rejected_mask)

            # Reference model log-probs (frozen, no grad) -- same pattern as HW3
            with torch.no_grad():
                ref_chosen = compute_logps(ref_model, chosen_ids, chosen_mask)
                ref_rejected = compute_logps(ref_model, rejected_ids, rejected_mask)

            loss = dpo_loss(
                policy_chosen, policy_rejected,
                ref_chosen, ref_rejected,
                beta,
            )
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

            if use_wandb and (step + 1) % grad_accum_steps == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,
                    "train/step": global_step,
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Avg train loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, ref_model, val_loader, beta, device)
            print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.2%}")

            if use_wandb:
                import wandb
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "epoch": epoch + 1,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(output_dir, "best")
                model.save_pretrained(save_path)
                print(f"New best model saved to {save_path}")

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="From-scratch DPO training (mirrors CS 234 HW3)"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="checkpoints/dpo-from-scratch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO KL penalty (same as HW3 beta)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--wandb-project", default=None,
                        help="Enable W&B logging with this project name")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or os.path.basename(args.output_dir),
            config=vars(args),
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model with 4-bit quantization (same as src/train_dpo.py)
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model} (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Reference model: frozen copy (same role as ref_policy in HW3)
    # ------------------------------------------------------------------
    print("Creating frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Apply LoRA to the policy model
    # ------------------------------------------------------------------
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = DPODataset(args.train_file)
    val_dataset = DPODataset(args.val_file)
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val:   {len(val_dataset)} pairs")

    collate_fn = partial(collate_dpo, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Optimizer (AdamW, same as HW3)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Starting DPO training...")
    train(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        beta=args.beta,
        grad_accum_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        device=device,
        output_dir=args.output_dir,
        use_wandb=use_wandb,
    )

    if use_wandb:
        import wandb
        wandb.finish()

    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
