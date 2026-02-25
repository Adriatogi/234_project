"""
Build DPO training dataset from a sycophantic model's results vs a non-sycophantic model's results.

Pairs where the rejected model deferred to the wrong suggestion and the chosen model did not
become DPO preference pairs: chosen = non-sycophantic response, rejected = sycophantic response.

Usage:
    python src/dataset_generation/build_dpo_dataset.py \
        --chosen-legal data/results/sycophancy_Qwen_Qwen2.5-7B-Instruct_legal.jsonl \
        --chosen-medical data/results/sycophancy_Qwen_Qwen2.5-7B-Instruct_medical.jsonl \
        --rejected-legal data/results/sycophancy_meta-llama_Llama-3.1-8B-Instruct_legal.jsonl \
        --rejected-medical data/results/sycophancy_meta-llama_Llama-3.1-8B-Instruct_medical.jsonl

Outputs:
    data/dpo_train.jsonl
    data/dpo_val.jsonl
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from eval.prompts import SYCOPHANCY_PROMPT

VARIANT_FILES = {
    "legal": os.path.join(config.DATA_DIR, "sycophancy_variants_legal.jsonl"),
    "medical": os.path.join(config.DATA_DIR, "sycophancy_variants_medical.jsonl"),
}


def _load_keyed(path: str) -> dict[tuple, dict]:
    """Load JSONL and key by (question_id, variant)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["question_id"], r["variant"])] = r
    return out


def _reconstruct_prompt(variant_row: dict) -> str:
    """Reconstruct the sycophancy prompt from a variant row."""
    options = variant_row["options"]
    return SYCOPHANCY_PROMPT.format(
        domain_expert=config.DOMAIN_EXPERTS[variant_row["domain"]],
        question=variant_row["question_text"],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        authority_description=variant_row["authority_description"],
        suggested_answer=variant_row["suggested_answer"],
        suggested_cot=variant_row["suggested_cot"],
    )


def build_pairs(chosen_path: str, rejected_path: str, domain: str) -> list[dict]:
    chosen = _load_keyed(chosen_path)
    rejected = _load_keyed(rejected_path)
    variants = _load_keyed(VARIANT_FILES[domain])

    common_keys = set(chosen.keys()) & set(rejected.keys()) & set(variants.keys())

    pairs = []
    for key in common_keys:
        c_row = chosen[key]
        r_row = rejected[key]

        if c_row["model_answer"] in ("INVALID", "ERROR"):
            continue
        if r_row["model_answer"] in ("INVALID", "ERROR"):
            continue
        if not r_row["deferred"]:
            continue
        if c_row["deferred"]:
            continue

        prompt = _reconstruct_prompt(variants[key])
        pairs.append({
            "prompt": prompt,
            "chosen": c_row["raw_response"],
            "rejected": r_row["raw_response"],
            "question_id": key[0],
            "variant": key[1],
            "domain": domain,
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build DPO dataset from chosen/rejected sycophancy results")
    parser.add_argument("--chosen-legal", required=True, help="Path to chosen model's legal results JSONL")
    parser.add_argument("--chosen-medical", required=True, help="Path to chosen model's medical results JSONL")
    parser.add_argument("--rejected-legal", required=True, help="Path to rejected model's legal results JSONL")
    parser.add_argument("--rejected-medical", required=True, help="Path to rejected model's medical results JSONL")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--output-dir", default=config.DATA_DIR, help="Directory for output files")
    args = parser.parse_args()

    legal_pairs = build_pairs(args.chosen_legal, args.rejected_legal, "legal")
    print(f"legal: {len(legal_pairs)} DPO pairs")

    medical_pairs = build_pairs(args.chosen_medical, args.rejected_medical, "medical")
    print(f"medical: {len(medical_pairs)} DPO pairs")

    all_pairs = legal_pairs + medical_pairs

    random.seed(args.seed)
    random.shuffle(all_pairs)

    n_val = int(len(all_pairs) * args.val_fraction)
    val_set = all_pairs[:n_val]
    train_set = all_pairs[n_val:]

    train_path = os.path.join(args.output_dir, "dpo_train.jsonl")
    val_path = os.path.join(args.output_dir, "dpo_val.jsonl")

    for path, dataset in [(train_path, train_set), (val_path, val_set)]:
        with open(path, "w") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_pairs)} DPO pairs")
    print(f"Train: {len(train_set)} -> {train_path}")
    print(f"Val:   {len(val_set)} -> {val_path}")


if __name__ == "__main__":
    main()
