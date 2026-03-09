"""
Build DPO training dataset.

Modes:
  cross-model:  Pairs a non-sycophantic model (chosen) vs a sycophantic model (rejected).
  self-play:    Single model's BF vs WM differential responses as DPO pairs.
                Auto-discovers legal+medical files and combines them.

Usage (self-play, no-COT):
    python src/dataset_generation/build_dpo_dataset.py self-play \
        --model meta-llama/Llama-3.1-8B-Instruct --prompt no-cot \
        --output-prefix dpo_selfplay_llama_nocot
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from single_turn_eval.prompts import SYCOPHANCY_PROMPT, SYCOPHANCY_NO_COT_PROMPT

PROMPT_TEMPLATES = {
    "cot": SYCOPHANCY_PROMPT,
    "no-cot": SYCOPHANCY_NO_COT_PROMPT,
}

BF_VARIANT = "sycophancy_authority_Black_female"
WM_VARIANT = "sycophancy_authority_White_male"

_PROMPT_MODE = "no-cot"


def _load_keyed(path: str) -> dict[tuple, dict]:
    """Load JSONL and key by (question_id, variant)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["question_id"], r["variant"])] = r
    return out


def _reconstruct_prompt(row: dict) -> str:
    """Reconstruct the sycophancy prompt from a result row's fields."""
    options = row["options"]
    template = PROMPT_TEMPLATES[_PROMPT_MODE]
    kwargs = dict(
        domain_expert=config.DOMAIN_EXPERTS[row["domain"]],
        question=row["question_text"],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        authority_description=row["authority_description"],
        suggested_answer=row["suggested_answer"],
    )
    if _PROMPT_MODE == "cot":
        kwargs["suggested_cot"] = row["suggested_cot"]
    return template.format(**kwargs)


# ── cross-model mode ─────────────────────────────────────────────────────

def build_cross_model_pairs(chosen_path: str, rejected_path: str) -> list[dict]:
    chosen = _load_keyed(chosen_path)
    rejected = _load_keyed(rejected_path)

    common_keys = set(chosen.keys()) & set(rejected.keys())

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

        prompt = _reconstruct_prompt(r_row)
        pairs.append({
            "prompt": prompt,
            "chosen": c_row["raw_response"],
            "rejected": r_row["raw_response"],
            "question_id": key[0],
            "variant": key[1],
            "domain": r_row["domain"],
        })

    return pairs


# ── self-play mode ────────────────────────────────────────────────────────

def build_selfplay_pairs(model_path: str, max_questions: int = 0) -> list[dict]:
    """Build DPO pairs from BF vs WM differential within a single model's results."""
    data = _load_keyed(model_path)

    question_ids = sorted({qid for (qid, _) in data.keys()})
    if max_questions > 0 and len(question_ids) > max_questions:
        random.seed(42)
        question_ids = set(random.sample(question_ids, max_questions))
    else:
        question_ids = set(question_ids)

    pairs = []
    for qid in sorted(question_ids):
        bf_key = (qid, BF_VARIANT)
        wm_key = (qid, WM_VARIANT)
        if bf_key not in data or wm_key not in data:
            continue

        bf_row = data[bf_key]
        wm_row = data[wm_key]

        if bf_row["model_answer"] in ("INVALID", "ERROR"):
            continue
        if wm_row["model_answer"] in ("INVALID", "ERROR"):
            continue

        bf_deferred = bf_row["deferred"]
        wm_deferred = wm_row["deferred"]

        if bf_deferred == wm_deferred:
            continue

        if bf_deferred and not wm_deferred:
            # Typical: model caves to BF, resists WM
            # Teach: when you see BF authority, respond like you did for WM
            prompt = _reconstruct_prompt(bf_row)
            chosen = wm_row["raw_response"]
            rejected = bf_row["raw_response"]
            pattern = "BF_caved"
        else:
            # Reverse: model caves to WM, resists BF
            prompt = _reconstruct_prompt(wm_row)
            chosen = bf_row["raw_response"]
            rejected = wm_row["raw_response"]
            pattern = "WM_caved"

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "question_id": qid,
            "domain": bf_row["domain"],
            "pattern": pattern,
        })

    return pairs


# ── shared output logic ───────────────────────────────────────────────────

def _write_split(all_pairs: list[dict], output_dir: str, prefix: str,
                 val_fraction: float, seed: int):
    random.seed(seed)
    random.shuffle(all_pairs)

    n_val = max(1, int(len(all_pairs) * val_fraction))
    val_set = all_pairs[:n_val]
    train_set = all_pairs[n_val:]

    train_path = os.path.join(output_dir, f"{prefix}_train.jsonl")
    val_path = os.path.join(output_dir, f"{prefix}_val.jsonl")

    for path, dataset in [(train_path, train_set), (val_path, val_set)]:
        with open(path, "w") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_pairs)} DPO pairs")
    print(f"Train: {len(train_set)} -> {train_path}")
    print(f"Val:   {len(val_set)} -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Build DPO dataset")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # cross-model mode
    p_cross = subparsers.add_parser("cross-model", help="Chosen model vs rejected model pairs")
    p_cross.add_argument("--chosen-legal", required=True)
    p_cross.add_argument("--chosen-medical", required=True)
    p_cross.add_argument("--rejected-legal", required=True)
    p_cross.add_argument("--rejected-medical", required=True)
    p_cross.add_argument("--output-prefix", default="dpo")

    # self-play mode
    p_self = subparsers.add_parser("self-play", help="BF vs WM self-play pairs (regressive + progressive, auto-combines domains)")
    p_self.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p_self.add_argument("--results-dir", default="data/results/single_turn",
                        help="Directory containing single-turn result JSONLs")
    p_self.add_argument("--output-prefix", default="dpo_selfplay")
    p_self.add_argument("--prompt", default="no-cot", choices=["cot", "no-cot"],
                        help="Prompt template for reconstructed prompts (cot includes reasoning, no-cot does not)")
    p_self.add_argument("--max-questions", type=int, default=500,
                        help="Cap questions per direction per domain (default 500, 0 = no cap)")

    for p in [p_cross, p_self]:
        p.add_argument("--val-fraction", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output-dir", default=config.DATA_DIR)

    args = parser.parse_args()

    global _PROMPT_MODE
    if hasattr(args, "prompt"):
        _PROMPT_MODE = args.prompt

    if args.mode == "cross-model":
        legal_pairs = build_cross_model_pairs(args.chosen_legal, args.rejected_legal)
        print(f"legal: {len(legal_pairs)} cross-model DPO pairs")
        medical_pairs = build_cross_model_pairs(args.chosen_medical, args.rejected_medical)
        print(f"medical: {len(medical_pairs)} cross-model DPO pairs")
        all_pairs = legal_pairs + medical_pairs

    elif args.mode == "self-play":
        file_prefix = "sycophancy_regressive_no_cot" if args.prompt == "no-cot" else "sycophancy_regressive"
        prog_prefix = "sycophancy_progressive_no_cot" if args.prompt == "no-cot" else "sycophancy_progressive"
        model_safe = args.model.replace("/", "_")
        max_q = args.max_questions

        pairs_by_direction: dict[str, list[dict]] = {}
        for direction, prefix in [("regressive", file_prefix), ("progressive", prog_prefix)]:
            dir_pairs = []
            for domain in ["legal", "medical"]:
                path = os.path.join(args.results_dir, f"{prefix}_{model_safe}_{domain}.jsonl")
                if not os.path.exists(path):
                    print(f"[{direction}] {domain}: file not found, skipping ({path})")
                    continue
                pairs = build_selfplay_pairs(path, max_questions=max_q)
                for p in pairs:
                    p["direction"] = direction
                bf_caved = sum(1 for p in pairs if p["pattern"] == "BF_caved")
                wm_caved = len(pairs) - bf_caved
                print(f"[{direction}] {domain}: {len(pairs)} pairs (BF caved: {bf_caved}, WM caved: {wm_caved})")
                dir_pairs.extend(pairs)
            print(f"[{direction}] total: {len(dir_pairs)} pairs")
            pairs_by_direction[direction] = dir_pairs

        groups = [v for v in pairs_by_direction.values() if v]
        if len(groups) >= 2:
            min_size = min(len(g) for g in groups)
            balanced = []
            for direction, group in pairs_by_direction.items():
                if len(group) > min_size:
                    random.seed(42)
                    group = random.sample(group, min_size)
                    print(f"[{direction}] downsampled to {min_size} for 50/50 balance")
                balanced.extend(group)
            all_pairs = balanced
        else:
            all_pairs = [p for g in groups for p in g]

    if not all_pairs:
        print("ERROR: No pairs generated. Check input files.")
        sys.exit(1)

    _write_split(all_pairs, args.output_dir, args.output_prefix,
                 args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
