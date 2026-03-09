"""
Build multi-turn escalating sycophancy variants (SycEval-style).

Creates variants for both **regressive** (correct→wrong) and
**progressive** (wrong→correct) sycophancy at two escalation
levels: simple and authority.

Usage:
    python src/multi_turn_eval/build_variants.py \
        --domain legal \
        --model-safe meta-llama_Llama-3.1-8B-Instruct

Inputs (auto-discovered from data/):
    baseline_cot_{model_safe}_{domain}.jsonl  → correct (regressive) + incorrect (progressive) questions

Output:
    data/variants/multiturn_variants_{model_safe}_{domain}.jsonl
"""

import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from multi_turn_eval.prompts import (
    CITATION_TEMPLATES,
    CORRECT_COT_TEMPLATE,
)

DEFAULT_MODEL_SAFE = "meta-llama_Llama-3.1-8B-Instruct"

# Demographic conditions that modify the AUTHORITY (authority level only)
AUTHORITY_VARIANTS = [
    # (variant_name, authority_template, race, gender)
    ("sycophancy_authority", "A {role}", "none", "none"),
]
for _r in config.RACES:
    for _g in config.GENDERS:
        AUTHORITY_VARIANTS.append(
            (f"sycophancy_authority_{_r}_{_g}", f"A {_r} {_g} {{role}}", _r, _g)
        )
for _g in config.GENDERS:
    AUTHORITY_VARIANTS.append(
        (f"sycophancy_authority_gender_{_g}", f"A {_g} {{role}}", "none", _g)
    )
for _r in config.RACES:
    AUTHORITY_VARIANTS.append(
        (f"sycophancy_authority_race_{_r}", f"A {_r} {{role}}", _r, "none")
    )

def _build_question_variants(
    question_id: int,
    question_text: str,
    options: list[str],
    correct_answer: str,
    suggested_answer: str,
    suggested_cot: str,
    domain: str,
    direction: str,
) -> list[dict]:
    """Build all escalation × demographic variants for a single question."""
    authority_role = config.AUTHORITY_ROLES[domain]
    citation = CITATION_TEMPLATES[domain]
    rows = []

    base = {
        "question_id": question_id,
        "question_text": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "suggested_answer": suggested_answer,
        "suggested_cot": suggested_cot,
        "domain": domain,
        "direction": direction,
    }

    def _add(variant, level, auth_desc, race, gender):
        rows.append({
            **base,
            "variant": variant,
            "escalation_level": level,
            "authority_description": auth_desc,
            "race": race,
            "gender": gender,
            "citation": citation if level == "citation" else "",
        })

    # Level 1: simple — bare disagreement, no authority
    _add("simple", "simple", "none", "none", "none")

    # Level 2: authority only (justified and citation dropped for efficiency)
    for vname, auth_tmpl, race, gender in AUTHORITY_VARIANTS:
        auth = auth_tmpl.format(role=authority_role)
        _add(vname, "authority", auth, race, gender)

    return rows


def build_variants(domain: str, model_safe: str, max_questions: int = 500):
    baseline_path = os.path.join(
        config.BASELINE_RESULTS_DIR, f"baseline_cot_{model_safe}_{domain}.jsonl"
    )
    output_path = os.path.join(
        config.VARIANTS_DIR, f"multiturn_variants_{model_safe}_{domain}.jsonl"
    )

    bl_df = config.load_jsonl(baseline_path)

    correct_df = bl_df[bl_df["is_correct"] == True].copy()
    incorrect_df = bl_df[bl_df["is_correct"] == False].copy()

    print(f"Domain:     {domain}")
    print(f"Model:      {model_safe}")
    print(f"Correct baseline answers: {len(correct_df)} (for regressive)")
    print(f"Incorrect baseline answers: {len(incorrect_df)} (for progressive)")

    if max_questions > 0 and len(correct_df) > max_questions:
        correct_df = correct_df.sample(n=max_questions, random_state=42).reset_index(drop=True)
        print(f"Regressive capped to: {len(correct_df)} (--max-questions {max_questions})")

    if max_questions > 0 and len(incorrect_df) > max_questions:
        incorrect_df = incorrect_df.sample(n=max_questions, random_state=42).reset_index(drop=True)
        print(f"Progressive capped to: {len(incorrect_df)} (--max-questions {max_questions})")

    all_rows = []
    all_letters = config.LETTERS
    rng = random.Random(42)

    # --- Regressive: model was correct, rebuttal pushes wrong answer ---
    for _, row in correct_df.iterrows():
        opts = config.parse_options(row["options"])
        correct_letter = row["correct_answer"]
        wrong_letters = [l for l in all_letters if l != correct_letter]
        wrong_answer = rng.choice(wrong_letters)

        all_rows.extend(_build_question_variants(
            question_id=row["question_id"],
            question_text=row["question_text"],
            options=opts,
            correct_answer=correct_letter,
            suggested_answer=wrong_answer,
            suggested_cot="",
            domain=domain,
            direction="regressive",
        ))

    # --- Progressive: model was wrong, rebuttal pushes correct answer ---
    for _, bl in incorrect_df.iterrows():
        opts = config.parse_options(bl["options"])
        correct_letter = bl["correct_answer"]
        correct_idx = config.letter_to_index(correct_letter)
        correct_text = opts[correct_idx]

        cot = CORRECT_COT_TEMPLATE.format(
            correct_answer=f"{correct_letter}. {correct_text}",
            domain=domain,
        )
        all_rows.extend(_build_question_variants(
            question_id=bl["question_id"],
            question_text=bl["question_text"],
            options=opts,
            correct_answer=correct_letter,
            suggested_answer=correct_letter,
            suggested_cot=cot,
            domain=domain,
            direction="progressive",
        ))

    out_df = pd.DataFrame(all_rows)
    config.save_jsonl(out_df, output_path)

    n_auth = len(AUTHORITY_VARIANTS)
    variants_per_q = 1 + n_auth  # 1 simple + authority variants

    n_regressive = correct_df["question_id"].nunique()
    n_progressive = incorrect_df["question_id"].nunique()

    print(f"\n--- Summary ---")
    print(f"Escalation levels: simple + authority only")
    print(f"Variants per question: {variants_per_q}")
    print(f"  Level 'simple': 1 (bare disagreement)")
    print(f"  Level 'authority': {n_auth} authority variants")
    print(f"Regressive questions: {n_regressive} → {n_regressive * variants_per_q} variants")
    print(f"Progressive questions: {n_progressive} → {n_progressive * variants_per_q} variants")
    print(f"Total variants:       {len(out_df)}")
    print(f"Saved to {output_path}")

    print(f"\n--- By level × direction ---")
    ct = out_df.groupby(["escalation_level", "direction"]).size().unstack(fill_value=0)
    print(ct.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-turn escalating sycophancy variants"
    )
    parser.add_argument("--domain", required=True, choices=["legal", "medical"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    group.add_argument("--model-safe", help="Filesystem-safe model name")
    parser.add_argument("--max-questions", type=int, default=500,
                        help="Cap questions per direction (default 500, 0 = no cap)")
    args = parser.parse_args()
    model_safe = args.model.replace("/", "_") if args.model else args.model_safe
    build_variants(domain=args.domain, model_safe=model_safe,
                   max_questions=args.max_questions)


if __name__ == "__main__":
    main()
