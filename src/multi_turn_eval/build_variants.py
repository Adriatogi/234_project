"""
Build multi-turn escalating sycophancy variants (SycEval-style).

Creates variants for both **regressive** (correct→wrong) and
**progressive** (wrong→correct) sycophancy at four cumulative
escalation levels: simple, authority, justified, citation.

Usage:
    python src/multi_turn_eval/build_variants.py \
        --domain legal \
        --model-safe meta-llama_Llama-3.1-8B-Instruct

Inputs (auto-discovered from data/):
    wrong_cots_{model_safe}_{domain}.jsonl   → regressive questions
    baseline_cot_{model_safe}_{domain}.jsonl  → progressive questions + baseline responses

Output:
    data/multiturn_variants_{model_safe}_{domain}.jsonl
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from dataset_generation.build_sycophancy_variants import inject_demographic
from multi_turn_eval.prompts import (
    CITATION_TEMPLATES,
    CORRECT_COT_TEMPLATE,
    ESCALATION_LEVELS,
)

DEFAULT_MODEL_SAFE = "meta-llama_Llama-3.1-8B-Instruct"

# Demographic conditions that modify the AUTHORITY (levels 2-4 only)
AUTHORITY_VARIANTS = [
    # (variant_name, authority_template, race, gender)
    ("sycophancy_user", "A user", "none", "none"),
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

# Person demographic conditions (modify question text, neutral authority)
PERSON_VARIANTS = []
for _r in config.RACES:
    for _g in config.GENDERS:
        PERSON_VARIANTS.append(
            (f"sycophancy_person_{_r}_{_g}", _r, _g)
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

    def _add(variant, level, auth_desc, race, gender, q_text=None):
        rows.append({
            **base,
            "variant": variant,
            "escalation_level": level,
            "authority_description": auth_desc,
            "race": race,
            "gender": gender,
            "citation": citation if level == "citation" else "",
            **({"question_text": q_text} if q_text else {}),
        })

    # Level 1: simple — one base + person demo variants
    _add("simple", "simple", "none", "none", "none")
    for pname, race, gender in PERSON_VARIANTS:
        injected = inject_demographic(question_text, race, gender)
        _add(f"simple_person_{race}_{gender}", "simple", "none", race, gender,
             q_text=injected)

    # Levels 2-4: authority / justified / citation
    for level in ESCALATION_LEVELS[1:]:
        for vname, auth_tmpl, race, gender in AUTHORITY_VARIANTS:
            auth = auth_tmpl.format(role=authority_role)
            _add(vname, level, auth, race, gender)

        for pname, race, gender in PERSON_VARIANTS:
            injected = inject_demographic(question_text, race, gender)
            auth = f"A {authority_role}"
            _add(pname, level, auth, race, gender, q_text=injected)

    return rows


def build_variants(domain: str, model_safe: str):
    wrong_cots_path = os.path.join(
        config.DATA_DIR, f"wrong_cots_{model_safe}_{domain}.jsonl"
    )
    baseline_path = os.path.join(
        config.RESULTS_DIR, f"baseline_cot_{model_safe}_{domain}.jsonl"
    )
    output_path = os.path.join(
        config.DATA_DIR, f"multiturn_variants_{model_safe}_{domain}.jsonl"
    )

    wc_df = config.load_jsonl(wrong_cots_path)
    bl_df = config.load_jsonl(baseline_path)

    print(f"Domain:     {domain}")
    print(f"Model:      {model_safe}")
    print(f"Wrong COTs: {len(wc_df)} (for regressive)")

    incorrect_df = bl_df[bl_df["is_correct"] == False].copy()
    print(f"Incorrect baseline answers: {len(incorrect_df)} (for progressive)")

    all_rows = []

    # --- Regressive: model was correct, rebuttal pushes wrong answer ---
    for _, wc in wc_df.iterrows():
        opts = config.parse_options(wc["options"])
        all_rows.extend(_build_question_variants(
            question_id=wc["question_id"],
            question_text=wc["question_text"],
            options=opts,
            correct_answer=wc["correct_answer"],
            suggested_answer=wc["wrong_answer"],
            suggested_cot=wc["wrong_cot"],
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

    n_levels = len(ESCALATION_LEVELS)
    n_auth = len(AUTHORITY_VARIANTS)
    n_person = len(PERSON_VARIANTS)
    variants_per_q = (1 + n_person) + (n_auth + n_person) * (n_levels - 1)

    n_regressive = wc_df["question_id"].nunique()
    n_progressive = incorrect_df["question_id"].nunique()

    print(f"\n--- Summary ---")
    print(f"Escalation levels: {ESCALATION_LEVELS}")
    print(f"Variants per question: {variants_per_q}")
    print(f"  Level 'simple': {1 + n_person} (1 base + {n_person} person demo)")
    print(f"  Levels 2-4:     {n_auth + n_person} each × {n_levels - 1} levels")
    print(f"Regressive questions: {n_regressive} → {n_regressive * variants_per_q} variants")
    print(f"Progressive questions: {n_progressive} → {n_progressive * variants_per_q} variants")
    print(f"Total variants:       {len(out_df)}")
    print(f"Saved to {output_path}")

    # Quick breakdown
    print(f"\n--- By level × direction ---")
    ct = out_df.groupby(["escalation_level", "direction"]).size().unstack(fill_value=0)
    print(ct.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-turn escalating sycophancy variants"
    )
    parser.add_argument("--domain", required=True, choices=["legal", "medical"])
    parser.add_argument(
        "--model-safe", default=DEFAULT_MODEL_SAFE,
        help="Model-safe name (must match wrong_cots / baseline_cot filenames)",
    )
    args = parser.parse_args()
    build_variants(domain=args.domain, model_safe=args.model_safe)


if __name__ == "__main__":
    main()
