"""
Run rebuttal sycophancy inference.

A cached baseline answer is injected as the assistant turn,
then an authority rebuttal is presented as a follow-up. This measures
whether the model defers to wrong suggestions delivered as rebuttals.

Usage:
    python src/rebuttal_eval/run_rebuttal_inference.py \
        --input data/sycophancy_variants_legal.jsonl \
        --baseline data/results/baseline_cot_meta-llama_Llama-3.1-8B-Instruct_legal.jsonl \
        --backend vllm --model meta-llama/Llama-3.1-8B-Instruct

Output:
    data/results/rebuttal_sycophancy_{model}_{domain}.jsonl
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import llm_backend
from rebuttal_eval.prompts import INITIAL_QUESTION_PROMPT, REBUTTAL_PROMPT

load_dotenv(os.path.join(config.DATA_DIR, "..", ".env"))


def format_turn1(row: pd.Series, options: list[str]) -> str:
    """Format the initial question prompt (turn 1, user)."""
    domain = row["domain"]
    return INITIAL_QUESTION_PROMPT.format(
        domain_expert=config.DOMAIN_EXPERTS[domain],
        question=row["question_text"],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def format_rebuttal(row: pd.Series) -> str:
    """Format the authority rebuttal prompt (turn 2, user)."""
    return REBUTTAL_PROMPT.format(
        authority_description=row["authority_description"],
        suggested_answer=row["suggested_answer"],
        suggested_cot=row["suggested_cot"],
    )


def build_messages(turn1: str, baseline_response: str, rebuttal: str) -> list[dict]:
    """Assemble the 3-message rebuttal conversation."""
    return [
        {"role": "user", "content": turn1},
        {"role": "assistant", "content": baseline_response},
        {"role": "user", "content": rebuttal},
    ]


def run_rebuttal(
    input_path: str,
    baseline_path: str,
    model: str,
    limit: int | None,
    question_ids: list[int] | None,
    batch_size: int,
    backend: str,
    max_tokens: int = 512,
):
    variants_df = config.load_jsonl(input_path)
    if "authority_description" not in variants_df.columns:
        raise ValueError(
            "Input lacks 'authority_description' column — "
            "not a sycophancy variants file."
        )

    baseline_df = config.load_jsonl(baseline_path)
    baseline_correct = baseline_df[baseline_df["is_correct"] == True].copy()

    baseline_lookup = {}
    for _, row in baseline_correct.iterrows():
        baseline_lookup[row["question_id"]] = row["raw_response"]

    if question_ids is not None:
        variants_df = variants_df[variants_df["question_id"].isin(question_ids)]
    elif limit is not None:
        unique_ids = sorted(variants_df["question_id"].unique())[:limit]
        variants_df = variants_df[variants_df["question_id"].isin(unique_ids)]

    domain = variants_df["domain"].iloc[0]
    model_safe = config.safe_model_name(model)

    print(f"Model:      {model}")
    print(f"Backend:    {backend}")
    print(f"Domain:     {domain}")
    print(f"Mode:       rebuttal")
    print(f"Input:      {os.path.basename(input_path)}")
    print(f"Baseline:   {os.path.basename(baseline_path)}")
    print(f"Baseline correct answers: {len(baseline_lookup)}")
    print(f"Batch size: {batch_size}")
    print(f"Variants:   {len(variants_df)}")
    print(f"Questions:  {variants_df['question_id'].nunique()}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(
        config.RESULTS_DIR, f"rebuttal_sycophancy_{model_safe}_{domain}.jsonl"
    )

    existing_df, completed_keys = llm_backend.load_existing(
        output_path, ["question_id", "variant"]
    )

    pending = []
    skipped_no_baseline = 0
    for _, row in variants_df.iterrows():
        key = (row["question_id"], row["variant"])
        if key in completed_keys:
            continue

        baseline_response = baseline_lookup.get(row["question_id"])
        if baseline_response is None:
            skipped_no_baseline += 1
            continue

        options = row["options"]
        if len(options) != 4:
            continue

        turn1 = format_turn1(row, options)
        rebuttal = format_rebuttal(row)
        messages = build_messages(turn1, baseline_response, rebuttal)

        pending.append((row, messages))

    print(f"Skipped (no baseline): {skipped_no_baseline}")
    print(f"Pending:               {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    def _base_result(row, answer, raw):
        return {
            "question_id": row["question_id"],
            "variant": row["variant"],
            "race": row["race"],
            "gender": row["gender"],
            "question_text": row["question_text"],
            "options": row["options"],
            "correct_answer": row["correct_answer"],
            "model_answer": answer,
            "is_correct": answer == row["correct_answer"],
            "raw_response": raw,
            "suggested_answer": row["suggested_answer"],
            "suggested_cot": row["suggested_cot"],
            "authority_description": row["authority_description"],
            "deferred": answer == row["suggested_answer"],
            "domain": row["domain"],
            "mode": "rebuttal",
            "model": model,
        }

    def _result(row, text):
        return _base_result(row, config.extract_answer(text), text)

    def _error(row, err):
        result = _base_result(row, "ERROR", err)
        result["is_correct"] = False
        result["deferred"] = False
        return result

    results = llm_backend.run_inference_rebuttal(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=max_tokens, total_count=len(variants_df),
        build_result=_result, build_error=_error,
    )

    if results:
        deferred_count = sum(1 for r in results if r.get("deferred"))
        print(f"Deference rate: {deferred_count}/{len(results)} "
              f"({deferred_count / len(results) * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Run rebuttal sycophancy inference"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input JSONL of sycophancy variants",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Baseline COT results JSONL (e.g. data/results/baseline_cot_..._legal.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", type=str, default=None)
    llm_backend.add_common_args(parser, default_max_tokens=512)

    args = parser.parse_args()
    llm_backend.init_backend(args.backend, args.model)

    question_ids = None
    if args.question_ids:
        question_ids = [int(x) for x in args.question_ids.split(",")]

    run_rebuttal(
        input_path=args.input, baseline_path=args.baseline,
        model=args.model, limit=args.limit,
        question_ids=question_ids, batch_size=args.batch_size,
        backend=args.backend, max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
