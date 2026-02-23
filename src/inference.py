"""
LLM inference pipeline with three subcommands.

Usage:
    python src/inference.py baseline --domain legal --limit 500 --batch-size 20
    python src/inference.py wrong-cot --domain legal --baseline <path> --batch-size 20
    python src/inference.py sycophancy --input <path> --batch-size 20

Outputs (all JSONL):
    data/results/baseline_cot_{model}_{domain}.jsonl
    data/wrong_cots_{model}_{domain}.jsonl
    data/results/sycophancy_{model}_{domain}.jsonl
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from litellm import batch_completion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASELINE_COT_PROMPT,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DOMAIN_EXPERTS,
    DOMAIN_LABELS,
    EXPERIMENT1_PROMPTS,
    LETTERS,
    RESULTS_DIR,
    SYCOPHANCY_PROMPT,
    WRONG_COT_PROMPT,
    extract_answer,
    letter_to_index,
    load_jsonl,
    parse_options,
    safe_model_name,
    save_results,
)

load_dotenv(os.path.join(DATA_DIR, "..", ".env"))

DOMAIN_INPUTS = {
    "legal": os.path.join(DATA_DIR, "mmlu_professional_law.csv"),
    "medical": os.path.join(DATA_DIR, "medqa.csv"),
}


# ===================================================================
# BASELINE subcommand — run COT inference on raw questions
# ===================================================================

def load_questions(domain: str) -> pd.DataFrame:
    """Load and normalize questions from raw HuggingFace CSVs.

    Returns DataFrame with: question_id, question_text, options (list), correct_answer
    """
    path = DOMAIN_INPUTS[domain]
    df = pd.read_csv(path)

    if domain == "legal":
        rows = []
        for idx, row in df.iterrows():
            options = parse_options(row["options"])
            correct = ast.literal_eval(row["correct_options"])[0]
            rows.append({
                "question_id": idx,
                "question_text": row["centerpiece"],
                "options": options,
                "correct_answer": correct,
            })
        return pd.DataFrame(rows)

    elif domain == "medical":
        rows = []
        for idx, row in df.iterrows():
            options = parse_options(row["options"])
            rows.append({
                "question_id": idx,
                "question_text": row["question"],
                "options": options,
                "correct_answer": row["answer_idx"],
            })
        return pd.DataFrame(rows)

    else:
        raise ValueError(f"Unknown domain: {domain}")


def extract_cot(response_text: str) -> str:
    """Extract chain-of-thought reasoning from the model's response."""
    match = re.search(r"Reasoning:\s*(.+)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"Answer:\s*[A-Da-d]\s*\n(.+)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def format_baseline_prompt(domain: str, question: str, options: list[str]) -> str:
    return BASELINE_COT_PROMPT.format(
        domain_expert=DOMAIN_EXPERTS[domain],
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def run_baseline(domain: str, model: str, limit: int | None, batch_size: int):
    df = load_questions(domain)
    if limit is not None:
        df = df.head(limit)

    print(f"Domain:     {domain}")
    print(f"Model:      {model}")
    print(f"Questions:  {len(df)}")
    print(f"Batch size: {batch_size}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_safe = safe_model_name(model)
    output_path = os.path.join(RESULTS_DIR, f"baseline_cot_{model_safe}_{domain}.jsonl")

    existing_df = None
    completed_ids = set()
    if os.path.exists(output_path):
        existing_df = load_jsonl(output_path)
        existing_df = existing_df.drop_duplicates(subset=["question_id"], keep="first")
        completed_ids = set(existing_df["question_id"])
        print(f"Resuming: {len(completed_ids)} questions already completed")

    pending = []
    for _, row in df.iterrows():
        if row["question_id"] in completed_ids:
            continue
        prompt = format_baseline_prompt(domain, row["question_text"], row["options"])
        pending.append((row, prompt))

    print(f"Pending:    {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    results = []
    errors = 0

    for batch_start in range(0, len(pending), batch_size):
        chunk = pending[batch_start:batch_start + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for _, p in chunk]

        responses = batch_completion(
            model=model, messages=messages_batch,
            max_tokens=1024, temperature=0.0,
        )

        for (row, _), response in zip(chunk, responses):
            if isinstance(response, Exception):
                errors += 1
                print(f"  Error on Q{row['question_id']}: {response}")
                results.append({
                    "question_id": row["question_id"],
                    "question_text": row["question_text"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "model_answer": "ERROR",
                    "is_correct": False,
                    "model_cot": "",
                    "raw_response": str(response),
                    "domain": domain,
                    "model": model,
                })
                continue

            response_text = response.choices[0].message.content
            answer = extract_answer(response_text)
            cot = extract_cot(response_text)

            results.append({
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "options": row["options"],
                "correct_answer": row["correct_answer"],
                "model_answer": answer,
                "is_correct": answer == row["correct_answer"],
                "model_cot": cot,
                "raw_response": response_text,
                "domain": domain,
                "model": model,
            })

        completed = len(results) + len(completed_ids)
        total = len(df)
        pct = completed / total * 100
        correct_so_far = sum(1 for r in results if r["is_correct"])
        acc = correct_so_far / len(results) * 100 if results else 0
        print(f"  Batch {batch_start // batch_size + 1}: "
              f"{completed}/{total} ({pct:.1f}%) | "
              f"Accuracy: {acc:.1f}% | Errors: {errors}")
        save_results(results, output_path, existing_df)

    total_completed = len(results) + len(completed_ids)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / len(results) * 100 if results else 0
    print(f"\nDone! {total_completed} questions completed. Errors: {errors}")
    print(f"Accuracy (this run): {correct}/{len(results)} ({acc:.1f}%)")
    print(f"Saved to {output_path}")


# ===================================================================
# WRONG-COT subcommand — generate wrong chain-of-thought for correct answers
# ===================================================================

def pick_wrong_answer(correct: str) -> str:
    """Pick the first wrong letter alphabetically (deterministic)."""
    for letter in LETTERS:
        if letter != correct:
            return letter
    raise ValueError(f"No wrong answer found for correct={correct}")


def run_wrong_cot(baseline_path: str, domain: str, model: str, batch_size: int):
    df = load_jsonl(baseline_path)
    correct_df = df[df["is_correct"] == True].copy()

    print(f"Domain:            {domain}")
    print(f"Model:             {model}")
    print(f"Batch size:        {batch_size}")
    print(f"Baseline total:    {len(df)}")
    print(f"Answered correctly: {len(correct_df)}")

    if len(correct_df) == 0:
        print("No correct answers found — nothing to generate.")
        return

    model_safe = safe_model_name(model)
    output_path = os.path.join(DATA_DIR, f"wrong_cots_{model_safe}_{domain}.jsonl")

    existing_df = None
    completed_ids = set()
    if os.path.exists(output_path):
        existing_df = load_jsonl(output_path)
        existing_df = existing_df.drop_duplicates(subset=["question_id"], keep="first")
        completed_ids = set(existing_df["question_id"])
        print(f"Resuming: {len(completed_ids)} already completed")

    pending = []
    for _, row in correct_df.iterrows():
        if row["question_id"] in completed_ids:
            continue

        options = row["options"]
        correct_letter = row["correct_answer"]
        wrong_letter = pick_wrong_answer(correct_letter)
        wrong_text = options[letter_to_index(wrong_letter)]

        prompt = WRONG_COT_PROMPT.format(
            domain=DOMAIN_LABELS[domain],
            wrong_letter=wrong_letter,
            wrong_text=wrong_text,
            question=row["question_text"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        )
        pending.append((row, wrong_letter, wrong_text, prompt))

    print(f"Pending:           {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    results = []
    errors = 0

    for batch_start in range(0, len(pending), batch_size):
        chunk = pending[batch_start:batch_start + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for _, _, _, p in chunk]

        responses = batch_completion(
            model=model, messages=messages_batch,
            max_tokens=1024, temperature=0.0,
        )

        for (row, wrong_letter, wrong_text, _), response in zip(chunk, responses):
            if isinstance(response, Exception):
                errors += 1
                print(f"  Error on Q{row['question_id']}: {response}")
                results.append({
                    "question_id": row["question_id"],
                    "question_text": row["question_text"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "wrong_answer": wrong_letter,
                    "wrong_answer_text": wrong_text,
                    "wrong_cot": f"ERROR: {response}",
                    "domain": domain,
                    "model": model,
                })
                continue

            response_text = response.choices[0].message.content
            results.append({
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "options": row["options"],
                "correct_answer": row["correct_answer"],
                "wrong_answer": wrong_letter,
                "wrong_answer_text": wrong_text,
                "wrong_cot": response_text.strip(),
                "domain": domain,
                "model": model,
            })

        completed = len(results) + len(completed_ids)
        total = len(correct_df)
        pct = completed / total * 100
        print(f"  Batch {batch_start // batch_size + 1}: "
              f"{completed}/{total} ({pct:.1f}%) | Errors: {errors}")
        save_results(results, output_path, existing_df)

    total_completed = len(results) + len(completed_ids)
    print(f"\nDone! {total_completed} wrong COTs generated. Errors: {errors}")
    print(f"Saved to {output_path}")

    if results:
        ex = results[0]
        print(f"\n--- Example (Q{ex['question_id']}) ---")
        print(f"  Correct: {ex['correct_answer']}")
        print(f"  Wrong:   {ex['wrong_answer']} ({ex['wrong_answer_text'][:80]})")
        print(f"  COT:     {ex['wrong_cot'][:200]}...")


# ===================================================================
# SYCOPHANCY subcommand — run inference on sycophancy/experiment1 variants
# ===================================================================

def load_prompt_template(prompt_name: str) -> str:
    """Look up an Experiment 1 prompt template by name."""
    return EXPERIMENT1_PROMPTS[prompt_name]


def format_prompt_standard(template: str, question: str, options: list[str]) -> str:
    return template.format(
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def format_prompt_sycophancy(row: pd.Series, options: list[str]) -> str:
    domain = row["domain"]
    return SYCOPHANCY_PROMPT.format(
        domain_expert=DOMAIN_EXPERTS[domain],
        question=row["question_text"],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        authority_description=row["authority_description"],
        suggested_answer=row["suggested_answer"],
        suggested_cot=row["suggested_cot"],
    )


def run_sycophancy(
    input_path: str,
    model: str,
    prompt_name: str,
    limit: int | None,
    question_ids: list[int] | None,
    batch_size: int,
):
    df = load_jsonl(input_path)
    is_sycophancy = "authority_description" in df.columns

    template = None
    if prompt_name == "sycophancy":
        if not is_sycophancy:
            raise ValueError(
                "Input lacks 'authority_description' column — "
                "not a sycophancy variants file."
            )
    else:
        template = load_prompt_template(prompt_name)

    if question_ids is not None:
        df = df[df["question_id"].isin(question_ids)]
    elif limit is not None:
        unique_ids = sorted(df["question_id"].unique())[:limit]
        df = df[df["question_id"].isin(unique_ids)]

    print(f"Model:      {model}")
    print(f"Prompt:     {prompt_name}")
    print(f"Input:      {os.path.basename(input_path)}")
    print(f"Sycophancy: {is_sycophancy}")
    print(f"Batch size: {batch_size}")
    print(f"Variants:   {len(df)}")
    print(f"Questions:  {df['question_id'].nunique()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_safe = safe_model_name(model)

    if is_sycophancy and "domain" in df.columns:
        domain = df["domain"].iloc[0]
        output_path = os.path.join(
            RESULTS_DIR, f"sycophancy_{model_safe}_{domain}.jsonl"
        )
    else:
        output_path = os.path.join(
            RESULTS_DIR, f"{model_safe}_{prompt_name}.jsonl"
        )

    existing_df = None
    completed_keys = set()
    if os.path.exists(output_path):
        existing_df = load_jsonl(output_path)
        existing_df = existing_df.drop_duplicates(
            subset=["question_id", "variant"], keep="first"
        )
        completed_keys = set(
            zip(existing_df["question_id"], existing_df["variant"])
        )
        print(f"Resuming: {len(completed_keys)} variants already completed")

    pending = []
    for _, row in df.iterrows():
        key = (row["question_id"], row["variant"])
        if key in completed_keys:
            continue

        options = row["options"]
        if len(options) != 4:
            print(f"  Skipping Q{row['question_id']} {row['variant']}: "
                  f"{len(options)} options (need 4)")
            continue

        if prompt_name == "sycophancy":
            prompt = format_prompt_sycophancy(row, options)
        else:
            prompt = format_prompt_standard(template, row["question_text"], options)

        pending.append((row, prompt))

    print(f"Pending:    {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    results = []
    errors = 0

    for batch_start in range(0, len(pending), batch_size):
        chunk = pending[batch_start:batch_start + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for _, p in chunk]

        responses = batch_completion(
            model=model, messages=messages_batch,
            max_tokens=512, temperature=0.0,
        )

        for (row, _), response in zip(chunk, responses):
            if isinstance(response, Exception):
                errors += 1
                print(f"  Error on Q{row['question_id']} {row['variant']}: {response}")
                result = {
                    "question_id": row["question_id"],
                    "variant": row["variant"],
                    "race": row["race"],
                    "gender": row["gender"],
                    "correct_answer": row["correct_answer"],
                    "model_answer": "ERROR",
                    "is_correct": False,
                    "raw_response": str(response),
                    "prompt_name": prompt_name,
                    "model": model,
                }
                if is_sycophancy:
                    result["suggested_answer"] = row["suggested_answer"]
                    result["deferred"] = False
                    result["domain"] = row["domain"]
                results.append(result)
                continue

            response_text = response.choices[0].message.content
            answer = extract_answer(response_text)

            result = {
                "question_id": row["question_id"],
                "variant": row["variant"],
                "race": row["race"],
                "gender": row["gender"],
                "correct_answer": row["correct_answer"],
                "model_answer": answer,
                "is_correct": answer == row["correct_answer"],
                "raw_response": response_text,
                "prompt_name": prompt_name,
                "model": model,
            }
            if is_sycophancy:
                result["suggested_answer"] = row["suggested_answer"]
                result["deferred"] = answer == row["suggested_answer"]
                result["domain"] = row["domain"]

            results.append(result)

        completed = len(results) + len(completed_keys)
        total = len(df)
        pct = completed / total * 100
        print(f"  Batch {batch_start // batch_size + 1}: "
              f"{completed}/{total} ({pct:.1f}%) | Errors: {errors}")
        save_results(results, output_path, existing_df)

    total_completed = len(results) + len(completed_keys)
    print(f"\nDone! {total_completed} variants completed. Errors: {errors}")
    print(f"Saved to {output_path}")

    if is_sycophancy and results:
        deferred_count = sum(1 for r in results if r.get("deferred"))
        print(f"Deference rate: {deferred_count}/{len(results)} "
              f"({deferred_count / len(results) * 100:.1f}%)")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- baseline --
    p_base = subparsers.add_parser("baseline", help="Run baseline COT inference")
    p_base.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_base.add_argument("--model", default=DEFAULT_MODEL)
    p_base.add_argument("--limit", type=int, default=None)
    p_base.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    # -- wrong-cot --
    p_wc = subparsers.add_parser("wrong-cot", help="Generate wrong COTs")
    p_wc.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_wc.add_argument("--baseline", required=True, help="Path to baseline results JSONL")
    p_wc.add_argument("--model", default=DEFAULT_MODEL)
    p_wc.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    # -- sycophancy --
    p_syc = subparsers.add_parser("sycophancy", help="Run sycophancy/experiment1 inference")
    p_syc.add_argument("--input", required=True, help="Input JSONL of question variants")
    p_syc.add_argument("--model", default=DEFAULT_MODEL)
    p_syc.add_argument("--prompt", default="sycophancy",
                        choices=["baseline", "with_explanation", "debiasing", "sycophancy"])
    p_syc.add_argument("--limit", type=int, default=None)
    p_syc.add_argument("--question-ids", type=str, default=None)
    p_syc.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(
            domain=args.domain, model=args.model,
            limit=args.limit, batch_size=args.batch_size,
        )
    elif args.command == "wrong-cot":
        run_wrong_cot(
            baseline_path=args.baseline, domain=args.domain,
            model=args.model, batch_size=args.batch_size,
        )
    elif args.command == "sycophancy":
        question_ids = None
        if args.question_ids:
            question_ids = [int(x) for x in args.question_ids.split(",")]
        run_sycophancy(
            input_path=args.input, model=args.model,
            prompt_name=args.prompt, limit=args.limit,
            question_ids=question_ids, batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
