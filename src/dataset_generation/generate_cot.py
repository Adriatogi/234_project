"""
Baseline COT inference and wrong COT generation.

Usage:
    python src/dataset_generation/generate_cot.py baseline --domain legal --backend vllm --model meta-llama/Llama-3.1-8B-Instruct
    python src/dataset_generation/generate_cot.py wrong-cot --domain legal --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --baseline <path>

Outputs:
    data/results/baseline_cot_{model}_{domain}.jsonl
    data/wrong_cots_{model}_{domain}.jsonl
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import llm_backend
from dataset_generation.prompts import BASELINE_COT_PROMPT, WRONG_COT_PROMPT

load_dotenv(os.path.join(config.DATA_DIR, "..", ".env"))


DOMAIN_INPUTS = {
    "legal": {
        "file": os.path.join(config.DATA_DIR, "mmlu_professional_law.csv"),
        "text_col": "centerpiece",
        "correct_fn": lambda row: ast.literal_eval(row["correct_options"])[0],
    },
    "medical": {
        "file": os.path.join(config.DATA_DIR, "medqa.csv"),
        "text_col": "question",
        "correct_fn": lambda row: row["answer_idx"],
    },
}


def load_questions(domain: str) -> pd.DataFrame:
    """Load and normalize questions from raw HuggingFace CSVs."""
    cfg = DOMAIN_INPUTS[domain]
    df = pd.read_csv(cfg["file"])

    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "question_id": idx,
            "question_text": row[cfg["text_col"]],
            "options": config.parse_options(row["options"]),
            "correct_answer": cfg["correct_fn"](row),
        })
    return pd.DataFrame(rows)


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
        domain_expert=config.DOMAIN_EXPERTS[domain],
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


# ===================================================================
# BASELINE subcommand
# ===================================================================

def run_baseline(domain: str, model: str, limit: int | None, batch_size: int, backend: str, max_tokens: int = 1024):
    df = load_questions(domain)
    if limit is not None:
        df = df.head(limit)

    print(f"Domain:     {domain}")
    print(f"Model:      {model}")
    print(f"Backend:    {backend}")
    print(f"Questions:  {len(df)}")
    print(f"Batch size: {batch_size}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model_safe = config.safe_model_name(model)
    output_path = os.path.join(config.RESULTS_DIR, f"baseline_cot_{model_safe}_{domain}.jsonl")

    existing_df, completed_ids = llm_backend.load_existing(output_path, ["question_id"])

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

    def _result(row, text):
        answer = config.extract_answer(text)
        return {
            "question_id": row["question_id"],
            "question_text": row["question_text"],
            "options": row["options"],
            "correct_answer": row["correct_answer"],
            "model_answer": answer,
            "is_correct": answer == row["correct_answer"],
            "model_cot": extract_cot(text),
            "raw_response": text,
            "domain": domain,
            "model": model,
        }

    def _error(row, err):
        return {
            "question_id": row["question_id"],
            "question_text": row["question_text"],
            "options": row["options"],
            "correct_answer": row["correct_answer"],
            "model_answer": "ERROR",
            "is_correct": False,
            "model_cot": "",
            "raw_response": err,
            "domain": domain,
            "model": model,
        }

    results = llm_backend.run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=max_tokens, total_count=len(df),
        build_result=_result, build_error=_error,
    )

    correct = sum(1 for r in results if r["is_correct"])
    if results:
        print(f"Accuracy (this run): {correct}/{len(results)} "
              f"({correct / len(results) * 100:.1f}%)")


# ===================================================================
# WRONG-COT subcommand
# ===================================================================

def pick_wrong_answer(correct: str) -> str:
    """Pick the first wrong letter alphabetically (deterministic)."""
    for letter in config.LETTERS:
        if letter != correct:
            return letter
    raise ValueError(f"No wrong answer found for correct={correct}")


def run_wrong_cot(baseline_path: str, domain: str, model: str, batch_size: int, backend: str, max_tokens: int = 1024):
    df = config.load_jsonl(baseline_path)
    correct_df = df[df["is_correct"] == True].copy()

    print(f"Domain:            {domain}")
    print(f"Model:             {model}")
    print(f"Backend:           {backend}")
    print(f"Batch size:        {batch_size}")
    print(f"Baseline total:    {len(df)}")
    print(f"Answered correctly: {len(correct_df)}")

    if len(correct_df) == 0:
        print("No correct answers found — nothing to generate.")
        return

    model_safe = config.safe_model_name(model)
    output_path = os.path.join(config.DATA_DIR, f"wrong_cots_{model_safe}_{domain}.jsonl")

    existing_df, completed_ids = llm_backend.load_existing(output_path, ["question_id"])

    pending = []
    for _, row in correct_df.iterrows():
        if row["question_id"] in completed_ids:
            continue

        options = row["options"]
        wrong_letter = pick_wrong_answer(row["correct_answer"])
        wrong_text = options[config.letter_to_index(wrong_letter)]

        prompt = WRONG_COT_PROMPT.format(
            domain=config.DOMAIN_LABELS[domain],
            wrong_letter=wrong_letter,
            wrong_text=wrong_text,
            question=row["question_text"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        )
        pending.append(((row, wrong_letter, wrong_text), prompt))

    print(f"Pending:    {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    def _result(ctx, text):
        row, wrong_letter, wrong_text = ctx
        return {
            "question_id": row["question_id"],
            "question_text": row["question_text"],
            "options": row["options"],
            "correct_answer": row["correct_answer"],
            "wrong_answer": wrong_letter,
            "wrong_answer_text": wrong_text,
            "wrong_cot": text.strip(),
            "domain": domain,
            "model": model,
        }

    def _error(ctx, err):
        row, wrong_letter, wrong_text = ctx
        return {
            "question_id": row["question_id"],
            "question_text": row["question_text"],
            "options": row["options"],
            "correct_answer": row["correct_answer"],
            "wrong_answer": wrong_letter,
            "wrong_answer_text": wrong_text,
            "wrong_cot": f"ERROR: {err}",
            "domain": domain,
            "model": model,
        }

    results = llm_backend.run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=max_tokens, total_count=len(correct_df),
        build_result=_result, build_error=_error,
    )

    if results:
        ex = results[0]
        print(f"\n--- Example (Q{ex['question_id']}) ---")
        print(f"  Correct: {ex['correct_answer']}")
        print(f"  Wrong:   {ex['wrong_answer']} ({ex['wrong_answer_text'][:80]})")
        print(f"  COT:     {ex['wrong_cot'][:200]}...")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline COT + wrong COT generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_base = subparsers.add_parser("baseline", help="Run baseline COT inference")
    p_base.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_base.add_argument("--limit", type=int, default=None)
    llm_backend.add_common_args(p_base, default_max_tokens=1024)

    p_wc = subparsers.add_parser("wrong-cot", help="Generate wrong COTs")
    p_wc.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_wc.add_argument("--baseline", required=True, help="Path to baseline results JSONL")
    llm_backend.add_common_args(p_wc, default_max_tokens=1024)

    args = parser.parse_args()
    llm_backend.init_backend(args.backend, args.model)

    if args.command == "baseline":
        run_baseline(
            domain=args.domain, model=args.model,
            limit=args.limit, batch_size=args.batch_size,
            backend=args.backend, max_tokens=args.max_tokens,
        )
    elif args.command == "wrong-cot":
        run_wrong_cot(
            baseline_path=args.baseline, domain=args.domain,
            model=args.model, batch_size=args.batch_size,
            backend=args.backend, max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
