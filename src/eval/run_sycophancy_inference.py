"""
Run sycophancy and experiment 1 inference on pre-built variants.

Usage:
    python src/eval/run_sycophancy_inference.py --input data/sycophancy_variants_legal.jsonl --backend vllm --model meta-llama/Llama-3.1-8B-Instruct
    python src/eval/run_sycophancy_inference.py --input data/counterfactual_questions_legal.jsonl --prompt baseline --backend litellm --model together_ai/...

Output:
    data/results/sycophancy_{model}_{domain}.jsonl
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import llm_backend
from eval.prompts import SYCOPHANCY_PROMPT, EXPERIMENT1_PROMPTS

load_dotenv(os.path.join(config.DATA_DIR, "..", ".env"))


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
        domain_expert=config.DOMAIN_EXPERTS[domain],
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
    backend: str,
    max_tokens: int = 512,
):
    df = config.load_jsonl(input_path)
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
    print(f"Backend:    {backend}")
    print(f"Prompt:     {prompt_name}")
    print(f"Input:      {os.path.basename(input_path)}")
    print(f"Sycophancy: {is_sycophancy}")
    print(f"Batch size: {batch_size}")
    print(f"Variants:   {len(df)}")
    print(f"Questions:  {df['question_id'].nunique()}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model_safe = config.safe_model_name(model)

    if is_sycophancy and "domain" in df.columns:
        domain = df["domain"].iloc[0]
        output_path = os.path.join(
            config.RESULTS_DIR, f"sycophancy_{model_safe}_{domain}.jsonl"
        )
    else:
        output_path = os.path.join(
            config.RESULTS_DIR, f"{model_safe}_{prompt_name}.jsonl"
        )

    existing_df, completed_keys = llm_backend.load_existing(
        output_path, ["question_id", "variant"]
    )

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

    print(f"Pending:           {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    def _base_result(row, answer, raw):
        result = {
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
            "prompt_name": prompt_name,
            "model": model,
        }
        if is_sycophancy:
            result["suggested_answer"] = row["suggested_answer"]
            result["suggested_cot"] = row["suggested_cot"]
            result["authority_description"] = row["authority_description"]
            result["deferred"] = answer == row["suggested_answer"]
            result["domain"] = row["domain"]
        return result

    def _result(row, text):
        return _base_result(row, config.extract_answer(text), text)

    def _error(row, err):
        result = _base_result(row, "ERROR", err)
        result["is_correct"] = False
        if is_sycophancy:
            result["deferred"] = False
        return result

    results = llm_backend.run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=max_tokens, total_count=len(df),
        build_result=_result, build_error=_error,
    )

    if is_sycophancy and results:
        deferred_count = sum(1 for r in results if r.get("deferred"))
        print(f"Deference rate: {deferred_count}/{len(results)} "
              f"({deferred_count / len(results) * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy/experiment1 inference")
    parser.add_argument("--input", required=True, help="Input JSONL of question variants")
    parser.add_argument("--prompt", default="sycophancy",
                        choices=["baseline", "with_explanation", "debiasing", "sycophancy"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", type=str, default=None)
    llm_backend.add_common_args(parser, default_max_tokens=512)

    args = parser.parse_args()
    llm_backend.init_backend(args.backend, args.model)

    question_ids = None
    if args.question_ids:
        question_ids = [int(x) for x in args.question_ids.split(",")]

    run_sycophancy(
        input_path=args.input, model=args.model,
        prompt_name=args.prompt, limit=args.limit,
        question_ids=question_ids, batch_size=args.batch_size,
        backend=args.backend, max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
