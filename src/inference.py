"""
LLM inference pipeline with three subcommands.

Usage:
    python src/inference.py baseline --domain legal --limit 500 --batch-size 20
    python src/inference.py wrong-cot --domain legal --baseline <path> --batch-size 20
    python src/inference.py sycophancy --input <path> --batch-size 20

    # Local GPU inference via vLLM:
    python src/inference.py baseline --domain legal --backend vllm --model Qwen/Qwen2.5-7B-Instruct

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

load_dotenv(os.path.join(config.DATA_DIR, "..", ".env"))


# ===================================================================
# Backend abstraction — litellm (cloud API) vs vllm (local GPU)
# ===================================================================

class _VLLMResponse:
    """Wraps vLLM output to match the litellm/OpenAI response shape."""
    def __init__(self, text: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": text})()})]


_vllm_instance = None


def _init_backend(backend: str, model: str):
    """Initialize the inference backend. For vllm, loads the model into GPU."""
    global _vllm_instance
    if backend == "vllm":
        from vllm import LLM
        _vllm_instance = LLM(model=model)
        print(f"vLLM: loaded {model}")


def run_batch(
    backend: str,
    model: str,
    messages_batch: list[list[dict]],
    max_tokens: int,
    temperature: float,
) -> list:
    """Run a batch of chat completions through the selected backend.

    Returns a list of response objects (or Exceptions) with
    .choices[0].message.content for each input.
    """
    if backend == "litellm":
        from litellm import batch_completion
        return batch_completion(
            model=model, messages=messages_batch,
            max_tokens=max_tokens, temperature=temperature,
        )

    if backend == "vllm":
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = _vllm_instance.chat(messages_batch, params)
        return [_VLLMResponse(o.outputs[0].text) for o in outputs]

    raise ValueError(f"Unknown backend: {backend}")


# ===================================================================
# Resume + generic batch loop
# ===================================================================

def _load_existing(
    output_path: str, dedup_cols: list[str],
) -> tuple[pd.DataFrame | None, set]:
    """Load previously saved results for resume support."""
    if not os.path.exists(output_path):
        return None, set()
    existing_df = config.load_jsonl(output_path)
    existing_df = existing_df.drop_duplicates(subset=dedup_cols, keep="first")
    if len(dedup_cols) == 1:
        done = set(existing_df[dedup_cols[0]])
    else:
        done = set(zip(*(existing_df[c] for c in dedup_cols)))
    print(f"Resuming: {len(done)} already completed")
    return existing_df, done


def _run_inference(
    pending: list[tuple],
    output_path: str,
    existing_df: pd.DataFrame | None,
    backend: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    total_count: int,
    build_result: callable,
    build_error: callable,
) -> list[dict]:
    """Generic batch-inference loop with progress and incremental saves.

    pending:      [(context, prompt), ...] -- context is passed through to build fns
    build_result: (context, response_text) -> dict
    build_error:  (context, error_string) -> dict
    """
    results: list[dict] = []
    already_done = total_count - len(pending)
    errors = 0

    for batch_start in range(0, len(pending), batch_size):
        chunk = pending[batch_start:batch_start + batch_size]
        messages_batch = [[{"role": "user", "content": prompt}] for _, prompt in chunk]

        responses = run_batch(
            backend, model, messages_batch,
            max_tokens=max_tokens, temperature=0.0,
        )

        for (ctx, _), response in zip(chunk, responses):
            if isinstance(response, Exception):
                errors += 1
                results.append(build_error(ctx, str(response)))
            else:
                text = response.choices[0].message.content
                results.append(build_result(ctx, text))

        completed = len(results) + already_done
        pct = completed / total_count * 100
        print(f"  Batch {batch_start // batch_size + 1}: "
              f"{completed}/{total_count} ({pct:.1f}%) | Errors: {errors}")
        config.save_results(results, output_path, existing_df)

    print(f"\nDone! {len(results) + already_done} completed. Errors: {errors}")
    print(f"Saved to {output_path}")
    return results


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


# ===================================================================
# BASELINE subcommand — run COT inference on raw questions
# ===================================================================

def load_questions(domain: str) -> pd.DataFrame:
    """Load and normalize questions from raw HuggingFace CSVs.

    Returns DataFrame with: question_id, question_text, options (list), correct_answer
    """
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
    return config.BASELINE_COT_PROMPT.format(
        domain_expert=config.DOMAIN_EXPERTS[domain],
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def run_baseline(domain: str, model: str, limit: int | None, batch_size: int, backend: str):
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

    existing_df, completed_ids = _load_existing(output_path, ["question_id"])

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

    results = _run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=1024, total_count=len(df),
        build_result=_result, build_error=_error,
    )

    correct = sum(1 for r in results if r["is_correct"])
    if results:
        print(f"Accuracy (this run): {correct}/{len(results)} "
              f"({correct / len(results) * 100:.1f}%)")


# ===================================================================
# WRONG-COT subcommand — generate wrong chain-of-thought for correct answers
# ===================================================================

def pick_wrong_answer(correct: str) -> str:
    """Pick the first wrong letter alphabetically (deterministic)."""
    for letter in config.LETTERS:
        if letter != correct:
            return letter
    raise ValueError(f"No wrong answer found for correct={correct}")


def run_wrong_cot(baseline_path: str, domain: str, model: str, batch_size: int, backend: str):
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

    existing_df, completed_ids = _load_existing(output_path, ["question_id"])

    pending = []
    for _, row in correct_df.iterrows():
        if row["question_id"] in completed_ids:
            continue

        options = row["options"]
        wrong_letter = pick_wrong_answer(row["correct_answer"])
        wrong_text = options[config.letter_to_index(wrong_letter)]

        prompt = config.WRONG_COT_PROMPT.format(
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

    results = _run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=1024, total_count=len(correct_df),
        build_result=_result, build_error=_error,
    )

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
    return config.EXPERIMENT1_PROMPTS[prompt_name]


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
    return config.SYCOPHANCY_PROMPT.format(
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

    existing_df, completed_keys = _load_existing(
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
            "correct_answer": row["correct_answer"],
            "model_answer": answer,
            "is_correct": answer == row["correct_answer"],
            "raw_response": raw,
            "prompt_name": prompt_name,
            "model": model,
        }
        if is_sycophancy:
            result["suggested_answer"] = row["suggested_answer"]
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

    results = _run_inference(
        pending, output_path, existing_df, backend, model,
        batch_size, max_tokens=512, total_count=len(df),
        build_result=_result, build_error=_error,
    )

    if is_sycophancy and results:
        deferred_count = sum(1 for r in results if r.get("deferred"))
        print(f"Deference rate: {deferred_count}/{len(results)} "
              f"({deferred_count / len(results) * 100:.1f}%)")


# ===================================================================
# CLI
# ===================================================================

def _add_common_args(subparser):
    """Add --model, --batch-size, --backend to a subcommand."""
    subparser.add_argument("--model", default=config.DEFAULT_MODEL)
    subparser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    subparser.add_argument(
        "--backend", required=True, choices=["litellm", "vllm"],
        help="litellm for cloud APIs, vllm for local GPU inference",
    )


def main():
    parser = argparse.ArgumentParser(description="LLM inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_base = subparsers.add_parser("baseline", help="Run baseline COT inference")
    p_base.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_base.add_argument("--limit", type=int, default=None)
    _add_common_args(p_base)

    p_wc = subparsers.add_parser("wrong-cot", help="Generate wrong COTs")
    p_wc.add_argument("--domain", required=True, choices=["legal", "medical"])
    p_wc.add_argument("--baseline", required=True, help="Path to baseline results JSONL")
    _add_common_args(p_wc)

    p_syc = subparsers.add_parser("sycophancy", help="Run sycophancy/experiment1 inference")
    p_syc.add_argument("--input", required=True, help="Input JSONL of question variants")
    p_syc.add_argument("--prompt", default="sycophancy",
                        choices=["baseline", "with_explanation", "debiasing", "sycophancy"])
    p_syc.add_argument("--limit", type=int, default=None)
    p_syc.add_argument("--question-ids", type=str, default=None)
    _add_common_args(p_syc)

    args = parser.parse_args()
    _init_backend(args.backend, args.model)

    if args.command == "baseline":
        run_baseline(
            domain=args.domain, model=args.model,
            limit=args.limit, batch_size=args.batch_size,
            backend=args.backend,
        )
    elif args.command == "wrong-cot":
        run_wrong_cot(
            baseline_path=args.baseline, domain=args.domain,
            model=args.model, batch_size=args.batch_size,
            backend=args.backend,
        )
    elif args.command == "sycophancy":
        question_ids = None
        if args.question_ids:
            question_ids = [int(x) for x in args.question_ids.split(",")]
        run_sycophancy(
            input_path=args.input, model=args.model,
            prompt_name=args.prompt, limit=args.limit,
            question_ids=question_ids, batch_size=args.batch_size,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
