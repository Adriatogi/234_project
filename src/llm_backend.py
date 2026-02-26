"""
Shared LLM inference engine: backend abstraction (vLLM / litellm),
resume support, and generic batch loop.

Both dataset_generation/generate_cot.py and eval/run_sycophancy_inference.py
import from here.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ===================================================================
# Backend abstraction — litellm (cloud API) vs vllm (local GPU)
# ===================================================================

class _VLLMResponse:
    """Wraps vLLM output to match the litellm/OpenAI response shape."""
    def __init__(self, text: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": text})()})]


_vllm_instance = None


def init_backend(backend: str, model: str):
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

def load_existing(
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


def run_inference(
    pending: list[tuple],
    output_path: str,
    existing_df: pd.DataFrame | None,
    backend: str,
    model: str,
    batch_size: int | None,
    max_tokens: int,
    total_count: int,
    build_result: callable,
    build_error: callable,
) -> list[dict]:
    """Generic batch-inference loop with progress and incremental saves.

    pending:      [(context, prompt), ...] -- context is passed through to build fns
    build_result: (context, response_text) -> dict
    build_error:  (context, error_string) -> dict
    batch_size:   None sends everything in one shot (vLLM handles scheduling)
    """
    results: list[dict] = []
    already_done = total_count - len(pending)
    errors = 0
    effective_batch = batch_size or len(pending)

    for batch_start in range(0, len(pending), effective_batch):
        chunk = pending[batch_start:batch_start + effective_batch]
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
        print(f"  Progress: {completed}/{total_count} ({pct:.1f}%) | Errors: {errors}")
        config.save_results(results, output_path, existing_df)

    print(f"\nDone! {len(results) + already_done} completed. Errors: {errors}")
    print(f"Saved to {output_path}")
    return results


def run_inference_rebuttal(
    pending: list[tuple],
    output_path: str,
    existing_df: pd.DataFrame | None,
    backend: str,
    model: str,
    batch_size: int | None,
    max_tokens: int,
    total_count: int,
    build_result: callable,
    build_error: callable,
) -> list[dict]:
    """Batch-inference loop for rebuttal conversations.

    Like run_inference, but pending items carry pre-built message lists
    instead of single prompt strings:
        pending = [(context, [{"role": ..., "content": ...}, ...]), ...]
    """
    results: list[dict] = []
    already_done = total_count - len(pending)
    errors = 0
    effective_batch = batch_size or len(pending)

    for batch_start in range(0, len(pending), effective_batch):
        chunk = pending[batch_start:batch_start + effective_batch]
        messages_batch = [messages for _, messages in chunk]

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
        print(f"  Progress: {completed}/{total_count} ({pct:.1f}%) | Errors: {errors}")
        config.save_results(results, output_path, existing_df)

    print(f"\nDone! {len(results) + already_done} completed. Errors: {errors}")
    print(f"Saved to {output_path}")
    return results


def add_common_args(subparser, default_max_tokens: int):
    """Add --model, --batch-size, --backend, --max-tokens to a subcommand."""
    subparser.add_argument("--model", default=config.DEFAULT_MODEL)
    subparser.add_argument("--batch-size", type=int, default=None,
                           help="Batch size for inference. Omit to send all at once (vLLM).")
    subparser.add_argument("--max-tokens", type=int, default=default_max_tokens)
    subparser.add_argument(
        "--backend", required=True, choices=["litellm", "vllm"],
        help="litellm for cloud APIs, vllm for local GPU inference",
    )
