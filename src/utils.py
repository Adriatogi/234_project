"""
Shared utility functions and constants for the sycophancy pipeline.

All scripts import from here instead of duplicating common logic.
"""

import ast
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# ---------------------------------------------------------------------------
# Model / inference constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_BATCH_SIZE = 20
VALID_ANSWERS = {"A", "B", "C", "D"}
LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a JSONL file."""
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_results(
    new_results: list[dict],
    output_path: str,
    existing_df: pd.DataFrame | None,
) -> None:
    """Write all results to JSONL: original existing data + all new results."""
    new_df = pd.DataFrame(new_results)
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    save_jsonl(combined, output_path)


# ---------------------------------------------------------------------------
# Model name helpers
# ---------------------------------------------------------------------------

def safe_model_name(model: str) -> str:
    """Convert model string to a filesystem-safe name."""
    return model.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str:
    """Extract the answer letter (A/B/C/D) from the model's response."""
    text = response_text.strip()

    # Try "Answer: X" pattern first (most structured)
    match = re.search(r"Answer:\s*([A-Da-d])", text)
    if match:
        return match.group(1).upper()

    # Try first character if it's a valid answer
    if text[:1].upper() in VALID_ANSWERS:
        return text[:1].upper()

    # Try first standalone A-D letter
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    return "INVALID"


def letter_to_index(letter: str) -> int:
    """Convert A/B/C/D to 0/1/2/3."""
    return LETTERS.index(letter)


# ---------------------------------------------------------------------------
# Options parsing (for raw HuggingFace CSVs)
# ---------------------------------------------------------------------------

def parse_options(options_val) -> list[str]:
    """Parse options from various formats into a list of 4 strings.

    Handles:
        - Python list (already parsed)
        - Python dict with A/B/C/D keys (MedQA format)
        - String repr of list with whitespace separators (MMLU format)
        - String repr of dict (MedQA CSV format)
    """
    if isinstance(options_val, list):
        return options_val
    if isinstance(options_val, dict):
        return [options_val["A"], options_val["B"], options_val["C"], options_val["D"]]

    s = str(options_val)
    # Fix MMLU whitespace-separated format: ['opt1' 'opt2'] -> ['opt1', 'opt2']
    fixed = re.sub(r"""(['"]) \s+ (['"])""", r'\1, \2', s, flags=re.VERBOSE)
    parsed = ast.literal_eval(fixed)

    if isinstance(parsed, dict):
        return [parsed["A"], parsed["B"], parsed["C"], parsed["D"]]
    return parsed


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramer's V from a contingency table."""
    if contingency_table.size == 0 or contingency_table.sum() == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))
