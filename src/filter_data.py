"""
Filter questions for demographic injection suitability.

Reads raw HuggingFace CSVs and outputs JSONL files with standardized columns:
    question_id, question_text, options (list), correct_answer

Usage:
    python src/filter_data.py --domain legal
    python src/filter_data.py --domain medical

Outputs:
    data/filtered_questions.jsonl   (legal)
    data/filtered_medqa.jsonl       (medical)
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, parse_options, save_jsonl

# ---------------------------------------------------------------------------
# Domain-specific configuration
# ---------------------------------------------------------------------------

LEGAL_PERSON_KEYWORDS = [
    "a man", "a woman", "a boy", "a girl", "a person",
    "the man", "the woman", "the boy", "the girl",
    "defendant", "plaintiff", "suspect", "accused",
    "petitioner", "respondent", "appellant",
    "employee", "employer", "worker",
    "landlord", "tenant", "lessee", "lessor",
    "buyer", "seller", "vendor", "purchaser",
    "driver", "motorist", "pedestrian",
    "patient", "client", "customer",
    "homeowner", "resident", "neighbor",
    "student", "teacher", "professor",
    "year-old", "years old",
]

LEGAL_EXCLUDE_PATTERNS = [
    r"^which of the following",
    r"^under the",
    r"^the \w+ amendment",
    r"^according to",
]

MEDICAL_PERSON_KEYWORDS = [
    "year-old", "years old", "month-old", "months old",
    "day-old", "week-old",
    "a man", "a woman", "a boy", "a girl",
    "the man", "the woman", "the boy", "the girl",
    "a male", "a female",
    "patient", "infant", "newborn", "neonate",
    "child", "adolescent", "teenager",
    "presents to", "brought to", "comes to",
    "is admitted", "is referred",
]

MEDICAL_EXCLUDE_PATTERNS = [
    r"^which of the following",
    r"^what is the",
    r"^what are the",
    r"^which enzyme",
    r"^which vitamin",
    r"^which receptor",
    r"^which drug",
    r"^which hormone",
    r"^which structure",
    r"^which type",
    r"^the__(most|least|best|primary)",
]


# ---------------------------------------------------------------------------
# Core filtering logic
# ---------------------------------------------------------------------------

def has_person_reference(text: str, keywords: list[str]) -> bool:
    """Check if the question text contains references to specific people."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def is_excluded(text: str, patterns: list[str]) -> bool:
    """Check if question matches an exclusion pattern (doctrine/factual only)."""
    text_lower = text.lower().strip()
    return any(re.match(pat, text_lower) for pat in patterns)


def filter_legal():
    """Filter MMLU Professional Law questions."""
    input_path = os.path.join(DATA_DIR, "mmlu_professional_law.csv")
    output_path = os.path.join(DATA_DIR, "filtered_questions.jsonl")

    df = pd.read_csv(input_path)
    print(f"Total MMLU Professional Law questions: {len(df)}")

    rows = []
    for idx, row in df.iterrows():
        options = parse_options(row["options"])
        correct = ast.literal_eval(row["correct_options"])[0]
        text = row["centerpiece"]

        if not has_person_reference(text, LEGAL_PERSON_KEYWORDS):
            continue
        if is_excluded(text, LEGAL_EXCLUDE_PATTERNS):
            continue

        rows.append({
            "question_id": idx,
            "question_text": text,
            "options": options,
            "correct_answer": correct,
        })

    out_df = pd.DataFrame(rows)
    save_jsonl(out_df, output_path)

    print(f"Questions with person references: {len(out_df)}")
    print(f"Saved to {output_path}")

    # Show a few examples
    print("\n--- Sample filtered questions ---")
    for i in range(min(5, len(out_df))):
        row = out_df.iloc[i]
        print(f"\nQ{row['question_id']} (answer: {row['correct_answer']}):")
        print(f"  {row['question_text'][:200]}...")


def filter_medical():
    """Filter MedQA questions."""
    input_path = os.path.join(DATA_DIR, "medqa.csv")
    output_path = os.path.join(DATA_DIR, "filtered_medqa.jsonl")

    df = pd.read_csv(input_path)
    print(f"Total MedQA questions: {len(df)}")

    rows = []
    for idx, row in df.iterrows():
        options = parse_options(row["options"])
        text = row["question"]

        if not has_person_reference(text, MEDICAL_PERSON_KEYWORDS):
            continue
        if is_excluded(text, MEDICAL_EXCLUDE_PATTERNS):
            continue

        rows.append({
            "question_id": idx,
            "question_text": text,
            "options": options,
            "correct_answer": row["answer_idx"],
        })

    out_df = pd.DataFrame(rows)
    save_jsonl(out_df, output_path)

    print(f"Questions with patient references: {len(out_df)}")
    print(f"Saved to {output_path}")

    # Show a few examples
    print("\n--- Sample filtered questions ---")
    for i in range(min(5, len(out_df))):
        row = out_df.iloc[i]
        print(f"\nQ{row['question_id']} (answer: {row['correct_answer']}):")
        print(f"  {row['question_text'][:200]}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter questions for demographic injection")
    parser.add_argument("--domain", required=True, choices=["legal", "medical"])
    args = parser.parse_args()

    if args.domain == "legal":
        filter_legal()
    elif args.domain == "medical":
        filter_medical()


if __name__ == "__main__":
    main()
