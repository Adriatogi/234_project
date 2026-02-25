"""
Filter person-centric MCQs from raw datasets.

Usage:
    python src/dataset_generation/filter_questions.py --domain legal
    python src/dataset_generation/filter_questions.py --domain medical

Outputs:
    data/filtered_questions.jsonl           (legal)
    data/filtered_medqa.jsonl               (medical)
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


FILTER_CONFIG = {
    "legal": {
        "input": "mmlu_professional_law.csv",
        "output": "filtered_questions.jsonl",
        "text_col": "centerpiece",
        "correct_fn": lambda row: ast.literal_eval(row["correct_options"])[0],
        "keywords": [
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
        ],
        "exclude": [
            r"^which of the following",
            r"^under the",
            r"^the \w+ amendment",
            r"^according to",
        ],
    },
    "medical": {
        "input": "medqa.csv",
        "output": "filtered_medqa.jsonl",
        "text_col": "question",
        "correct_fn": lambda row: row["answer_idx"],
        "keywords": [
            "year-old", "years old", "month-old", "months old",
            "day-old", "week-old",
            "a man", "a woman", "a boy", "a girl",
            "the man", "the woman", "the boy", "the girl",
            "a male", "a female",
            "patient", "infant", "newborn", "neonate",
            "child", "adolescent", "teenager",
            "presents to", "brought to", "comes to",
            "is admitted", "is referred",
        ],
        "exclude": [
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
        ],
    },
}


def filter_domain(domain: str):
    cfg = FILTER_CONFIG[domain]
    input_path = os.path.join(config.DATA_DIR, cfg["input"])
    output_path = os.path.join(config.DATA_DIR, cfg["output"])

    df = pd.read_csv(input_path)
    print(f"Total {domain} questions: {len(df)}")

    text_lower_cache = df[cfg["text_col"]].str.lower()
    has_person = text_lower_cache.apply(
        lambda t: any(kw in t for kw in cfg["keywords"])
    )
    excluded = text_lower_cache.str.strip().apply(
        lambda t: any(re.match(pat, t) for pat in cfg["exclude"])
    )
    keep = has_person & ~excluded

    rows = []
    for idx, row in df[keep].iterrows():
        rows.append({
            "question_id": idx,
            "question_text": row[cfg["text_col"]],
            "options": config.parse_options(row["options"]),
            "correct_answer": cfg["correct_fn"](row),
        })

    out_df = pd.DataFrame(rows)
    config.save_jsonl(out_df, output_path)

    print(f"Filtered: {len(out_df)} questions")
    print(f"Saved to {output_path}")

    for i in range(min(3, len(out_df))):
        r = out_df.iloc[i]
        print(f"\n  Q{r['question_id']} ({r['correct_answer']}): {r['question_text'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Filter person-centric MCQs")
    parser.add_argument("--domain", required=True, choices=["legal", "medical"])
    args = parser.parse_args()
    filter_domain(args.domain)


if __name__ == "__main__":
    main()
