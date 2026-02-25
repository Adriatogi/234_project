"""
Download MedQA USMLE 4-option dataset from HuggingFace.

Usage:
    python src/dataset_generation/download_medqa.py

Outputs:
    data/medqa.csv — 1,273 USMLE-style MCQ questions (test split)

Source: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
"""

import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

load_dotenv(os.path.join(config.PROJECT_ROOT, ".env"))

OUTPUT_PATH = os.path.join(config.DATA_DIR, "medqa.csv")


def main():
    print("Downloading MedQA USMLE 4-option from HuggingFace...")
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        token=os.getenv("HF_TOKEN"),
    )

    df = ds["test"].to_pandas()
    print(f"  Downloaded {len(df)} questions")

    df = df.drop(columns=["metamap_phrases"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    print("\n--- Sample row ---")
    row = df.iloc[0]
    print(f"  Question:   {str(row['question'])[:200]}...")
    print(f"  Options:    {row['options']}")
    print(f"  Answer:     {row['answer_idx']} ({row['answer']})")
    print(f"  USMLE step: {row['meta_info']}")

    print(f"\n--- Dataset stats ---")
    print(f"  Total questions: {len(df)}")
    print(f"  By step: {df['meta_info'].value_counts().to_dict()}")
    print(f"  Answer distribution: {df['answer_idx'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
