"""
Download MMLU Professional Law dataset from HuggingFace.

Usage:
    python src/dataset_generation/download_mmlu_professional_law.py

Outputs:
    data/mmlu_professional_law.csv — 1,533 bar-exam-style MCQ questions

Source: https://huggingface.co/datasets/brucewlee1/mmlu-professional-law
"""

import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

load_dotenv(os.path.join(config.PROJECT_ROOT, ".env"))

OUTPUT_PATH = os.path.join(config.DATA_DIR, "mmlu_professional_law.csv")


def main():
    print("Downloading MMLU Professional Law from HuggingFace...")
    ds = load_dataset("brucewlee1/mmlu-professional-law", token=os.getenv("HF_TOKEN"))

    df = ds["test"].to_pandas()
    print(f"  Downloaded {len(df)} questions")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    print("\n--- Sample row ---")
    row = df.iloc[0]
    print(f"  Question: {str(row['centerpiece'])[:200]}...")
    print(f"  Options:  {str(row['options'])[:200]}...")
    print(f"  Answer:   {row['correct_options']}")


if __name__ == "__main__":
    main()
