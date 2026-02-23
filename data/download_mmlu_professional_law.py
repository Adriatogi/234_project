"""
Download MMLU Professional Law dataset from HuggingFace.

Outputs:
    mmlu_professional_law.csv — 1,533 bar-exam-style MCQ questions

Each row contains:
    - centerpiece: the legal fact pattern / question stem
    - options: list of 4 answer choices
    - correct_options: list with the correct letter (e.g. ['B'])
    - correct_options_idx: list with the correct 0-based index (e.g. [1])
    - correct_options_literal: list with the full text of the correct answer

Source: https://huggingface.co/datasets/brucewlee1/mmlu-professional-law
"""

import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mmlu_professional_law.csv")


def main():
    print("Downloading MMLU Professional Law from HuggingFace...")
    ds = load_dataset("brucewlee1/mmlu-professional-law", token=os.getenv("HF_TOKEN"))

    df = ds["test"].to_pandas()
    print(f"  Downloaded {len(df)} questions")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    # Print a sample
    print("\n--- Sample row ---")
    row = df.iloc[0]
    print(f"  Question: {str(row['centerpiece'])[:200]}...")
    print(f"  Options:  {str(row['options'])[:200]}...")
    print(f"  Answer:   {row['correct_options']}")


if __name__ == "__main__":
    main()
