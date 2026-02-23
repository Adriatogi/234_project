"""
Download MedQA USMLE 4-option dataset from HuggingFace.

Outputs:
    medqa.csv — 1,273 USMLE-style MCQ questions (test split)

Each row contains:
    - question: clinical vignette / question stem
    - options: dict of 4 answer choices {"A": "...", "B": "...", "C": "...", "D": "..."}
    - answer: correct answer text
    - answer_idx: correct answer letter (A/B/C/D)
    - meta_info: USMLE step level (step1, step2&3)

Source: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
"""

import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "medqa.csv")


def main():
    print("Downloading MedQA USMLE 4-option from HuggingFace...")
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        token=os.getenv("HF_TOKEN"),
    )

    df = ds["test"].to_pandas()
    print(f"  Downloaded {len(df)} questions")

    # Drop metamap_phrases — large and not needed for our pipeline
    df = df.drop(columns=["metamap_phrases"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    # Print a sample
    print("\n--- Sample row ---")
    row = df.iloc[0]
    print(f"  Question:   {str(row['question'])[:200]}...")
    print(f"  Options:    {row['options']}")
    print(f"  Answer:     {row['answer_idx']} ({row['answer']})")
    print(f"  USMLE step: {row['meta_info']}")

    # Stats
    print(f"\n--- Dataset stats ---")
    print(f"  Total questions: {len(df)}")
    print(f"  By step: {df['meta_info'].value_counts().to_dict()}")
    print(f"  Answer distribution: {df['answer_idx'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
