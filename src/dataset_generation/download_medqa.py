"""
Download MedQA USMLE 4-option dataset from HuggingFace.

Usage:
    python src/dataset_generation/download_medqa.py
    python src/dataset_generation/download_medqa.py --include-train 1000

Outputs:
    data/medqa.csv — USMLE-style MCQ questions (test split + optional train sample)

Source: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
"""

import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

load_dotenv(os.path.join(config.PROJECT_ROOT, ".env"))

OUTPUT_PATH = os.path.join(config.DATA_DIR, "medqa.csv")


def main():
    parser = argparse.ArgumentParser(description="Download MedQA USMLE 4-option dataset")
    parser.add_argument("--include-train", type=int, default=0,
                        help="Sample N questions from train split and append (seed 42, deduplicated)")
    args = parser.parse_args()

    print("Downloading MedQA USMLE 4-option from HuggingFace...")
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        token=os.getenv("HF_TOKEN"),
    )

    test_df = ds["test"].to_pandas()
    print(f"  Test split: {len(test_df)} questions")
    test_df = test_df.drop(columns=["metamap_phrases"])

    if args.include_train > 0:
        train_df = ds["train"].to_pandas()
        print(f"  Train split: {len(train_df)} questions")
        train_df = train_df.drop(columns=["metamap_phrases"])

        test_questions = set(test_df["question"].str.strip())
        dedup_mask = ~train_df["question"].str.strip().isin(test_questions)
        train_df = train_df[dedup_mask].reset_index(drop=True)
        print(f"  After dedup vs test: {len(train_df)} unique train questions")

        n_sample = min(args.include_train, len(train_df))
        train_sample = train_df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(train_sample)} from train (seed 42)")

        df = pd.concat([test_df, train_sample], ignore_index=True)
        print(f"  Combined: {len(df)} total questions (test={len(test_df)}, train={len(train_sample)})")
    else:
        df = test_df

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    print(f"\n--- Dataset stats ---")
    print(f"  Total questions: {len(df)}")
    print(f"  Answer distribution: {df['answer_idx'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
