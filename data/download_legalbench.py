"""
Download LegalBench dataset from GitHub and consolidate into CSVs.

Outputs:
    legalbench/                — one CSV per task (162 tasks)
    legalbench_summary.csv    — summary of all tasks with row counts and column info

LegalBench contains 162 legal reasoning tasks from Stanford/HazyResearch.
Most tasks are binary classification (Yes/No) over legal text.

Source: https://github.com/HazyResearch/legalbench
"""

import os
import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGALBENCH_DIR = os.path.join(OUTPUT_DIR, "legalbench")
REPO_URL = "https://github.com/HazyResearch/legalbench.git"
CLONE_DIR = os.path.join(OUTPUT_DIR, "_legalbench_repo")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "legalbench_summary.csv")


def main():
    # Clone the repo if not already present
    if not os.path.exists(os.path.join(CLONE_DIR, "tasks")):
        print(f"Cloning LegalBench repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
            check=True,
        )
    else:
        print(f"LegalBench repo already cloned at {CLONE_DIR}")

    tasks_dir = os.path.join(CLONE_DIR, "tasks")
    os.makedirs(LEGALBENCH_DIR, exist_ok=True)

    summary_rows = []
    task_count = 0

    for task_name in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        train_tsv = os.path.join(task_path, "train.tsv")
        if not os.path.exists(train_tsv):
            continue

        df = pd.read_csv(train_tsv, sep="\t")
        out_path = os.path.join(LEGALBENCH_DIR, f"{task_name}.csv")
        df.to_csv(out_path, index=False)

        summary_rows.append({
            "task": task_name,
            "rows": len(df),
            "columns": ", ".join(df.columns),
            "answer_values": ", ".join(df["answer"].unique().astype(str)[:5]) if "answer" in df.columns else "N/A",
        })
        task_count += 1

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"\n  Converted {task_count} tasks to CSV in {LEGALBENCH_DIR}/")
    print(f"  Summary saved to {SUMMARY_PATH}")
    print(f"\n--- Task breakdown ---")
    print(f"  Total tasks: {task_count}")
    print(f"  Total rows:  {summary_df['rows'].sum()}")
    print(f"\n--- Sample tasks ---")
    for _, row in summary_df.head(5).iterrows():
        print(f"  {row['task']}: {row['rows']} rows [{row['answer_values']}]")


if __name__ == "__main__":
    main()
