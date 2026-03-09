#!/usr/bin/env bash
# Download raw datasets and filter questions. Run once.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Downloading MMLU Professional Law ==="
python src/dataset_generation/download_mmlu_professional_law.py

echo "=== Downloading MedQA (with extra train split) ==="
python src/dataset_generation/download_medqa.py --include-train 1000

echo "=== Filtering legal questions ==="
python src/dataset_generation/filter_questions.py --domain legal

echo "=== Filtering medical questions ==="
python src/dataset_generation/filter_questions.py --domain medical

echo "=== Done. Outputs: ==="
ls -lh data/mmlu_professional_law.csv data/medqa.csv data/filtered_questions.jsonl data/filtered_medqa.jsonl
