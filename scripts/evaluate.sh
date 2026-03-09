#!/usr/bin/env bash
# Generate all comparison tables (base vs DPO) on held-out evaluation set.
#
# Usage:
#   bash scripts/evaluate.sh              # held-out test set (default)
#   bash scripts/evaluate.sh train        # training set only
#   bash scripts/evaluate.sh all          # all questions
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

SPLIT="${1:-test}"

echo "============================================================"
echo "  Evaluation tables (split=${SPLIT})"
echo "============================================================"

echo ""
echo "1. Single-Turn Regressive — COMBINED"
echo "------------------------------------------------------------"
python src/single_turn_eval/analyze_results.py comparison-table \
  --tsv --combined --split "$SPLIT"

echo ""
echo "2. Single-Turn Progressive — COMBINED"
echo "------------------------------------------------------------"
python src/single_turn_eval/analyze_results.py comparison-table \
  --tsv --combined --direction progressive --split "$SPLIT"

echo ""
echo "3. Multi-Turn Regressive — COMBINED"
echo "------------------------------------------------------------"
python src/multi_turn_eval/analyze_results.py comparison-table \
  --tsv --combined --direction regressive --split "$SPLIT"

echo ""
echo "4. Multi-Turn Progressive — COMBINED"
echo "------------------------------------------------------------"
python src/multi_turn_eval/analyze_results.py comparison-table \
  --tsv --combined --direction progressive --split "$SPLIT"

echo ""
echo "============================================================"
echo "  Accuracy Tables (split=${SPLIT})"
echo "============================================================"

echo ""
echo "5. Single-Turn Accuracy — COMBINED"
echo "------------------------------------------------------------"
python src/single_turn_eval/analyze_results.py accuracy-table \
  --tsv --combined --split "$SPLIT"

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
