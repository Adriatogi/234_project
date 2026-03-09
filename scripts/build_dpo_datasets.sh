#!/usr/bin/env bash
# Build DPO self-play datasets for all 3 models.
# Uses 350 training questions per direction per domain (300 held out for eval).
# No GPU needed.
#
# Pass --legacy-pairs to reproduce v3 behavior (deference always treated as bad).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_QUESTIONS=350
EXTRA_ARGS="${*}"  # forward any extra flags (e.g. --legacy-pairs)

for SHORT in llama gemma qwen; do
  case $SHORT in
    llama) MODEL=meta-llama/Llama-3.1-8B-Instruct ;;
    gemma) MODEL=google/gemma-2-9b-it ;;
    qwen)  MODEL=Qwen/Qwen2.5-7B-Instruct ;;
  esac

  echo "========== $SHORT =========="
  python src/dataset_generation/build_dpo_dataset.py self-play \
    --model "$MODEL" --prompt no-cot --max-questions $MAX_QUESTIONS \
    --output-prefix "dpo_selfplay_${SHORT}_nocot" $EXTRA_ARGS
  echo
done
