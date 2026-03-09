#!/usr/bin/env bash
# Build single-turn (regressive + progressive) and multi-turn variants for all 3 models.
# No GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_QUESTIONS=650

for MODEL in meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it Qwen/Qwen2.5-7B-Instruct; do
  SHORT=$(echo "$MODEL" | sed 's|.*/||')
  for DOMAIN in legal medical; do
    echo "=== [$SHORT] ST variants $DOMAIN ==="
    python src/dataset_generation/build_sycophancy_variants.py sycophancy \
      --domain $DOMAIN --model "$MODEL" --max-questions $MAX_QUESTIONS

    echo "=== [$SHORT] MT variants $DOMAIN ==="
    python src/multi_turn_eval/build_variants.py \
      --domain $DOMAIN --model "$MODEL" --max-questions $MAX_QUESTIONS
  done
done

echo "=== Variant counts ==="
wc -l data/variants/*.jsonl
