#!/usr/bin/env bash
# Run baseline COT inference for all 3 models across both domains.
# Assigns 1 GPU per model for parallel execution.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

BACKEND=vllm
MAX_TOKENS=2048

run_model() {
  local GPU=$1 MODEL=$2 NAME=$3
  export CUDA_VISIBLE_DEVICES=$GPU
  for DOMAIN in legal medical; do
    echo "=== [$NAME] Baseline $DOMAIN ==="
    python src/dataset_generation/generate_cot.py baseline \
      --domain $DOMAIN --backend $BACKEND --model "$MODEL" --max-tokens $MAX_TOKENS
  done
  echo "=== [$NAME] Baseline DONE ==="
}

mkdir -p logs
run_model 0 meta-llama/Llama-3.1-8B-Instruct LLaMA   > logs/baseline_llama.log 2>&1 &
run_model 1 google/gemma-2-9b-it              Gemma   > logs/baseline_gemma.log 2>&1 &
run_model 2 Qwen/Qwen2.5-7B-Instruct         Qwen    > logs/baseline_qwen.log  2>&1 &
wait
echo "========== ALL BASELINES COMPLETE =========="
