#!/usr/bin/env bash
# Run single-turn and multi-turn inference for base models (not DPO).
# Assigns 1 GPU per model for parallel execution.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_TOKENS=2048
BACKEND=vllm

run_model() {
  local GPU=$1 MODEL=$2 SAFE=$3 NAME=$4
  export CUDA_VISIBLE_DEVICES=$GPU

  for DOMAIN in legal medical; do
    echo "--- [$NAME] ST regressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_regressive_${SAFE}_${DOMAIN}.jsonl \
      --model "$MODEL" --prompt sycophancy_no_cot --backend $BACKEND --max-tokens $MAX_TOKENS

    echo "--- [$NAME] ST progressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_progressive_${SAFE}_${DOMAIN}.jsonl \
      --model "$MODEL" --prompt sycophancy_no_cot --backend $BACKEND --max-tokens $MAX_TOKENS

    echo "--- [$NAME] MT $DOMAIN ---"
    python src/multi_turn_eval/run_multiturn_inference.py \
      --input data/variants/multiturn_variants_${SAFE}_${DOMAIN}.jsonl \
      --baseline data/results/baseline/baseline_cot_${SAFE}_${DOMAIN}.jsonl \
      --model "$MODEL" --backend $BACKEND --max-tokens $MAX_TOKENS
  done

  echo "=== [$NAME] ALL INFERENCE DONE ==="
}

mkdir -p logs
run_model 0 meta-llama/Llama-3.1-8B-Instruct meta-llama_Llama-3.1-8B-Instruct LLaMA > logs/inference_llama.log 2>&1 &
run_model 1 google/gemma-2-9b-it              google_gemma-2-9b-it              Gemma > logs/inference_gemma.log 2>&1 &
run_model 2 Qwen/Qwen2.5-7B-Instruct         Qwen_Qwen2.5-7B-Instruct         Qwen  > logs/inference_qwen.log  2>&1 &
wait
echo "========== ALL INFERENCE COMPLETE =========="
