#!/usr/bin/env bash
set -euo pipefail
cd /matx/u/agamarra/234_project
source venv/bin/activate

# ──────────────────────────────────────────────────────────────
# GPU 0 — LLaMA: DPO train → merge → inference (all conditions)
# ──────────────────────────────────────────────────────────────
run_llama() {
  export CUDA_VISIBLE_DEVICES=0
  MODEL=meta-llama/Llama-3.1-8B-Instruct
  SAFE=meta-llama_Llama-3.1-8B-Instruct
  CKPT=checkpoints/dpo-llama-selfplay-nocot
  MERGED=checkpoints/dpo-llama-selfplay-nocot-merged

  echo "=== [LLaMA] Step 6: DPO Training ==="
  python src/train_dpo.py \
    --train-file data/dpo_selfplay_llama_nocot_train.jsonl \
    --val-file data/dpo_selfplay_llama_nocot_val.jsonl \
    --model $MODEL \
    --output-dir $CKPT \
    --epochs 3 --eval-steps 10 --grad-accum 1 \
    --wandb-project dpo-selfplay --run-name selfplay-nocot-llama-v5

  echo "=== [LLaMA] Step 7a: Merge LoRA ==="
  python src/single_turn_eval/merge_lora.py \
    --adapter $CKPT --output $MERGED

  echo "=== [LLaMA] Step 7b: DPO Model Inference ==="
  for DOMAIN in legal medical; do
    echo "--- [LLaMA-DPO] ST regressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_regressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [LLaMA-DPO] ST progressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_progressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [LLaMA-DPO] MT $DOMAIN ---"
    python src/multi_turn_eval/run_multiturn_inference.py \
      --input data/variants/multiturn_variants_${SAFE}_${DOMAIN}.jsonl \
      --baseline data/results/baseline/baseline_cot_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --backend vllm --max-tokens 2048
  done

  echo "=== [LLaMA] ALL DONE ==="
}

# ──────────────────────────────────────────────────────────────
# GPU 2 — Qwen: DPO train → merge → inference (all conditions)
# ──────────────────────────────────────────────────────────────
run_qwen() {
  export CUDA_VISIBLE_DEVICES=2
  MODEL=Qwen/Qwen2.5-7B-Instruct
  SAFE=Qwen_Qwen2.5-7B-Instruct
  CKPT=checkpoints/dpo-qwen-selfplay-nocot
  MERGED=checkpoints/dpo-qwen-selfplay-nocot-merged

  echo "=== [Qwen] Step 6: DPO Training ==="
  python src/train_dpo.py \
    --train-file data/dpo_selfplay_qwen_nocot_train.jsonl \
    --val-file data/dpo_selfplay_qwen_nocot_val.jsonl \
    --model $MODEL \
    --output-dir $CKPT \
    --epochs 3 --eval-steps 10 --grad-accum 1 \
    --wandb-project dpo-selfplay --run-name selfplay-nocot-qwen-v5

  echo "=== [Qwen] Step 7a: Merge LoRA ==="
  python src/single_turn_eval/merge_lora.py \
    --adapter $CKPT --output $MERGED

  echo "=== [Qwen] Step 7b: DPO Model Inference ==="
  for DOMAIN in legal medical; do
    echo "--- [Qwen-DPO] ST regressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_regressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [Qwen-DPO] ST progressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_progressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [Qwen-DPO] MT $DOMAIN ---"
    python src/multi_turn_eval/run_multiturn_inference.py \
      --input data/variants/multiturn_variants_${SAFE}_${DOMAIN}.jsonl \
      --baseline data/results/baseline/baseline_cot_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --backend vllm --max-tokens 2048
  done

  echo "=== [Qwen] ALL DONE ==="
}

# ──────────────────────────────────────────────────────────────
# GPU 1 — Gemma: DPO train → merge → inference (all conditions)
# ──────────────────────────────────────────────────────────────
run_gemma() {
  export CUDA_VISIBLE_DEVICES=1
  MODEL=google/gemma-2-9b-it
  SAFE=google_gemma-2-9b-it
  CKPT=checkpoints/dpo-gemma-selfplay-nocot
  MERGED=checkpoints/dpo-gemma-selfplay-nocot-merged

  echo "=== [Gemma] Step 6: DPO Training ==="
  python src/train_dpo.py \
    --train-file data/dpo_selfplay_gemma_nocot_train.jsonl \
    --val-file data/dpo_selfplay_gemma_nocot_val.jsonl \
    --model $MODEL \
    --output-dir $CKPT \
    --epochs 3 --eval-steps 10 --grad-accum 1 \
    --wandb-project dpo-selfplay --run-name selfplay-nocot-gemma-v5

  echo "=== [Gemma] Step 7a: Merge LoRA ==="
  python src/single_turn_eval/merge_lora.py \
    --adapter $CKPT --output $MERGED

  echo "=== [Gemma] Step 7b: DPO Model Inference ==="
  for DOMAIN in legal medical; do
    echo "--- [Gemma-DPO] ST regressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_regressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [Gemma-DPO] ST progressive $DOMAIN ---"
    python src/single_turn_eval/run_sycophancy_inference.py \
      --input data/variants/sycophancy_progressive_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --prompt sycophancy_no_cot --backend vllm --max-tokens 2048

    echo "--- [Gemma-DPO] MT $DOMAIN ---"
    python src/multi_turn_eval/run_multiturn_inference.py \
      --input data/variants/multiturn_variants_${SAFE}_${DOMAIN}.jsonl \
      --baseline data/results/baseline/baseline_cot_${SAFE}_${DOMAIN}.jsonl \
      --model $MERGED --backend vllm --max-tokens 2048
  done

  echo "=== [Gemma] ALL DONE ==="
}

# Launch all 3 in parallel, each with its own log file
mkdir -p logs
run_llama > logs/dpo_llama.log 2>&1 &
run_gemma > logs/dpo_gemma.log 2>&1 &
run_qwen  > logs/dpo_qwen.log  2>&1 &
wait
echo "========== ALL 3 MODELS COMPLETE =========="
