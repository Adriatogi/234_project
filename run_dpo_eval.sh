#!/usr/bin/env bash
# Run merge + eval for the DPO-trained model (use after training is done).
set -euo pipefail
source venv/bin/activate

ADAPTER="checkpoints/dpo-llama-anti-syco/checkpoint-459"
MERGED="checkpoints/dpo-llama-anti-syco-merged"
MODEL_SAFE="meta-llama_Llama-3.1-8B-Instruct"
DPO_MODEL_SAFE="checkpoints_dpo-llama-anti-syco-merged"

# 1. Merge LoRA adapter
echo "=== Merge LoRA adapter ==="
python src/single_turn_eval/merge_lora.py --adapter "$ADAPTER" --output "$MERGED"

# 2. Sycophancy inference on merged model
echo "=== Sycophancy inference ==="
python src/single_turn_eval/run_sycophancy_inference.py \
    --input data/variants/sycophancy_variants_legal.jsonl \
    --backend vllm --model "$MERGED" --max-tokens 2048

python src/single_turn_eval/run_sycophancy_inference.py \
    --input data/variants/sycophancy_variants_medical.jsonl \
    --backend vllm --model "$MERGED" --max-tokens 2048

# 3. Analyze
echo "=== Analysis ==="
python src/single_turn_eval/analyze_results.py sycophancy --file "data/results/single_turn/sycophancy_${DPO_MODEL_SAFE}_legal.jsonl"
python src/single_turn_eval/analyze_results.py sycophancy --file "data/results/single_turn/sycophancy_${DPO_MODEL_SAFE}_medical.jsonl"

echo ""
echo "=== Done ==="
echo "Compare before/after:"
echo "  Before: data/results/single_turn/analysis_sycophancy_${MODEL_SAFE}_*.csv"
echo "  After:  data/results/single_turn/analysis_sycophancy_${DPO_MODEL_SAFE}_*.csv"
