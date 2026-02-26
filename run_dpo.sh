#!/usr/bin/env bash
# End-to-end DPO anti-sycophancy pipeline:
#   1. Build DPO dataset from LLaMA (sycophantic) vs Qwen (non-sycophantic)
#   2. Train LLaMA with QLoRA DPO
#   3. Merge LoRA adapter into base model
#   4. Run sycophancy eval on merged model
#   5. Analyze results
set -euo pipefail
source venv/bin/activate

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SAFE="meta-llama_Llama-3.1-8B-Instruct"
CHECKPOINT_DIR="checkpoints/dpo-llama-anti-syco"
MERGED_DIR="checkpoints/dpo-llama-anti-syco-merged"
BACKEND="vllm"

# ── 1. Build DPO dataset ─────────────────────────────────────────────────
echo "=== Step 1: Build DPO dataset ==="
python src/dataset_generation/build_dpo_dataset.py \
    --chosen-legal data/results/sycophancy_Qwen_Qwen2.5-7B-Instruct_legal.jsonl \
    --chosen-medical data/results/sycophancy_Qwen_Qwen2.5-7B-Instruct_medical.jsonl \
    --rejected-legal data/results/sycophancy_${MODEL_SAFE}_legal.jsonl \
    --rejected-medical data/results/sycophancy_${MODEL_SAFE}_medical.jsonl

# ── 2. Train ──────────────────────────────────────────────────────────────
echo "=== Step 2: DPO training ==="
python src/train_dpo.py \
    --train-file data/dpo_train.jsonl \
    --val-file data/dpo_val.jsonl \
    --model "$BASE_MODEL" \
    --output-dir "$CHECKPOINT_DIR" \
    --epochs 3 \
    --lr 5e-5 \
    --beta 0.1 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-length 2048 \
    --wandb-project dpo-anti-sycophancy

# ── 3. Merge LoRA adapter into base model ─────────────────────────────────
echo "=== Step 3: Merge LoRA adapter ==="
python -c "
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

print('Loading adapter from $CHECKPOINT_DIR...')
model = AutoPeftModelForCausalLM.from_pretrained(
    '$CHECKPOINT_DIR',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
)
print('Merging...')
model = model.merge_and_unload()
model.save_pretrained('$MERGED_DIR')

tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL')
tokenizer.save_pretrained('$MERGED_DIR')
print('Merged model saved to $MERGED_DIR')
"

# ── 4. Run sycophancy eval on merged model ─────────────────────────────────
echo "=== Step 4: Sycophancy eval on DPO model ==="
DPO_MODEL_SAFE="checkpoints_dpo-llama-anti-syco-merged"

python src/eval/run_sycophancy_inference.py \
    --input data/sycophancy_variants_legal.jsonl \
    --backend $BACKEND --model "$MERGED_DIR" --max-tokens 2048

python src/eval/run_sycophancy_inference.py \
    --input data/sycophancy_variants_medical.jsonl \
    --backend $BACKEND --model "$MERGED_DIR" --max-tokens 2048

# ── 5. Analyze ────────────────────────────────────────────────────────────
echo "=== Step 5: Analysis ==="
python src/eval/analyze_results.py sycophancy \
    --file "data/results/sycophancy_${DPO_MODEL_SAFE}_legal.jsonl"
python src/eval/analyze_results.py sycophancy \
    --file "data/results/sycophancy_${DPO_MODEL_SAFE}_medical.jsonl"

echo ""
echo "=== Done ==="
echo "Compare before/after:"
echo "  Before: data/results/analysis_sycophancy_${MODEL_SAFE}_*.csv"
echo "  After:  data/results/analysis_sycophancy_${DPO_MODEL_SAFE}_*.csv"
