#!/usr/bin/env bash
set -euo pipefail

MODEL="google/gemma-2-9b-it"
MODEL_SAFE="google_gemma-2-9b-it"
BATCH=512
BACKEND="vllm"

source venv/bin/activate

# 1. Baseline COT
python src/inference.py baseline --domain legal   --backend $BACKEND --model "$MODEL" --batch-size $BATCH
python src/inference.py baseline --domain medical  --backend $BACKEND --model "$MODEL" --batch-size $BATCH

# 2. Wrong COT
python src/inference.py wrong-cot --domain legal   --backend $BACKEND --model "$MODEL" --batch-size $BATCH \
    --baseline "data/results/baseline_cot_${MODEL_SAFE}_legal.jsonl"
python src/inference.py wrong-cot --domain medical  --backend $BACKEND --model "$MODEL" --batch-size $BATCH \
    --baseline "data/results/baseline_cot_${MODEL_SAFE}_medical.jsonl"

# 3. Sycophancy variants
python src/prepare.py sycophancy --domain legal   --model-safe "$MODEL_SAFE"
python src/prepare.py sycophancy --domain medical  --model-safe "$MODEL_SAFE"

# 4. Sycophancy inference
python src/inference.py sycophancy --input data/sycophancy_variants_legal.jsonl   --backend $BACKEND --model "$MODEL" --batch-size $BATCH
python src/inference.py sycophancy --input data/sycophancy_variants_medical.jsonl  --backend $BACKEND --model "$MODEL" --batch-size $BATCH

# 5. Analysis
python src/analyze.py sycophancy --file "data/results/sycophancy_${MODEL_SAFE}_legal.jsonl"
python src/analyze.py sycophancy --file "data/results/sycophancy_${MODEL_SAFE}_medical.jsonl"
