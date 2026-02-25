#!/usr/bin/env bash
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SAFE="meta-llama_Llama-3.1-8B-Instruct"
BACKEND="vllm"

source venv/bin/activate

# 1. Baseline COT
python src/dataset_generation/generate_cot.py baseline --domain legal   --backend $BACKEND --model "$MODEL"
python src/dataset_generation/generate_cot.py baseline --domain medical  --backend $BACKEND --model "$MODEL"

# 2. Wrong COT
python src/dataset_generation/generate_cot.py wrong-cot --domain legal   --backend $BACKEND --model "$MODEL" \
    --baseline "data/results/baseline_cot_${MODEL_SAFE}_legal.jsonl"
python src/dataset_generation/generate_cot.py wrong-cot --domain medical  --backend $BACKEND --model "$MODEL" \
    --baseline "data/results/baseline_cot_${MODEL_SAFE}_medical.jsonl"

# 3. Sycophancy variants
python src/dataset_generation/build_sycophancy_variants.py sycophancy --domain legal   --model-safe "$MODEL_SAFE"
python src/dataset_generation/build_sycophancy_variants.py sycophancy --domain medical  --model-safe "$MODEL_SAFE"

# 4. Sycophancy inference
python src/eval/run_sycophancy_inference.py --input data/sycophancy_variants_legal.jsonl   --backend $BACKEND --model "$MODEL" --max-tokens 2048
python src/eval/run_sycophancy_inference.py --input data/sycophancy_variants_medical.jsonl  --backend $BACKEND --model "$MODEL" --max-tokens 2048

# 5. Analysis
python src/eval/analyze_results.py sycophancy --file "data/results/sycophancy_${MODEL_SAFE}_legal.jsonl"
python src/eval/analyze_results.py sycophancy --file "data/results/sycophancy_${MODEL_SAFE}_medical.jsonl"
