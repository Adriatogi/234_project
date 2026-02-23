# Legal LLM Bias Analysis

Measuring racial and gender bias in LLM legal reasoning using counterfactual demographic injection on MMLU Professional Law questions.

Inspired by ["How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?"](https://arxiv.org/html/2410.16574v1), adapted from the medical domain to legal reasoning.

## Approach

1. **Filter** 1,183 bar-exam-style MCQs from MMLU Professional Law that contain person-centric fact patterns
2. **Inject** counterfactual demographics (5 races x 2 genders = 10 variants + 1 neutral = 13,013 total variants)
3. **Run inference** through LLMs via [litellm](https://docs.litellm.ai/) (Together AI, or local via vllm-mlx)
4. **Analyze** whether protected characteristics cause models to change their answers

Demographics tested: White, Black, Asian, Hispanic, Arab x Male, Female

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd 234_project

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your TOGETHERAI_API_KEY
```

## Usage

### 1. Download datasets

```bash
python data/download_mmlu_professional_law.py   # MMLU Professional Law (1,533 MCQs)
python data/download_legalbench.py               # LegalBench (162 legal tasks)
python data/download_medquad.py                  # MedQuAD (47k medical QA pairs)
```

### 2. Prepare counterfactual variants

```bash
python src/filter_questions.py          # Filter for person-centric questions
python src/inject_demographics.py       # Create demographic variants
```

### 3. Run inference

```bash
# Test run (2 questions, 22 variants)
python src/run_inference.py --prompt baseline --limit 2

# Full run with baseline prompt
python src/run_inference.py --prompt baseline

# With explanation prompt
python src/run_inference.py --prompt with_explanation

# With debiasing prompt
python src/run_inference.py --prompt debiasing

# Use a different model
python src/run_inference.py --model together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 4. Analyze results

```bash
python src/analyze_results.py                    # Analyze all result files
python src/analyze_results.py --file data/results/specific_file.csv
```

## Inference Providers

**Together AI (cloud):** Default. Set `TOGETHERAI_API_KEY` in `.env`. Default model: `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`.

**Local via vllm-mlx (Apple Silicon):** Install `vllm-mlx`, start a local server, then point litellm at it:
```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Llama-3.1-8B-Instruct-4bit
# In another terminal:
python src/run_inference.py --model openai/mlx-community/Llama-3.1-8B-Instruct-4bit
```

## Project Structure

```
234_project/
├── .env.example              # API key template
├── requirements.txt          # Python dependencies
├── data/
│   ├── download_*.py         # Dataset download scripts
│   ├── mmlu_professional_law.csv
│   ├── filtered_questions.csv
│   ├── counterfactual_questions.csv
│   └── results/              # Inference outputs and analysis
├── src/
│   ├── filter_questions.py   # Filter injectable questions
│   ├── inject_demographics.py # Create counterfactual variants
│   ├── run_inference.py      # LLM inference via litellm
│   ├── analyze_results.py    # Bias analysis and statistics
│   └── prompts/
│       ├── baseline.txt
│       ├── with_explanation.txt
│       └── debiasing.txt
├── diagnose_treat_bias_llm/  # Reference: medical bias paper code
├── legalbench/               # Reference: legal reasoning benchmark
└── MedQuAD/                  # Reference: medical QA dataset
```

## Reference Datasets

- **MMLU Professional Law** — 1,533 bar-exam MCQs ([HuggingFace](https://huggingface.co/datasets/brucewlee1/mmlu-professional-law))
- **LegalBench** — 162 legal reasoning tasks ([GitHub](https://github.com/HazyResearch/legalbench))
- **MedQuAD** — 47k medical QA pairs ([GitHub](https://github.com/abachaa/MedQuAD))
