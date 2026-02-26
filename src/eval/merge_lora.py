"""
Merge a PEFT LoRA adapter into its base model and save the result.

Usage:
    python src/eval/merge_lora.py \
        --adapter checkpoints/dpo-llama-anti-syco/checkpoint-459 \
        --output checkpoints/dpo-llama-anti-syco-merged

    python src/eval/merge_lora.py \
        --adapter checkpoints/dpo-llama-anti-syco/checkpoint-459 \
        --output checkpoints/dpo-llama-anti-syco-merged \
        --base meta-llama/Llama-3.1-8B-Instruct
"""

import argparse

import torch
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    parser.add_argument("--base", default=None,
                        help="Base model name/path (auto-detected from adapter config if omitted)")
    args = parser.parse_args()

    if args.base is None:
        peft_config = PeftConfig.from_pretrained(args.adapter)
        base_model = peft_config.base_model_name_or_path
    else:
        base_model = args.base

    print(f"Base model:  {base_model}")
    print(f"Adapter:     {args.adapter}")
    print(f"Output:      {args.output}")

    print("Loading adapter + base model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Merging weights...")
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(args.output)

    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
