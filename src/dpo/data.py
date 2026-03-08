"""
Load DPO preference pairs from the existing format (dpo_train.jsonl / dpo_val.jsonl)
produced by dataset_generation/build_dpo_dataset.py.

Each row has: prompt, chosen, rejected, question_id, variant, domain.
"""

import json

from torch.utils.data import Dataset


class DPODataset(Dataset):
    """Simple map-style dataset of (prompt, chosen, rejected) triples."""

    def __init__(self, path: str):
        self.rows = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                self.rows.append({
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def tokenize_pair(tokenizer, prompt: str, response: str, max_length: int):
    """Tokenize a prompt+response and return input_ids, labels, and a
    response-only mask (1 where the loss should be computed).

    The mask zeroes out prompt tokens so the loss is only over the response.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_text = prompt + response
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

    return {
        "input_ids": full_ids,
        "mask": mask,
    }


def collate_dpo(batch, tokenizer, max_length: int):
    """Collate a list of dataset rows into padded tensors for chosen and
    rejected sequences.

    Returns dict with keys:
        chosen_ids, chosen_mask, rejected_ids, rejected_mask
    each of shape (B, T) as padded torch tensors.
    """
    import torch

    chosen_encs = [
        tokenize_pair(tokenizer, row["prompt"], row["chosen"], max_length)
        for row in batch
    ]
    rejected_encs = [
        tokenize_pair(tokenizer, row["prompt"], row["rejected"], max_length)
        for row in batch
    ]

    def _pad(encodings, pad_id):
        max_len = max(len(e["input_ids"]) for e in encodings)
        ids = []
        masks = []
        for e in encodings:
            pad_len = max_len - len(e["input_ids"])
            ids.append(e["input_ids"] + [pad_id] * pad_len)
            masks.append(e["mask"] + [0] * pad_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.float)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    chosen_ids, chosen_mask = _pad(chosen_encs, pad_id)
    rejected_ids, rejected_mask = _pad(rejected_encs, pad_id)

    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask,
    }
