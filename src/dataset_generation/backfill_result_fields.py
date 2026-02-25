"""
One-time script to backfill existing sycophancy result files with prompt fields
(question_text, options, suggested_cot, authority_description) by joining against
the model-specific wrong_cots files.

Auto-detects which model's wrong_cots was used to build the variants for each
result file (some models were evaluated using Qwen's wrong COTs).

Fails hard on any mismatch.

Usage:
    python src/dataset_generation/backfill_result_fields.py --dry-run
    python src/dataset_generation/backfill_result_fields.py
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from dataset_generation.build_sycophancy_variants import inject_demographic

FIELDS_TO_ADD = ["question_text", "options", "suggested_cot", "authority_description"]

AUTHORITY_ROLES = config.AUTHORITY_ROLES


def _parse_result_filename(basename: str) -> tuple[str, str]:
    """Extract (model_safe, domain) from sycophancy_{model_safe}_{domain}.jsonl."""
    name = basename.removeprefix("sycophancy_").removesuffix(".jsonl")
    model_safe, domain = name.rsplit("_", 1)
    assert domain in ("legal", "medical"), f"Unexpected domain '{domain}' from {basename}"
    return model_safe, domain


def _load_wrong_cots_index(path: str) -> dict[int, dict]:
    """Load wrong_cots file keyed by question_id."""
    index = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qid = r["question_id"]
            assert qid not in index, f"Duplicate question_id {qid} in {path}"
            index[qid] = r
    return index


def _find_matching_wrong_cots(
    result_qids: set[int], domain: str
) -> tuple[str, dict[int, dict]]:
    """Find the wrong_cots file whose question_ids are a superset of result_qids.

    Returns (wrong_cots_model_safe, wrong_cots_index).
    Raises RuntimeError if no unique match is found.
    """
    candidates = sorted(glob.glob(
        os.path.join(config.DATA_DIR, f"wrong_cots_*_{domain}.jsonl")
    ))
    matches = []
    for wc_path in candidates:
        wc_name = os.path.basename(wc_path).removeprefix("wrong_cots_").removesuffix(".jsonl")
        wc_model = wc_name.rsplit("_", 1)[0]
        index = _load_wrong_cots_index(wc_path)
        missing = result_qids - set(index.keys())
        if len(missing) == 0:
            matches.append((wc_model, index, wc_path))

    if len(matches) == 0:
        raise RuntimeError(
            f"No wrong_cots file for domain={domain} covers all {len(result_qids)} result question_ids"
        )

    if len(matches) == 1:
        return matches[0][0], matches[0][1]

    # Multiple matches — prefer the one with the exact same count (tightest fit)
    matches.sort(key=lambda m: len(m[1]))
    best = matches[0]
    return best[0], best[1]


def _reconstruct_authority_description(variant: str, domain: str) -> str:
    """Deterministically reconstruct authority_description from variant name."""
    role = AUTHORITY_ROLES[domain]
    if variant == "sycophancy_user":
        return "A user"
    if variant == "sycophancy_authority":
        return f"A {role}"

    m = re.fullmatch(r"sycophancy_authority_(\w+)_(\w+)", variant)
    if m:
        return f"A {m.group(1)} {m.group(2)} {role}"

    m = re.fullmatch(r"sycophancy_authority_gender_(\w+)", variant)
    if m:
        return f"A {m.group(1)} {role}"

    m = re.fullmatch(r"sycophancy_authority_race_(\w+)", variant)
    if m:
        return f"A {m.group(1)} {role}"

    m = re.fullmatch(r"sycophancy_person_(\w+)_(\w+)", variant)
    if m:
        return f"A {role}"

    raise ValueError(f"Cannot reconstruct authority_description for variant '{variant}'")


def _reconstruct_question_text(
    variant: str, base_question_text: str
) -> str:
    """For person variants, apply demographic injection; otherwise return base text."""
    m = re.fullmatch(r"sycophancy_person_(\w+)_(\w+)", variant)
    if m:
        race, gender = m.group(1), m.group(2)
        return inject_demographic(base_question_text, race, gender)
    return base_question_text


def main():
    parser = argparse.ArgumentParser(description="Backfill result files with prompt fields")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    result_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "sycophancy_*_*.jsonl")))
    result_files = [f for f in result_files if "analysis_" not in os.path.basename(f)]
    print(f"Found {len(result_files)} result files\n")

    for rpath in result_files:
        basename = os.path.basename(rpath)
        declared_model, domain = _parse_result_filename(basename)

        rows = []
        with open(rpath) as f:
            for line in f:
                rows.append(json.loads(line))

        if not rows:
            print(f"EMPTY {basename}")
            continue

        already_has = all(field in rows[0] for field in FIELDS_TO_ADD)
        if already_has:
            print(f"SKIP  {basename} ({len(rows)} rows, already backfilled)")
            continue

        result_qids = {r["question_id"] for r in rows}
        source_model, wc_index = _find_matching_wrong_cots(result_qids, domain)

        if source_model != declared_model:
            print(f"NOTE  {basename}: variants came from {source_model}'s wrong COTs (not own)")

        for i, r in enumerate(rows):
            qid = r["question_id"]
            variant = r["variant"]
            wc = wc_index[qid]

            r["question_text"] = _reconstruct_question_text(variant, wc["question_text"])
            r["options"] = wc["options"]
            r["suggested_cot"] = wc["wrong_cot"]
            r["authority_description"] = _reconstruct_authority_description(variant, domain)

        # Validate a sample
        sample = rows[0]
        assert isinstance(sample["question_text"], str) and len(sample["question_text"]) > 10, \
            f"question_text looks wrong: {sample['question_text']!r}"
        assert isinstance(sample["options"], (dict, list)), \
            f"options has unexpected type: {type(sample['options'])}"
        assert isinstance(sample["suggested_cot"], str) and len(sample["suggested_cot"]) > 5, \
            f"suggested_cot looks wrong: {sample['suggested_cot']!r}"
        assert isinstance(sample["authority_description"], str), \
            f"authority_description has unexpected type: {type(sample['authority_description'])}"

        if args.dry_run:
            print(f"OK    {basename}: {len(rows)} rows would be updated (dry-run, source={source_model})")
        else:
            with open(rpath, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"DONE  {basename}: {len(rows)} rows updated (source={source_model})")

    print("\nFinished.")


if __name__ == "__main__":
    main()
