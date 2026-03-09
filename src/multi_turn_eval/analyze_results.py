"""
Statistical analysis of multi-turn escalating sycophancy results.

Key analyses (SycEval-inspired):
  1. Progressive vs regressive sycophancy rates
  2. Sycophancy by escalation level (simple → authority → justified → citation)
  3. Demographic effects within each escalation level
  4. Break-point analysis: at which level does the model first cave?
  5. Persistence: once the model caves, does it stay caved at higher levels?
  6. Statistical tests (chi-square, Cramer's V, two-proportion z-test)

Usage:
    python src/multi_turn_eval/analyze_results.py --file data/results/multi_turn/multiturn_sycophancy_..._legal.jsonl
    python src/multi_turn_eval/analyze_results.py   # auto-discover all multiturn result files
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Train/test split helpers (held-out evaluation)
# ---------------------------------------------------------------------------

_MODEL_SHORT = {
    "meta-llama_Llama-3.1-8B-Instruct": "llama",
    "google_gemma-2-9b-it": "gemma",
    "Qwen_Qwen2.5-7B-Instruct": "qwen",
    "checkpoints_dpo-llama-selfplay-nocot-merged": "llama",
    "checkpoints_dpo-gemma-selfplay-nocot-merged": "gemma",
    "checkpoints_dpo-qwen-selfplay-nocot-merged": "qwen",
}


def _load_train_qids(model_safe: str, direction: str, domain: str) -> set[int]:
    """Load training question_ids from the DPO manifest for a model/direction/domain."""
    short = _MODEL_SHORT[model_safe]
    manifest_path = os.path.join(config.DATA_DIR, f"dpo_selfplay_{short}_nocot_train_question_ids.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return set(manifest[direction][domain])


def _apply_split(df: pd.DataFrame, model_safe: str, direction: str,
                 domain: str, split: str) -> pd.DataFrame:
    """Filter DataFrame by train/test/all split using the DPO training manifest."""
    if split == "all":
        return df
    train_qids = _load_train_qids(model_safe, direction, domain)
    if split == "train":
        return df[df["question_id"].isin(train_qids)].reset_index(drop=True)
    return df[~df["question_id"].isin(train_qids)].reset_index(drop=True)

LEVEL_ORDER = ["simple", "authority", "justified", "citation"]


def _subsample_questions(df: pd.DataFrame, max_questions: int, seed: int = 42) -> pd.DataFrame:
    """Subsample to at most max_questions unique question_ids."""
    if max_questions <= 0:
        return df
    qids = df["question_id"].unique()
    if len(qids) <= max_questions:
        return df
    import random
    rng = random.Random(seed)
    sampled = set(rng.sample(list(qids), max_questions))
    return df[df["question_id"].isin(sampled)].reset_index(drop=True)

DEMOGRAPHIC_LAYERS = [
    ("User", "sycophancy_user"),
    ("Neutral Authority", "sycophancy_authority"),
    ("Demographic Authority", "sycophancy_authority_", ["sycophancy_authority_gender_", "sycophancy_authority_race_"]),
    ("Gender-Only Authority", "sycophancy_authority_gender_"),
    ("Race-Only Authority", "sycophancy_authority_race_"),
    ("Person Demographic", "sycophancy_person_"),
]


def _load_results(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".jsonl"):
        return config.load_jsonl(filepath)
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unknown file extension: {filepath}")


def _discover_files() -> list[str]:
    jsonl = glob.glob(os.path.join(config.MULTI_TURN_RESULTS_DIR, "multiturn_sycophancy_*.jsonl"))
    csv = glob.glob(os.path.join(config.MULTI_TURN_RESULTS_DIR, "multiturn_sycophancy_*.csv"))
    all_files = jsonl + csv
    all_files = [f for f in all_files if "analysis_" not in os.path.basename(f)]
    return sorted(all_files)


def _two_prop_z(n1, p1, n2, p2):
    """Two-proportion z-test. Returns (z_stat, p_value)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val


def _filter_layer(df: pd.DataFrame, layer_entry) -> pd.DataFrame:
    if len(layer_entry) == 3:
        _, variant_match, exclude_prefixes = layer_entry
    else:
        _, variant_match = layer_entry
        exclude_prefixes = []

    if variant_match.endswith("_"):
        sub = df[df["variant"].str.startswith(variant_match)]
        for exc in exclude_prefixes:
            sub = sub[~sub["variant"].str.startswith(exc)]
    else:
        sub = df[df["variant"] == variant_match]
    return sub


def analyze_multiturn(filepath: str):
    df_raw = _load_results(filepath)
    model = df_raw["model"].iloc[0]
    domain = df_raw["domain"].iloc[0] if "domain" in df_raw.columns else "unknown"

    total_raw = len(df_raw)
    invalid_mask = df_raw["model_answer"].isin(["INVALID", "ERROR"])
    n_invalid = invalid_mask.sum()
    df = df_raw[~invalid_mask].reset_index(drop=True)

    print(f"\n{'='*75}")
    print(f"MULTI-TURN ESCALATING SYCOPHANCY ANALYSIS")
    print(f"{'='*75}")
    print(f"Model:    {model}")
    print(f"Domain:   {domain}")
    print(f"Mode:     multi-turn escalating (SycEval-style)")
    print(f"Total:    {total_raw}")
    if n_invalid > 0:
        print(f"Excluded: {n_invalid} INVALID/ERROR ({n_invalid / total_raw * 100:.1f}%)")
    print(f"Analyzed: {len(df)}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"{'='*75}")

    # =================================================================
    # Section 1: Overall rates by direction
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"1. OVERALL SYCOPHANCY BY DIRECTION")
    print(f"{'─'*75}")

    for direction in ["regressive", "progressive"]:
        ddf = df[df["direction"] == direction]
        if len(ddf) == 0:
            continue
        syc_rate = ddf["deferred"].mean()
        n_syc = ddf["deferred"].sum()
        n_total = len(ddf)
        n_questions = ddf["question_id"].nunique()
        print(f"\n  {direction.upper()}:")
        print(f"    Questions: {n_questions}")
        print(f"    Sycophancy rate: {syc_rate:.4f} ({n_syc}/{n_total})")

    total_syc = df["deferred"].mean()
    print(f"\n  COMBINED: {total_syc:.4f} ({df['deferred'].sum()}/{len(df)})")

    # =================================================================
    # Section 2: Sycophancy by escalation level
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"2. SYCOPHANCY BY ESCALATION LEVEL")
    print(f"{'─'*75}")

    print(f"\n  {'Level':<12} {'Overall':>10} {'Regressive':>12} {'Progressive':>13} {'N':>8}")
    print(f"  {'─'*57}")

    level_stats = {}
    for level in LEVEL_ORDER:
        ldf = df[df["escalation_level"] == level]
        if len(ldf) == 0:
            continue

        overall = ldf["deferred"].mean()
        reg_df = ldf[ldf["direction"] == "regressive"]
        prog_df = ldf[ldf["direction"] == "progressive"]
        reg_rate = reg_df["deferred"].mean() if len(reg_df) > 0 else float("nan")
        prog_rate = prog_df["deferred"].mean() if len(prog_df) > 0 else float("nan")

        level_stats[level] = {
            "overall": overall,
            "regressive": reg_rate,
            "progressive": prog_rate,
            "n": len(ldf),
            "n_reg": len(reg_df),
            "n_prog": len(prog_df),
        }

        reg_str = f"{reg_rate:.4f}" if not np.isnan(reg_rate) else "N/A"
        prog_str = f"{prog_rate:.4f}" if not np.isnan(prog_rate) else "N/A"
        print(f"  {level:<12} {overall:>10.4f} {reg_str:>12} {prog_str:>13} {len(ldf):>8}")

    # Level-to-level z-tests
    if len(level_stats) > 1:
        print(f"\n  Level-to-level comparisons (two-proportion z-test):")
        prev_level = None
        for level in LEVEL_ORDER:
            if level not in level_stats:
                continue
            if prev_level is not None:
                z, p = _two_prop_z(
                    level_stats[prev_level]["n"], level_stats[prev_level]["overall"],
                    level_stats[level]["n"], level_stats[level]["overall"],
                )
                sig = "**" if p < 0.05 else ""
                print(f"    {prev_level} → {level}: z={z:+.3f}, p={p:.4f} {sig}")
            prev_level = level

    # =================================================================
    # Section 3: Demographic effects per level
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"3. DEMOGRAPHIC EFFECTS BY ESCALATION LEVEL")
    print(f"{'─'*75}")

    for level in ["authority", "justified", "citation"]:
        ldf = df[df["escalation_level"] == level]
        if len(ldf) == 0:
            continue

        print(f"\n  ── Level: {level.upper()} ──")

        auth_neutral = ldf[ldf["variant"] == "sycophancy_authority"]
        neutral_rate = auth_neutral["deferred"].mean() if len(auth_neutral) > 0 else 0.0

        print(f"  {'Variant':<28} {'Deference':>10} {'vs Neutral':>11} {'N':>6}")
        print(f"  {'─'*57}")
        print(f"  {'Neutral Authority':<28} {neutral_rate:>10.4f} {'---':>11} {len(auth_neutral):>6}")

        auth_demo = ldf[
            ldf["variant"].str.startswith("sycophancy_authority_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_gender_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_race_")
        ]
        gender_only = ldf[ldf["variant"].str.startswith("sycophancy_authority_gender_")]
        race_only = ldf[ldf["variant"].str.startswith("sycophancy_authority_race_")]

        def _show(label, subset):
            if len(subset) == 0:
                return
            r = subset["deferred"].mean()
            print(f"  {label:<28} {r:>10.4f} {r - neutral_rate:>+10.4f}p {len(subset):>6}")

        for gender in config.GENDERS:
            _show(f"  {gender} (gender-only)", gender_only[gender_only["gender"] == gender])
        for race in config.RACES:
            _show(f"  {race} (race-only)", race_only[race_only["race"] == race])
        for race in config.RACES:
            for gender in config.GENDERS:
                sub = auth_demo[(auth_demo["race"] == race) & (auth_demo["gender"] == gender)]
                _show(f"  {race} {gender}", sub)

        # BF vs WM two-proportion z-test
        bf = auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")]
        wm = auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")]
        if len(bf) > 0 and len(wm) > 0:
            bf_rate = bf["deferred"].mean()
            wm_rate = wm["deferred"].mean()
            _, p_bfwm = _two_prop_z(len(bf), bf_rate, len(wm), wm_rate)
            sig = "***" if p_bfwm < 0.001 else ("**" if p_bfwm < 0.01 else ("*" if p_bfwm < 0.05 else ""))
            print(f"\n  BF vs WM @ {level}: BF={bf_rate:.4f} WM={wm_rate:.4f} diff={bf_rate-wm_rate:+.4f} p={p_bfwm:.4f} {sig}")

    # =================================================================
    # Section 4: Break-point analysis
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"4. BREAK-POINT ANALYSIS")
    print(f"{'─'*75}")
    print("  At which escalation level does the model first defer?")

    for direction in ["regressive", "progressive"]:
        ddf = df[df["direction"] == direction]
        if len(ddf) == 0:
            continue

        print(f"\n  {direction.upper()}:")

        # For break-point, use authority-level variants (not simple which has no demographics).
        # Group by question_id and find the first level where it deferred.
        # We need to look at the same variant across levels.
        # Use "sycophancy_authority" (neutral authority) for clean comparison.
        authority_df = ddf[ddf["variant"] == "sycophancy_authority"]

        break_counts = {level: 0 for level in LEVEL_ORDER}
        never_count = 0
        total_questions = 0

        for qid in authority_df["question_id"].unique():
            qdf = authority_df[authority_df["question_id"] == qid]
            total_questions += 1
            first_break = None
            for level in LEVEL_ORDER:
                level_row = qdf[qdf["escalation_level"] == level]
                if len(level_row) > 0 and level_row.iloc[0]["deferred"]:
                    first_break = level
                    break
            if first_break:
                break_counts[first_break] += 1
            else:
                never_count += 1

        if total_questions > 0:
            print(f"    {'Level':<12} {'First Break':>12} {'Cumulative':>12}")
            print(f"    {'─'*38}")
            cumulative = 0
            for level in LEVEL_ORDER:
                cumulative += break_counts[level]
                pct = break_counts[level] / total_questions * 100
                cum_pct = cumulative / total_questions * 100
                print(f"    {level:<12} {break_counts[level]:>5} ({pct:>5.1f}%) {cumulative:>5} ({cum_pct:>5.1f}%)")
            print(f"    {'never':<12} {never_count:>5} ({never_count / total_questions * 100:>5.1f}%)")
            print(f"    Total questions (with neutral authority at all levels): {total_questions}")

    # =================================================================
    # Section 5: Persistence analysis
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"5. PERSISTENCE ANALYSIS")
    print(f"{'─'*75}")
    print("  Once the model caves, does it stay caved at higher levels?")

    for direction in ["regressive", "progressive"]:
        ddf = df[df["direction"] == direction]
        if len(ddf) == 0:
            continue

        print(f"\n  {direction.upper()}:")

        authority_df = ddf[ddf["variant"] == "sycophancy_authority"]
        persistent = 0
        non_persistent = 0
        total_chains = 0

        for qid in authority_df["question_id"].unique():
            qdf = authority_df[authority_df["question_id"] == qid].copy()
            qdf["_level_idx"] = qdf["escalation_level"].map(
                {l: i for i, l in enumerate(LEVEL_ORDER)}
            )
            qdf = qdf.sort_values("_level_idx")

            if len(qdf) < 2:
                continue

            deference_seq = qdf["deferred"].tolist()
            total_chains += 1

            # Persistent = once caved, stays caved (monotonic: 0...0,1...1)
            first_true = None
            is_persistent = True
            for i, d in enumerate(deference_seq):
                if d and first_true is None:
                    first_true = i
                elif not d and first_true is not None:
                    is_persistent = False
                    break

            if first_true is None:
                is_persistent = True

            if is_persistent:
                persistent += 1
            else:
                non_persistent += 1

        if total_chains > 0:
            rate = persistent / total_chains
            print(f"    Persistent chains: {persistent}/{total_chains} ({rate:.1%})")
            print(f"    Non-persistent:    {non_persistent}/{total_chains} ({non_persistent / total_chains:.1%})")

    # =================================================================
    # Section 6: Sycophancy ladder (demographic layers, all levels combined)
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"6. DEMOGRAPHIC SYCOPHANCY LADDER (all levels combined)")
    print(f"{'─'*75}")

    # Exclude level "simple" since it doesn't have demographic variants
    auth_levels = df[df["escalation_level"].isin(["authority", "justified", "citation"])]

    print(f"\n  {'Layer':<30} {'Deference':>12} {'N':>6}")
    print(f"  {'─'*50}")

    for layer_entry in DEMOGRAPHIC_LAYERS:
        layer_name = layer_entry[0]
        sub = _filter_layer(auth_levels, layer_entry)
        if len(sub) == 0:
            continue
        rate = sub["deferred"].mean()
        print(f"  {layer_name:<30} {rate:>12.4f} {len(sub):>6}")

    # =================================================================
    # Section 7: Statistical tests
    # =================================================================
    print(f"\n{'─'*75}")
    print(f"7. STATISTICAL TESTS")
    print(f"{'─'*75}")

    # 7a. Chi-square: escalation_level vs deferred (overall)
    ct_level = pd.crosstab(df["escalation_level"], df["deferred"])
    if ct_level.shape[0] > 1 and ct_level.shape[1] > 1:
        chi2, p, _, _ = chi2_contingency(ct_level.values)
        v = config.cramers_v(ct_level.values)
        sig = "**" if p < 0.05 else ""
        print(f"\n  Escalation level → deference:")
        print(f"    χ²={chi2:.2f}, p={p:.4f}, V={v:.4f} {sig}")

    # 7b. Direction vs deferred
    ct_dir = pd.crosstab(df["direction"], df["deferred"])
    if ct_dir.shape[0] > 1 and ct_dir.shape[1] > 1:
        chi2, p, _, _ = chi2_contingency(ct_dir.values)
        v = config.cramers_v(ct_dir.values)
        sig = "**" if p < 0.05 else ""
        print(f"\n  Direction (regressive/progressive) → deference:")
        print(f"    χ²={chi2:.2f}, p={p:.4f}, V={v:.4f} {sig}")

    # 7c. Demographics at each authority level
    for level in ["authority", "justified", "citation"]:
        ldf = df[df["escalation_level"] == level]
        auth_demo = ldf[
            ldf["variant"].str.startswith("sycophancy_authority_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_gender_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_race_")
        ]
        if len(auth_demo) == 0:
            continue
        ct = pd.crosstab(auth_demo["variant"], auth_demo["deferred"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(ct.values)
            v = config.cramers_v(ct.values)
            sig = "**" if p < 0.05 else ""
            print(f"\n  Demo authority (race×gender) → deference @ {level}:")
            print(f"    χ²={chi2:.2f}, p={p:.4f}, V={v:.4f} {sig}")

        gender_only = ldf[ldf["variant"].str.startswith("sycophancy_authority_gender_")]
        if len(gender_only) > 0:
            ct_g = pd.crosstab(gender_only["gender"], gender_only["deferred"])
            if ct_g.shape[0] > 1 and ct_g.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(ct_g.values)
                v = config.cramers_v(ct_g.values)
                sig = "**" if p < 0.05 else ""
                print(f"\n  Gender-only → deference @ {level}:")
                print(f"    χ²={chi2:.2f}, p={p:.4f}, V={v:.4f} {sig}")

        race_only = ldf[ldf["variant"].str.startswith("sycophancy_authority_race_")]
        if len(race_only) > 0:
            ct_r = pd.crosstab(race_only["race"], race_only["deferred"])
            if ct_r.shape[0] > 1 and ct_r.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(ct_r.values)
                v = config.cramers_v(ct_r.values)
                sig = "**" if p < 0.05 else ""
                print(f"\n  Race-only → deference @ {level}:")
                print(f"    χ²={chi2:.2f}, p={p:.4f}, V={v:.4f} {sig}")

    # =================================================================
    # Save detailed breakdown
    # =================================================================
    question_stats = []
    for qid in sorted(df["question_id"].unique()):
        qdf = df[df["question_id"] == qid]
        direction = qdf["direction"].iloc[0]

        row_data = {
            "question_id": qid,
            "direction": direction,
            "correct_answer": qdf["correct_answer"].iloc[0],
            "suggested_answer": qdf["suggested_answer"].iloc[0],
        }

        for level in LEVEL_ORDER:
            level_df = qdf[qdf["escalation_level"] == level]
            if len(level_df) > 0:
                row_data[f"deference_{level}"] = level_df["deferred"].mean()
                row_data[f"n_{level}"] = len(level_df)
            else:
                row_data[f"deference_{level}"] = float("nan")
                row_data[f"n_{level}"] = 0

        row_data["total_deference"] = qdf["deferred"].mean()
        if "sycophantic" in qdf.columns:
            row_data["sycophantic_rate"] = qdf["sycophantic"].mean()
        question_stats.append(row_data)

    q_df = pd.DataFrame(question_stats).sort_values("total_deference", ascending=False)

    reg_df = df[df["direction"] == "regressive"]
    prog_df = df[df["direction"] == "progressive"]
    reg_def = reg_df["deferred"].mean() if len(reg_df) > 0 else float("nan")
    prog_def = prog_df["deferred"].mean() if len(prog_df) > 0 else float("nan")

    # Summary stats per level for the consolidated summary table
    summary_cols = {}

    # Combined demographic rates across authority+justified+citation (for simple table)
    demo_levels_df = df[df["escalation_level"].isin(["authority", "justified", "citation"])]
    auth_neutral_combined = demo_levels_df[demo_levels_df["variant"] == "sycophancy_authority"]
    gender_only_combined = demo_levels_df[demo_levels_df["variant"].str.startswith("sycophancy_authority_gender_")]
    race_only_combined = demo_levels_df[demo_levels_df["variant"].str.startswith("sycophancy_authority_race_")]
    auth_demo_combined = demo_levels_df[
        demo_levels_df["variant"].str.startswith("sycophancy_authority_")
        & ~demo_levels_df["variant"].str.startswith("sycophancy_authority_gender_")
        & ~demo_levels_df["variant"].str.startswith("sycophancy_authority_race_")
    ]
    def _rate(subset):
        return round(subset["deferred"].mean(), 4) if len(subset) > 0 else float("nan")

    summary_cols["neutral_rate"] = _rate(auth_neutral_combined)
    summary_cols["male_rate"] = _rate(gender_only_combined[gender_only_combined["gender"] == "male"])
    summary_cols["female_rate"] = _rate(gender_only_combined[gender_only_combined["gender"] == "female"])
    summary_cols["white_rate"] = _rate(race_only_combined[race_only_combined["race"] == "White"])
    summary_cols["black_rate"] = _rate(race_only_combined[race_only_combined["race"] == "Black"])
    summary_cols["white_male_rate"] = _rate(auth_demo_combined[(auth_demo_combined["race"] == "White") & (auth_demo_combined["gender"] == "male")])
    summary_cols["white_female_rate"] = _rate(auth_demo_combined[(auth_demo_combined["race"] == "White") & (auth_demo_combined["gender"] == "female")])
    summary_cols["black_male_rate"] = _rate(auth_demo_combined[(auth_demo_combined["race"] == "Black") & (auth_demo_combined["gender"] == "male")])
    summary_cols["black_female_rate"] = _rate(auth_demo_combined[(auth_demo_combined["race"] == "Black") & (auth_demo_combined["gender"] == "female")])

    for level in LEVEL_ORDER:
        ldf = df[df["escalation_level"] == level]
        summary_cols[f"level_{level}_overall"] = round(ldf["deferred"].mean(), 4) if len(ldf) > 0 else float("nan")

    for level in ["authority", "justified", "citation"]:
        ldf = df[df["escalation_level"] == level]
        auth_neutral_ldf = ldf[ldf["variant"] == "sycophancy_authority"]
        gender_only_ldf = ldf[ldf["variant"].str.startswith("sycophancy_authority_gender_")]
        race_only_ldf = ldf[ldf["variant"].str.startswith("sycophancy_authority_race_")]
        auth_demo = ldf[
            ldf["variant"].str.startswith("sycophancy_authority_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_gender_")
            & ~ldf["variant"].str.startswith("sycophancy_authority_race_")
        ]

        # All per-level demographic rates
        summary_cols[f"{level}_neutral"] = _rate(auth_neutral_ldf)
        summary_cols[f"{level}_male"] = _rate(gender_only_ldf[gender_only_ldf["gender"] == "male"])
        summary_cols[f"{level}_female"] = _rate(gender_only_ldf[gender_only_ldf["gender"] == "female"])
        summary_cols[f"{level}_white"] = _rate(race_only_ldf[race_only_ldf["race"] == "White"])
        summary_cols[f"{level}_black"] = _rate(race_only_ldf[race_only_ldf["race"] == "Black"])
        summary_cols[f"{level}_wm"] = _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")])
        summary_cols[f"{level}_wf"] = _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "female")])
        summary_cols[f"{level}_bm"] = _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "male")])
        summary_cols[f"{level}_bf"] = _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")])

        bf_rate = summary_cols[f"{level}_bf"]
        wm_rate = summary_cols[f"{level}_wm"]
        bf_rows = auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")]
        wm_rows = auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")]
        summary_cols[f"{level}_bf_wm_diff"] = round(bf_rate - wm_rate, 4) if not (np.isnan(bf_rate) or np.isnan(wm_rate)) else float("nan")
        _, p_bfwm = _two_prop_z(len(bf_rows), bf_rate, len(wm_rows), wm_rate) if (len(bf_rows) > 0 and len(wm_rows) > 0) else (0.0, float("nan"))
        summary_cols[f"{level}_bf_wm_p"] = round(p_bfwm, 4) if not np.isnan(p_bfwm) else float("nan")

        # Cramer's V for race x gender demo authority at this level
        cramers_v_level = float("nan")
        if len(auth_demo) > 0:
            ct = pd.crosstab(auth_demo["variant"], auth_demo["deferred"])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                cramers_v_level = round(config.cramers_v(ct.values), 4)
        summary_cols[f"{level}_cramers_v"] = cramers_v_level

    q_df.insert(0, "total_responses", total_raw)
    q_df.insert(1, "invalid_count", n_invalid)
    q_df.insert(2, "invalid_pct", round(n_invalid / total_raw * 100, 2))
    q_df.insert(3, "regressive_deference", round(reg_def, 4))
    q_df.insert(4, "progressive_deference", round(prog_def, 4))
    for i, (col, val) in enumerate(summary_cols.items()):
        q_df.insert(5 + i, col, val)

    basename = os.path.splitext(os.path.basename(filepath))[0]
    analysis_path = os.path.join(config.MULTI_TURN_RESULTS_DIR, f"analysis_{basename}.csv")
    q_df.to_csv(analysis_path, index=False)
    print(f"\n  Per-question breakdown saved to {analysis_path}")


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _print_table(rows: list[dict], title: str, columns: list[str] | None = None, tsv: bool = False):
    summary_df = pd.DataFrame(rows)
    if columns:
        summary_df = summary_df[[c for c in columns if c in summary_df.columns]]
    if tsv:
        print("\t".join(summary_df.columns))
        for _, r in summary_df.iterrows():
            print("\t".join(str(r[col]) for col in summary_df.columns))
        return
    col_widths = {col: max(len(col), summary_df[col].astype(str).str.len().max()) for col in summary_df.columns}
    header = "  " + "  ".join(col.ljust(col_widths[col]) for col in summary_df.columns)
    sep = "  " + "  ".join("-" * col_widths[col] for col in summary_df.columns)
    print(f"\n{'='*len(header)}")
    print(title)
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for _, r in summary_df.iterrows():
        print("  " + "  ".join(str(r[col]).ljust(col_widths[col]) for col in summary_df.columns))
    print(f"{'='*len(header)}\n")


def _print_tables_by_domain(rows: list[dict], title_base: str,
                            columns: list[str] | None = None, tsv: bool = False):
    """Split rows by Domain, print one table per domain (medical first)."""
    domain_order = {"medical": 0, "legal": 1}
    domains = sorted({r["Domain"] for r in rows}, key=lambda d: domain_order.get(d, 99))
    for domain in domains:
        domain_rows = [{k: v for k, v in r.items() if k != "Domain"} for r in rows if r["Domain"] == domain]
        cols = [c for c in columns if c != "Domain"] if columns else None
        _print_table(domain_rows, f"{title_base} — {domain.upper()}", columns=cols, tsv=tsv)


FOCUS_MODELS = {
    "meta-llama_Llama-3.1-8B-Instruct",
    "google_gemma-2-9b-it",
    "Qwen_Qwen2.5-7B-Instruct",
    "checkpoints_dpo-llama-selfplay-nocot-merged",
    "checkpoints_dpo-gemma-selfplay-nocot-merged",
    "checkpoints_dpo-qwen-selfplay-nocot-merged",
}

MODEL_PAIRS = [
    ("meta-llama_Llama-3.1-8B-Instruct", "checkpoints_dpo-llama-selfplay-nocot-merged", "LLaMA"),
    ("google_gemma-2-9b-it", "checkpoints_dpo-gemma-selfplay-nocot-merged", "Gemma"),
    ("Qwen_Qwen2.5-7B-Instruct", "checkpoints_dpo-qwen-selfplay-nocot-merged", "Qwen"),
]


def print_summary_table(simple: bool = False, tsv: bool = False, level: str | None = None):
    """Read all analysis_multiturn_*.csv files and print a consolidated table.

    When ``level`` is specified with ``simple``, shows that level's demographic
    rates with "vs Neu" columns (matching the single-turn table format).
    """
    pattern = os.path.join(config.MULTI_TURN_RESULTS_DIR, "analysis_multiturn_sycophancy_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No multi-turn analysis CSVs found. Run evaluate_multi_turn.sh first.")
        return

    rows = []
    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        row = df.iloc[0]

        basename = os.path.splitext(os.path.basename(fpath))[0]
        parts = basename.replace("analysis_multiturn_sycophancy_", "").rsplit("_", 1)
        model_safe = parts[0] if len(parts) == 2 else basename
        domain = parts[1] if len(parts) == 2 else "?"

        def _fmt(key):
            v = row.get(key, float("nan"))
            return f"{v:.1%}" if not np.isnan(v) else "N/A"

        prog_def = row.get("progressive_deference", float("nan"))
        prog_syc = (1 - prog_def) if not np.isnan(prog_def) else float("nan")

        entry = {
            "Model": model_safe,
            "Domain": domain,
            "N": int(row.get("total_responses", 0)),
            "Invalid": int(row.get("invalid_count", 0)),
            "Inv%": f"{row.get('invalid_pct', float('nan')):.1f}%",
            "Regressive": f"{row.get('regressive_deference', float('nan')):.1%}",
            "Prog Syc↓": f"{prog_syc:.1%}" if not np.isnan(prog_syc) else "N/A",
            # Combined demographic rates (across auth+justified+citation)
            "Neutral": _fmt("neutral_rate"),
            "White": _fmt("white_rate"),
            "Black": _fmt("black_rate"),
            "Female": _fmt("female_rate"),
            "Male": _fmt("male_rate"),
            "BF": _fmt("black_female_rate"),
            "WM": _fmt("white_male_rate"),
        }
        for lv in LEVEL_ORDER:
            val = row.get(f"level_{lv}_overall", float("nan"))
            entry[lv.capitalize()] = f"{val:.1%}" if not np.isnan(val) else "N/A"

        for lv in ["authority", "justified", "citation"]:
            lshort = lv[:3].upper()
            for demo_key, col_label in [
                (f"{lv}_neutral", f"{lshort} Neu"),
                (f"{lv}_male",    f"{lshort} M"),
                (f"{lv}_female",  f"{lshort} F"),
                (f"{lv}_white",   f"{lshort} W"),
                (f"{lv}_black",   f"{lshort} B"),
                (f"{lv}_wm",      f"{lshort} WM"),
                (f"{lv}_wf",      f"{lshort} WF"),
                (f"{lv}_bm",      f"{lshort} BM"),
                (f"{lv}_bf",      f"{lshort} BF"),
            ]:
                v = row.get(demo_key, float("nan"))
                entry[col_label] = f"{v:.1%}" if not np.isnan(v) else "N/A"

            diff = row.get(f"{lv}_bf_wm_diff", float("nan"))
            p = row.get(f"{lv}_bf_wm_p", float("nan"))
            cramers = row.get(f"{lv}_cramers_v", float("nan"))
            entry[f"{lshort} BF-WM"] = f"{diff:+.1%}" if not np.isnan(diff) else "N/A"
            entry[f"{lshort} p"] = f"{p:.4f}" if not np.isnan(p) else "N/A"
            entry[f"{lshort} sig"] = _sig_stars(p)
            entry[f"{lshort} V"] = f"{cramers:.3f}" if not np.isnan(cramers) else "N/A"

        # Combined BF–WM using the authority-level p as representative
        auth_diff = row.get("authority_bf_wm_diff", float("nan"))
        entry["BF–WM"] = f"{auth_diff:+.1%}" if not np.isnan(auth_diff) else "N/A"

        # Level-specific demographic columns with "vs Neu" diffs
        if level:
            def _get(key):
                return row.get(key, float("nan"))

            def _pct(v):
                return f"{v:.2%}" if not np.isnan(v) else "N/A"

            def _diff(v, baseline):
                if np.isnan(v) or np.isnan(baseline):
                    return "N/A"
                d = (v - baseline) * 100
                if d < 0:
                    return f"{d:.1f}pp"
                return f"{d:.1f}pp"

            neu = _get(f"{level}_neutral")
            m = _get(f"{level}_male")
            f_ = _get(f"{level}_female")
            w = _get(f"{level}_white")
            b = _get(f"{level}_black")
            wm_v = _get(f"{level}_wm")
            wf_v = _get(f"{level}_wf")
            bm_v = _get(f"{level}_bm")
            bf_v = _get(f"{level}_bf")
            bf_wm_d = _get(f"{level}_bf_wm_diff")
            bf_wm_pv = _get(f"{level}_bf_wm_p")

            entry["Neutral"] = _pct(neu)
            entry["Male"] = _pct(m)
            entry["Male vs Neu"] = _diff(m, neu)
            entry["Female"] = _pct(f_)
            entry["Female vs Neu"] = _diff(f_, neu)
            entry["White"] = _pct(w)
            entry["White vs Neu"] = _diff(w, neu)
            entry["Black"] = _pct(b)
            entry["Black vs Neu"] = _diff(b, neu)
            entry["White Male"] = _pct(wm_v)
            entry["WM vs Neu"] = _diff(wm_v, neu)
            entry["White Female"] = _pct(wf_v)
            entry["WF vs Neu"] = _diff(wf_v, neu)
            entry["Black Male"] = _pct(bm_v)
            entry["BM vs Neu"] = _diff(bm_v, neu)
            entry["Black Female"] = _pct(bf_v)
            entry["BF vs Neu"] = _diff(bf_v, neu)
            entry["BF vs WM"] = f"{bf_wm_d * 100:.1f}pp" if not np.isnan(bf_wm_d) else "N/A"
            entry["p-value"] = f"{bf_wm_pv:.4f}" if not np.isnan(bf_wm_pv) else "N/A"

        rows.append(entry)

    rows = [r for r in rows if r["Model"] in FOCUS_MODELS]
    if not rows:
        print("No data rows found in analysis CSVs.")
        return

    rows.sort(key=lambda r: r["Model"])

    if simple and level:
        level_cols = [
            "Model", "Neutral",
            "Male", "Male vs Neu", "Female", "Female vs Neu",
            "White", "White vs Neu", "Black", "Black vs Neu",
            "White Male", "WM vs Neu", "White Female", "WF vs Neu",
            "Black Male", "BM vs Neu", "Black Female", "BF vs Neu",
            "BF vs WM", "p-value",
        ]
        _print_tables_by_domain(
            rows,
            f"MULTI-TURN SYCOPHANCY ({level.upper()} LEVEL)",
            columns=level_cols,
            tsv=tsv,
        )
    elif simple:
        _print_tables_by_domain(
            rows,
            "MULTI-TURN SYCOPHANCY",
            columns=["Model", "Neutral", "White", "Black", "Female", "Male", "BF", "WM", "BF–WM", "AUT p", "AUT sig"],
            tsv=tsv,
        )
    else:
        _print_tables_by_domain(rows, "MULTI-TURN SYCOPHANCY", tsv=tsv)


def print_level_table(tsv: bool = False, model_filter: str | None = None):
    """Print a level × demographic breakdown table for each model/domain."""
    pattern = os.path.join(config.MULTI_TURN_RESULTS_DIR, "analysis_multiturn_sycophancy_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No multi-turn analysis CSVs found. Run evaluate_multi_turn.sh first.")
        return

    # Columns: level as rows, demographics as columns
    DEMO_COLS = [
        ("Neutral",  "{level}_neutral"),
        ("Male",     "{level}_male"),
        ("Female",   "{level}_female"),
        ("White",    "{level}_white"),
        ("Black",    "{level}_black"),
        ("WM",       "{level}_wm"),
        ("WF",       "{level}_wf"),
        ("BM",       "{level}_bm"),
        ("BF",       "{level}_bf"),
        ("BF–WM",    "{level}_bf_wm_diff"),
        ("p",        "{level}_bf_wm_p"),
        ("sig",      "{level}_bf_wm_p"),  # derived
    ]

    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue

        basename = os.path.splitext(os.path.basename(fpath))[0]
        parts = basename.replace("analysis_multiturn_sycophancy_", "").rsplit("_", 1)
        model_safe = parts[0] if len(parts) == 2 else basename
        domain = parts[1] if len(parts) == 2 else "?"

        if model_safe not in FOCUS_MODELS:
            continue
        if model_filter and model_filter.lower() not in model_safe.lower():
            continue

        row = df.iloc[0]
        rows = []
        for level in LEVEL_ORDER:
            entry = {"Level": level}
            if level == "simple":
                overall = row.get("level_simple_overall", float("nan"))
                entry["Overall"] = f"{overall:.1%}" if not np.isnan(overall) else "N/A"
                for col_label, _ in DEMO_COLS:
                    entry[col_label] = "—"
            else:
                overall = row.get(f"level_{level}_overall", float("nan"))
                entry["Overall"] = f"{overall:.1%}" if not np.isnan(overall) else "N/A"
                for col_label, key_template in DEMO_COLS:
                    key = key_template.format(level=level)
                    raw = row.get(key, float("nan"))
                    if col_label == "sig":
                        entry[col_label] = _sig_stars(raw) if not np.isnan(raw) else "N/A"
                    elif col_label == "BF–WM":
                        entry[col_label] = f"{raw:+.1%}" if not np.isnan(raw) else "N/A"
                    elif col_label == "p":
                        entry[col_label] = f"{raw:.4f}" if not np.isnan(raw) else "N/A"
                    else:
                        entry[col_label] = f"{raw:.1%}" if not np.isnan(raw) else "N/A"
            rows.append(entry)

        title = f"{model_safe}  |  {domain}"
        _print_table(rows, title, tsv=tsv)


def _load_baseline_accuracy() -> dict[tuple[str, str], float]:
    """Return {(model_safe, domain): accuracy} from baseline result files."""
    baseline_files = glob.glob(os.path.join(config.BASELINE_RESULTS_DIR, "baseline_cot_*.jsonl"))
    result = {}
    for fpath in baseline_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        remainder = basename.replace("baseline_cot_", "", 1)
        parts = remainder.rsplit("_", 1)
        model_safe = parts[0]
        domain = parts[1]
        bdf = config.load_jsonl(fpath)
        result[(model_safe, domain)] = bdf["is_correct"].mean()
    return result


def print_accuracy_table(tsv: bool = False, level: str | None = None,
                         combined: bool = False):
    """Accuracy per demographic from raw multi-turn JSONL results (regressive only)."""
    from scipy.stats import norm as _norm

    result_files = _discover_files()

    if not result_files:
        print("No multi-turn JSONL files found.")
        return

    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for fpath in result_files:
        df = _load_results(fpath)
        if df.empty:
            continue
        invalid_mask = df["model_answer"].isin(["INVALID", "ERROR"])
        df = df[~invalid_mask].reset_index(drop=True)

        model = df["model"].iloc[0]
        domain = df["domain"].iloc[0] if "domain" in df.columns else "unknown"
        model_safe = model.replace("/", "_")

        if model_safe not in FOCUS_MODELS:
            continue

        reg = df[df["direction"] == "regressive"]
        if level:
            reg = reg[reg["escalation_level"] == level]
        if reg.empty:
            continue

        domain_label = "combined" if combined else domain
        groups.setdefault((model_safe, domain_label), []).append(reg)

    def _pct(v):
        return f"{v:.1%}" if not np.isnan(v) else "N/A"

    def _diff(v, ref):
        if np.isnan(v) or np.isnan(ref):
            return "N/A"
        d = (v - ref) * 100
        return f"{d:.1f}pp" if d >= 0 else f"{d:.1f}pp"

    rows = []
    for (model_safe, domain_label), dfs in groups.items():
        reg = pd.concat(dfs, ignore_index=True)

        auth_neutral = reg[reg["variant"] == "sycophancy_authority"]
        auth_demo = reg[
            reg["variant"].str.startswith("sycophancy_authority_")
            & ~reg["variant"].str.startswith("sycophancy_authority_gender_")
            & ~reg["variant"].str.startswith("sycophancy_authority_race_")
        ]

        def _acc(subset):
            return subset["is_correct"].mean() if len(subset) > 0 else float("nan")

        neu_acc = _acc(auth_neutral)
        wm_df = auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")]
        wf_df = auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "female")]
        bm_df = auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "male")]
        bf_df = auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")]

        wm_acc = _acc(wm_df)
        wf_acc = _acc(wf_df)
        bm_acc = _acc(bm_df)
        bf_acc = _acc(bf_df)

        bf_wm_p = float("nan")
        if len(bf_df) > 0 and len(wm_df) > 0 and not (np.isnan(bf_acc) or np.isnan(wm_acc)):
            n1, n2 = len(bf_df), len(wm_df)
            p_pool = (bf_acc * n1 + wm_acc * n2) / (n1 + n2)
            if 0 < p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
                if se > 0:
                    z = (bf_acc - wm_acc) / se
                    bf_wm_p = 2 * (1 - _norm.cdf(abs(z)))

        rows.append({
            "Model": model_safe,
            "Domain": domain_label,
            "Neutral": _pct(neu_acc),
            "WM": _pct(wm_acc),
            "WM vs Neu": _diff(wm_acc, neu_acc),
            "WF": _pct(wf_acc),
            "WF vs Neu": _diff(wf_acc, neu_acc),
            "BM": _pct(bm_acc),
            "BM vs Neu": _diff(bm_acc, neu_acc),
            "BF": _pct(bf_acc),
            "BF vs Neu": _diff(bf_acc, neu_acc),
            "BF-WM": _diff(bf_acc, wm_acc),
            "BF-WM p": f"{bf_wm_p:.4f}" if not np.isnan(bf_wm_p) else "N/A",
            "sig": _sig_stars(bf_wm_p),
        })

    rows.sort(key=lambda r: r["Model"])

    level_label = f" ({level.upper()} LEVEL)" if level else " (ALL LEVELS)"
    _print_tables_by_domain(rows, f"MULTI-TURN ACCURACY BY DEMOGRAPHIC — REGRESSIVE{level_label}", tsv=tsv)


def _compute_deference_row(model_safe: str, domain_label: str,
                           sub: pd.DataFrame) -> dict:
    """Compute a deference-table row from a filtered DataFrame."""
    auth_neutral = sub[sub["variant"] == "sycophancy_authority"]
    auth_demo = sub[
        sub["variant"].str.startswith("sycophancy_authority_")
        & ~sub["variant"].str.startswith("sycophancy_authority_gender_")
        & ~sub["variant"].str.startswith("sycophancy_authority_race_")
    ]
    gender_only = sub[sub["variant"].str.startswith("sycophancy_authority_gender_")]
    race_only = sub[sub["variant"].str.startswith("sycophancy_authority_race_")]

    def _rate(subset):
        return subset["deferred"].mean() if len(subset) > 0 else float("nan")

    neu = _rate(auth_neutral)
    ma = _rate(gender_only[gender_only["gender"] == "male"])
    fe = _rate(gender_only[gender_only["gender"] == "female"])
    wh = _rate(race_only[race_only["race"] == "White"])
    bl = _rate(race_only[race_only["race"] == "Black"])
    wm_r = _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")])
    wf_r = _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "female")])
    bm_r = _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "male")])
    bf_r = _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")])

    bf_wm_p = float("nan")
    n_bf = len(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")])
    n_wm = len(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")])
    if n_bf > 0 and n_wm > 0 and not (np.isnan(bf_r) or np.isnan(wm_r)):
        _, bf_wm_p = _two_prop_z(n_bf, bf_r, n_wm, wm_r)

    def _pct(v):
        return f"{v:.1%}" if not np.isnan(v) else "N/A"

    def _diff(v, ref):
        if np.isnan(v) or np.isnan(ref):
            return "N/A"
        d = (v - ref) * 100
        return f"{d:.1f}pp" if d >= 0 else f"{d:.1f}pp"

    return {
        "Model": model_safe,
        "Domain": domain_label,
        "Neutral": _pct(neu),
        "Male": _pct(ma), "Male vs Neu": _diff(ma, neu),
        "Female": _pct(fe), "Female vs Neu": _diff(fe, neu),
        "White": _pct(wh), "White vs Neu": _diff(wh, neu),
        "Black": _pct(bl), "Black vs Neu": _diff(bl, neu),
        "WM": _pct(wm_r), "WM vs Neu": _diff(wm_r, neu),
        "WF": _pct(wf_r), "WF vs Neu": _diff(wf_r, neu),
        "BM": _pct(bm_r), "BM vs Neu": _diff(bm_r, neu),
        "BF": _pct(bf_r), "BF vs Neu": _diff(bf_r, neu),
        "BF-WM": _diff(bf_r, wm_r),
        "p-value": f"{bf_wm_p:.4f}" if not np.isnan(bf_wm_p) else "N/A",
        "sig": _sig_stars(bf_wm_p),
    }


def print_deference_table(tsv: bool = False, level: str | None = None,
                          direction: str = "regressive", combined: bool = False,
                          max_questions: int = 0, split: str = "all"):
    """Deference per demographic from raw multi-turn JSONL results."""
    result_files = _discover_files()
    if not result_files:
        print("No multi-turn JSONL files found.")
        return

    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for fpath in result_files:
        if not fpath.endswith(".jsonl"):
            continue
        df = _load_results(fpath)
        if df.empty:
            continue
        invalid_mask = df["model_answer"].isin(["INVALID", "ERROR"])
        df = df[~invalid_mask].reset_index(drop=True)

        model = df["model"].iloc[0]
        domain = df["domain"].iloc[0] if "domain" in df.columns else "unknown"
        model_safe = model.replace("/", "_")
        if model_safe not in FOCUS_MODELS:
            continue

        sub = df[df["direction"] == direction]
        if model_safe in _MODEL_SHORT:
            sub = _apply_split(sub, model_safe, direction, domain, split)
        sub = _subsample_questions(sub, max_questions)
        if level:
            sub = sub[sub["escalation_level"] == level]
        if sub.empty:
            continue

        domain_label = "combined" if combined else domain
        groups.setdefault((model_safe, domain_label), []).append(sub)

    rows = []
    for (model_safe, domain_label), dfs in groups.items():
        pooled = pd.concat(dfs, ignore_index=True)
        rows.append(_compute_deference_row(model_safe, domain_label, pooled))

    rows.sort(key=lambda r: r["Model"])

    level_label = f" ({level.upper()})" if level else " (ALL LEVELS)"
    dir_label = direction.upper()
    _print_tables_by_domain(rows, f"MULTI-TURN DEFERENCE — {dir_label}{level_label}", tsv=tsv)


# ===================================================================
# Before / After comparison table
# ===================================================================

_METRIC_ROWS = [
    ("Overall deference", "overall", "pct"),
    ("Neutral", "neutral", "pct"),
    ("Male", "male", "pct"),
    ("Female", "female", "pct"),
    ("White", "white", "pct"),
    ("Black", "black", "pct"),
    ("White Male", "wm", "pct"),
    ("White Female", "wf", "pct"),
    ("Black Male", "bm", "pct"),
    ("Black Female", "bf", "pct"),
    ("BF-WM gap", "bf_wm_gap", "pp"),
    ("Cramer's V", "cramers_v", "v"),
    ("p-value", "bf_wm_p", "pval"),
]


def _fmt_metric(value: float, kind: str) -> str:
    if np.isnan(value):
        return "N/A"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "pp":
        return f"{value * 100:.1f}pp"
    if kind == "v":
        return f"{value:.3f}"
    if kind == "pval":
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"
    return str(value)


def _fmt_delta(before: float, after: float, kind: str) -> str:
    if np.isnan(before) or np.isnan(after):
        return "N/A"
    d = after - before
    if kind == "pct":
        return f"{d * 100:.1f}pp"
    if kind == "pp":
        return f"{d * 100:.1f}pp"
    if kind == "v":
        return f"{d:.3f}"
    if kind == "pval":
        return ""
    return str(d)


def _compute_mt_comparison_metrics(sub: pd.DataFrame) -> dict[str, float]:
    """Compute deference metrics from a multi-turn filtered DataFrame."""
    auth_neutral = sub[sub["variant"] == "sycophancy_authority"]
    auth_demo = sub[
        sub["variant"].str.startswith("sycophancy_authority_")
        & ~sub["variant"].str.startswith("sycophancy_authority_gender_")
        & ~sub["variant"].str.startswith("sycophancy_authority_race_")
    ]
    gender_only = sub[sub["variant"].str.startswith("sycophancy_authority_gender_")]
    race_only = sub[sub["variant"].str.startswith("sycophancy_authority_race_")]
    all_authority = pd.concat([auth_neutral, auth_demo, gender_only, race_only])

    def _rate(subset):
        return subset["deferred"].mean() if len(subset) > 0 else float("nan")

    neu = _rate(auth_neutral)
    bf_r = _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")])
    wm_r = _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")])

    n_bf = len(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "female")])
    n_wm = len(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "male")])
    bf_wm_p = float("nan")
    if n_bf > 0 and n_wm > 0 and not (np.isnan(bf_r) or np.isnan(wm_r)):
        _, bf_wm_p = _two_prop_z(n_bf, bf_r, n_wm, wm_r)

    cramers = float("nan")
    if len(auth_demo) > 0:
        ct = pd.crosstab(auth_demo["variant"], auth_demo["deferred"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            cramers = config.cramers_v(ct.values)

    return {
        "overall": _rate(all_authority),
        "neutral": neu,
        "male": _rate(gender_only[gender_only["gender"] == "male"]),
        "female": _rate(gender_only[gender_only["gender"] == "female"]),
        "white": _rate(race_only[race_only["race"] == "White"]),
        "black": _rate(race_only[race_only["race"] == "Black"]),
        "wm": wm_r,
        "wf": _rate(auth_demo[(auth_demo["race"] == "White") & (auth_demo["gender"] == "female")]),
        "bm": _rate(auth_demo[(auth_demo["race"] == "Black") & (auth_demo["gender"] == "male")]),
        "bf": bf_r,
        "bf_wm_gap": (bf_r - wm_r) if not (np.isnan(bf_r) or np.isnan(wm_r)) else float("nan"),
        "cramers_v": cramers,
        "bf_wm_p": bf_wm_p,
    }


def _print_comparison_sections(all_sections: list[tuple[str, list[dict]]], tsv: bool):
    """Print one or more model comparison sections."""
    for model_label, metric_rows in all_sections:
        if not metric_rows:
            continue
        cols = list(metric_rows[0].keys())
        if tsv:
            print(f"--- {model_label} ---")
            print("\t".join(cols))
            for row in metric_rows:
                print("\t".join(str(row[c]) for c in cols))
            print()
        else:
            col_widths = {}
            for c in cols:
                max_w = max(len(c), max(len(str(row[c])) for row in metric_rows))
                col_widths[c] = max_w
            header = "  ".join(c.ljust(col_widths[c]) for c in cols)
            sep = "  ".join("-" * col_widths[c] for c in cols)
            print(f"\n{'=' * len(header)}")
            print(f"{model_label} (Base vs DPO)")
            print(f"{'=' * len(header)}")
            print(header)
            print(sep)
            for row in metric_rows:
                print("  ".join(str(row[c]).ljust(col_widths[c]) for c in cols))
            print(f"{'=' * len(header)}\n")


def print_comparison_table(tsv: bool = False, combined: bool = False,
                           direction: str = "regressive",
                           level: str | None = None,
                           max_questions: int = 0,
                           split: str = "all"):
    """Before/after DPO comparison table for multi-turn deference."""
    result_files = _discover_files()
    if not result_files:
        print("No multi-turn JSONL files found.")
        return

    data: dict[tuple[str, str], pd.DataFrame] = {}
    for fpath in result_files:
        if not fpath.endswith(".jsonl"):
            continue
        df = _load_results(fpath)
        if df.empty:
            continue
        invalid_mask = df["model_answer"].isin(["INVALID", "ERROR"])
        df = df[~invalid_mask].reset_index(drop=True)
        model_safe = df["model"].iloc[0].replace("/", "_")
        domain = df["domain"].iloc[0] if "domain" in df.columns else "unknown"

        sub = df[df["direction"] == direction]
        if model_safe in _MODEL_SHORT:
            sub = _apply_split(sub, model_safe, direction, domain, split)
        sub = _subsample_questions(sub, max_questions)
        if level:
            sub = sub[sub["escalation_level"] == level]
        if sub.empty:
            continue
        data[(model_safe, domain)] = sub

    domains = ["combined"] if combined else ["medical", "legal"]

    all_sections = []
    for base_safe, dpo_safe, label in MODEL_PAIRS:
        domain_metrics: dict[str, tuple[dict, dict]] = {}

        for domain in domains:
            if domain == "combined":
                before_dfs = [data[(base_safe, d)] for d in ["medical", "legal"] if (base_safe, d) in data]
                after_dfs = [data[(dpo_safe, d)] for d in ["medical", "legal"] if (dpo_safe, d) in data]
                before_df = pd.concat(before_dfs, ignore_index=True) if before_dfs else pd.DataFrame()
                after_df = pd.concat(after_dfs, ignore_index=True) if after_dfs else pd.DataFrame()
            else:
                before_df = data.get((base_safe, domain), pd.DataFrame())
                after_df = data.get((dpo_safe, domain), pd.DataFrame())

            if before_df.empty and after_df.empty:
                continue

            m_before = _compute_mt_comparison_metrics(before_df) if not before_df.empty else {k: float("nan") for _, k, _ in _METRIC_ROWS}
            m_after = _compute_mt_comparison_metrics(after_df) if not after_df.empty else {k: float("nan") for _, k, _ in _METRIC_ROWS}
            domain_metrics[domain] = (m_before, m_after)

        if not domain_metrics:
            continue

        metric_rows = []
        for row_label, key, kind in _METRIC_ROWS:
            row: dict[str, str] = {"Metric": row_label}
            for domain in domains:
                if domain not in domain_metrics:
                    continue
                m_before, m_after = domain_metrics[domain]
                cap = domain.capitalize()
                row[f"{cap} Before"] = _fmt_metric(m_before[key], kind)
                row[f"{cap} After"] = _fmt_metric(m_after[key], kind)
                row[f"\u0394 {cap}"] = _fmt_delta(m_before[key], m_after[key], kind)
            metric_rows.append(row)

        all_sections.append((label, metric_rows))

    dir_label = direction.capitalize()
    level_label = f" @ {level}" if level else ""
    title_suffix = f" ({dir_label}{level_label})"
    for i, (lbl, rows) in enumerate(all_sections):
        all_sections[i] = (f"{lbl}{title_suffix}", rows)

    _print_comparison_sections(all_sections, tsv)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-turn escalating sycophancy results"
    )
    subparsers = parser.add_subparsers(dest="command")

    p_analyze = subparsers.add_parser("analyze", help="Analyze a result file (default when no subcommand)")
    p_analyze.add_argument("--file", type=str, default=None, help="Specific result file to analyze")

    p_sum = subparsers.add_parser("summary-table", help="Print cross-model summary table from all analysis CSVs")
    p_sum.add_argument("--simple", action="store_true", help="Print a simplified table")
    p_sum.add_argument("--level", type=str, default=None, choices=["authority", "justified", "citation"], help="Filter to a specific escalation level (use with --simple for full demographic breakdown)")
    p_sum.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")

    p_lvl = subparsers.add_parser("level-table", help="Print level × demographic breakdown table per model/domain")
    p_lvl.add_argument("--model", type=str, default=None, help="Filter to a specific model (substring match)")
    p_lvl.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")

    p_acc = subparsers.add_parser("accuracy-table", help="Print accuracy per demographic (regressive direction)")
    p_acc.add_argument("--level", type=str, default=None, choices=["simple", "authority", "justified", "citation"], help="Filter to a specific escalation level")
    p_acc.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")
    p_acc.add_argument("--combined", action="store_true", help="Pool medical + legal into a single combined table")

    p_def = subparsers.add_parser("deference-table", help="Print deference per demographic from raw JONLs")
    p_def.add_argument("--level", type=str, default=None, choices=["simple", "authority", "justified", "citation"])
    p_def.add_argument("--direction", type=str, default="regressive", choices=["regressive", "progressive"])
    p_def.add_argument("--tsv", action="store_true")
    p_def.add_argument("--combined", action="store_true", help="Pool medical + legal into a single combined table")
    p_def.add_argument("--max-questions", type=int, default=0, help="Subsample to N questions per file (0 = no cap)")
    p_def.add_argument("--split", default="all", choices=["train", "test", "all"],
                        help="Filter questions: train=DPO training set, test=held-out, all=no filter")

    p_cmp = subparsers.add_parser("comparison-table", help="Before/after DPO comparison table")
    p_cmp.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")
    p_cmp.add_argument("--combined", action="store_true", help="Show only combined (medical+legal pooled) columns")
    p_cmp.add_argument("--direction", type=str, default="regressive", choices=["regressive", "progressive"])
    p_cmp.add_argument("--level", type=str, default=None, choices=["simple", "authority", "justified", "citation"])
    p_cmp.add_argument("--max-questions", type=int, default=0, help="Subsample to N questions per file (0 = no cap)")
    p_cmp.add_argument("--split", required=True, choices=["train", "test", "all"],
                        help="Filter questions: train=DPO training set, test=held-out, all=no filter")

    # Support legacy usage: no subcommand + optional --file
    parser.add_argument("--file", type=str, default=None, help="Specific result file to analyze")

    args = parser.parse_args()

    if args.command == "comparison-table":
        print_comparison_table(
            tsv=args.tsv, combined=args.combined,
            direction=args.direction, level=args.level,
            max_questions=args.max_questions,
            split=args.split,
        )
        return

    if args.command == "deference-table":
        print_deference_table(
            tsv=args.tsv, level=args.level,
            direction=args.direction, combined=args.combined,
            max_questions=args.max_questions,
            split=getattr(args, "split", "all"),
        )
        return

    if args.command == "summary-table":
        print_summary_table(
            simple=getattr(args, "simple", False),
            tsv=getattr(args, "tsv", False),
            level=getattr(args, "level", None),
        )
        return

    if args.command == "level-table":
        print_level_table(tsv=getattr(args, "tsv", False), model_filter=getattr(args, "model", None))
        return

    if args.command == "accuracy-table":
        print_accuracy_table(
            tsv=getattr(args, "tsv", False),
            level=getattr(args, "level", None),
            combined=getattr(args, "combined", False),
        )
        return

    # Default: analyze (subcommand or bare --file)
    file_arg = getattr(args, "file", None)
    if file_arg:
        analyze_multiturn(file_arg)
    else:
        result_files = _discover_files()
        if not result_files:
            print("No multiturn sycophancy result files found in data/results/multi_turn/")
            return
        for filepath in result_files:
            analyze_multiturn(filepath)


if __name__ == "__main__":
    main()
