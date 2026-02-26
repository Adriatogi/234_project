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
    python src/multi_turn_eval/analyze_results.py --file data/results/multiturn_sycophancy_..._legal.jsonl
    python src/multi_turn_eval/analyze_results.py   # auto-discover all multiturn result files
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

LEVEL_ORDER = ["simple", "authority", "justified", "citation"]

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
    jsonl = glob.glob(os.path.join(config.RESULTS_DIR, "multiturn_sycophancy_*.jsonl"))
    csv = glob.glob(os.path.join(config.RESULTS_DIR, "multiturn_sycophancy_*.csv"))
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
        question_stats.append(row_data)

    q_df = pd.DataFrame(question_stats).sort_values("total_deference", ascending=False)

    basename = os.path.splitext(os.path.basename(filepath))[0]
    analysis_path = os.path.join(config.RESULTS_DIR, f"analysis_{basename}.csv")
    q_df.to_csv(analysis_path, index=False)
    print(f"\n  Per-question breakdown saved to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-turn escalating sycophancy results"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Specific result file to analyze",
    )
    args = parser.parse_args()

    if args.file:
        analyze_multiturn(args.file)
    else:
        result_files = _discover_files()
        if not result_files:
            print("No multiturn sycophancy result files found in data/results/")
            return
        for filepath in result_files:
            analyze_multiturn(filepath)


if __name__ == "__main__":
    main()
