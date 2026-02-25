"""
Statistical analysis of sycophancy and demographic bias results.

Usage:
    python src/eval/analyze_results.py sycophancy --file data/results/sycophancy_..._legal.jsonl
    python src/eval/analyze_results.py sycophancy   # auto-discover all sycophancy result files
    python src/eval/analyze_results.py experiment1   # auto-discover all experiment1 result files

Accepts both JSONL and CSV inputs (auto-detected by extension).
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

LAYER_ORDER = [
    ("User", "sycophancy_user"),
    ("Neutral Authority", "sycophancy_authority"),
    ("Demographic Authority", "sycophancy_authority_", ["sycophancy_authority_gender_", "sycophancy_authority_race_"]),
    ("Gender-Only Authority", "sycophancy_authority_gender_"),
    ("Race-Only Authority", "sycophancy_authority_race_"),
    ("Person Demographic", "sycophancy_person_"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(filepath: str) -> pd.DataFrame:
    """Load results from JSONL or CSV based on file extension."""
    if filepath.endswith(".jsonl"):
        return config.load_jsonl(filepath)
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unknown file extension: {filepath}")


def _discover_files(pattern_prefix: str) -> list[str]:
    """Find result files matching a prefix (both JSONL and CSV)."""
    jsonl_files = glob.glob(os.path.join(config.RESULTS_DIR, f"{pattern_prefix}*.jsonl"))
    csv_files = glob.glob(os.path.join(config.RESULTS_DIR, f"{pattern_prefix}*.csv"))
    all_files = jsonl_files + csv_files
    all_files = [f for f in all_files if "analysis_" not in os.path.basename(f)]
    return sorted(all_files)


def _print_deference_breakdown(
    title: str,
    subset_df: pd.DataFrame,
    baseline_rate: float,
    races: list[str] | None = None,
    genders: list[str] | None = None,
    label_suffix: str = "",
):
    """Print a deference-rate table broken down by race and/or gender."""
    if len(subset_df) == 0:
        return
    print(f"\n--- {title} ---")
    print(f"  Baseline: {baseline_rate:.4f}")
    print(f"\n  {'Group':<25} {'Deference':>12} {'Delta':>12} {'N':>6}")
    print(f"  {'-'*57}")

    def _row(label, sub):
        if len(sub) == 0:
            return
        rate = sub["deferred"].mean()
        print(f"  {label:<25} {rate:>12.4f} {rate - baseline_rate:>+12.4f} {len(sub):>6}")

    if races:
        for race in races:
            _row(f"{race}{label_suffix}", subset_df[subset_df["race"] == race])
        if genders:
            print()

    if genders:
        for gender in genders:
            _row(f"{gender}{label_suffix}", subset_df[subset_df["gender"] == gender])

    if races and genders:
        print()
        for race in races:
            for gender in genders:
                sub = subset_df[(subset_df["race"] == race) & (subset_df["gender"] == gender)]
                _row(f"{race} {gender}", sub)


# ===================================================================
# EXPERIMENT1 subcommand — demographic counterfactual analysis
# ===================================================================

def analyze_experiment1(filepath: str):
    df_raw = _load_results(filepath)
    model = df_raw["model"].iloc[0]
    prompt = df_raw["prompt_name"].iloc[0]

    total_raw = len(df_raw)
    invalid_mask = df_raw["model_answer"].isin(["INVALID", "ERROR"])
    n_invalid = invalid_mask.sum()
    df = df_raw[~invalid_mask].reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: DEMOGRAPHIC COUNTERFACTUAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Model:  {model}")
    print(f"Prompt: {prompt}")
    print(f"Total:  {total_raw}")
    if n_invalid > 0:
        print(f"Excluded: {n_invalid} INVALID/ERROR ({n_invalid / total_raw * 100:.1f}%)")
    print(f"Analyzed: {len(df)}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"{'='*70}")

    overall_acc = df["is_correct"].mean()
    neutral_acc = df[df["variant"] == "neutral"]["is_correct"].mean()
    demo_acc = df[df["variant"] != "neutral"]["is_correct"].mean()

    print(f"\n--- Overall Accuracy ---")
    print(f"  All variants:    {overall_acc:.4f}")
    print(f"  Neutral:         {neutral_acc:.4f}")
    print(f"  Demographic avg: {demo_acc:.4f}")
    print(f"  Delta:           {demo_acc - neutral_acc:+.4f}")

    print(f"\n--- Accuracy by Race ---")
    print(f"  {'Race':<12} {'Accuracy':>10} {'Delta vs Neutral':>18}")
    print(f"  {'-'*42}")
    print(f"  {'Neutral':<12} {neutral_acc:>10.4f} {'---':>18}")
    for race in config.RACES_EXTENDED:
        race_df = df[df["race"] == race]
        if len(race_df) == 0:
            continue
        acc = race_df["is_correct"].mean()
        delta = acc - neutral_acc
        print(f"  {race:<12} {acc:>10.4f} {delta:>+18.4f}")

    print(f"\n--- Accuracy by Gender ---")
    print(f"  {'Gender':<12} {'Accuracy':>10} {'Delta vs Neutral':>18}")
    print(f"  {'-'*42}")
    print(f"  {'Neutral':<12} {neutral_acc:>10.4f} {'---':>18}")
    for gender in config.GENDERS:
        gender_df = df[df["gender"] == gender]
        if len(gender_df) == 0:
            continue
        acc = gender_df["is_correct"].mean()
        delta = acc - neutral_acc
        print(f"  {gender:<12} {acc:>10.4f} {delta:>+18.4f}")

    print(f"\n--- Accuracy by Race x Gender ---")
    print(f"  {'Group':<20} {'Accuracy':>10} {'Delta vs Neutral':>18}")
    print(f"  {'-'*50}")
    for race in config.RACES_EXTENDED:
        for gender in config.GENDERS:
            group_df = df[(df["race"] == race) & (df["gender"] == gender)]
            if len(group_df) == 0:
                continue
            acc = group_df["is_correct"].mean()
            delta = acc - neutral_acc
            print(f"  {race + ' ' + gender:<20} {acc:>10.4f} {delta:>+18.4f}")

    flipped_questions = []
    for qid in df["question_id"].unique():
        qdf = df[df["question_id"] == qid]
        neutral_rows = qdf[qdf["variant"] == "neutral"]
        if len(neutral_rows) == 0:
            continue
        neutral_answer = neutral_rows["model_answer"].iloc[0]
        demo_answers = qdf[qdf["variant"] != "neutral"]["model_answer"]
        flips = (demo_answers != neutral_answer).sum()
        if flips > 0:
            flipped_questions.append({
                "question_id": qid,
                "neutral_answer": neutral_answer,
                "correct_answer": qdf["correct_answer"].iloc[0],
                "flips": flips,
                "flip_rate": flips / len(demo_answers),
                "flipped_to": demo_answers[demo_answers != neutral_answer].value_counts().to_dict(),
            })

    flip_df = pd.DataFrame(flipped_questions)
    if len(flip_df) > 0:
        flip_df = flip_df.sort_values("flips", ascending=False)

    print(f"\n--- Answer Flips ---")
    print(f"  Questions with any flip: {len(flip_df)} / {df['question_id'].nunique()}")
    if len(flip_df) > 0:
        print(f"  Average flip rate (among flipped): {flip_df['flip_rate'].mean():.4f}")
        print(f"\n  Top 10 most-flipped questions:")
        for _, row in flip_df.head(10).iterrows():
            print(f"    Q{row['question_id']}: {row['flips']} flips "
                  f"(neutral={row['neutral_answer']}, correct={row['correct_answer']}) "
                  f"-> {row['flipped_to']}")

    v_values = []
    for qid in df["question_id"].unique():
        qdf = df[(df["question_id"] == qid) & (df["variant"] != "neutral")]
        if len(qdf) == 0:
            continue
        ct = pd.crosstab(qdf["variant"], qdf["is_correct"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            v = config.cramers_v(ct.values)
            v_values.append({"question_id": qid, "cramers_v": v})

    if v_values:
        v_df = pd.DataFrame(v_values)
        print(f"\n--- Cramer's V (effect size of demographics on correctness) ---")
        print(f"  Mean:   {v_df['cramers_v'].mean():.4f}")
        print(f"  Median: {v_df['cramers_v'].median():.4f}")
        print(f"  Max:    {v_df['cramers_v'].max():.4f}")
        print(f"  Questions with V > 0.3: {(v_df['cramers_v'] > 0.3).sum()}")

    basename = os.path.splitext(os.path.basename(filepath))[0]
    analysis_path = os.path.join(config.RESULTS_DIR, f"analysis_{basename}.csv")
    if len(flip_df) > 0:
        flip_df.to_csv(analysis_path, index=False)
        print(f"\n  Detailed analysis saved to {analysis_path}")


# ===================================================================
# SYCOPHANCY subcommand — deference pattern analysis
# ===================================================================

def analyze_sycophancy(filepath: str):
    df_raw = _load_results(filepath)
    model = df_raw["model"].iloc[0]
    domain = df_raw["domain"].iloc[0] if "domain" in df_raw.columns else "unknown"

    total_raw = len(df_raw)
    invalid_mask = df_raw["model_answer"].isin(["INVALID", "ERROR"])
    n_invalid = invalid_mask.sum()
    df = df_raw[~invalid_mask].reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"SYCOPHANCY ANALYSIS")
    print(f"{'='*70}")
    print(f"Model:    {model}")
    print(f"Domain:   {domain}")
    print(f"Total:    {total_raw}")
    if n_invalid > 0:
        print(f"Excluded: {n_invalid} INVALID/ERROR ({n_invalid / total_raw * 100:.1f}%)")
    print(f"Analyzed: {len(df)}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"{'='*70}")

    overall_deference = df["deferred"].mean()
    overall_accuracy = df["is_correct"].mean()
    print(f"\n--- Overall ---")
    print(f"  Deference rate (chose suggested wrong answer): {overall_deference:.4f} "
          f"({df['deferred'].sum()}/{len(df)})")
    print(f"  Accuracy: {overall_accuracy:.4f}")

    print(f"\n--- Sycophancy Ladder ---")
    print(f"  {'Layer':<30} {'Deference':>12} {'Accuracy':>12} {'N':>6}")
    print(f"  {'-'*62}")

    layer_rates = {}
    for layer_entry in LAYER_ORDER:
        if len(layer_entry) == 3:
            layer_name, variant_match, exclude_prefixes = layer_entry
        else:
            layer_name, variant_match = layer_entry
            exclude_prefixes = []

        if variant_match.endswith("_"):
            layer_df = df[df["variant"].str.startswith(variant_match)]
            for exc in exclude_prefixes:
                layer_df = layer_df[~layer_df["variant"].str.startswith(exc)]
        else:
            layer_df = df[df["variant"] == variant_match]

        if len(layer_df) == 0:
            continue

        deference = layer_df["deferred"].mean()
        accuracy = layer_df["is_correct"].mean()
        layer_rates[layer_name] = deference

        print(f"  {layer_name:<30} {deference:>12.4f} {accuracy:>12.4f} {len(layer_df):>6}")

    if "User" in layer_rates and "Neutral Authority" in layer_rates:
        escalation = layer_rates["Neutral Authority"] - layer_rates["User"]
        print(f"\n  Authority escalation (Authority - User): {escalation:+.4f}")
    if "Neutral Authority" in layer_rates and "Demographic Authority" in layer_rates:
        demo_esc = layer_rates["Demographic Authority"] - layer_rates["Neutral Authority"]
        print(f"  Demographic effect (Demo Auth - Neutral): {demo_esc:+.4f}")

    # All authority variants vs neutral baseline
    auth_neutral_df = df[df["variant"] == "sycophancy_authority"]
    auth_neutral_rate = auth_neutral_df["deferred"].mean()

    auth_demo_df = df[
        df["variant"].str.startswith("sycophancy_authority_")
        & ~df["variant"].str.startswith("sycophancy_authority_gender_")
        & ~df["variant"].str.startswith("sycophancy_authority_race_")
    ]
    gender_only_df = df[df["variant"].str.startswith("sycophancy_authority_gender_")]
    race_only_df = df[df["variant"].str.startswith("sycophancy_authority_race_")]

    def _var_rate(subset):
        return subset["deferred"].mean() if len(subset) > 0 else float("nan")

    go_m = _var_rate(gender_only_df[gender_only_df["gender"] == "male"])
    go_f = _var_rate(gender_only_df[gender_only_df["gender"] == "female"])
    ro_w = _var_rate(race_only_df[race_only_df["race"] == "White"])
    ro_b = _var_rate(race_only_df[race_only_df["race"] == "Black"])
    wm = _var_rate(auth_demo_df[(auth_demo_df["race"] == "White") & (auth_demo_df["gender"] == "male")])
    wf = _var_rate(auth_demo_df[(auth_demo_df["race"] == "White") & (auth_demo_df["gender"] == "female")])
    bm = _var_rate(auth_demo_df[(auth_demo_df["race"] == "Black") & (auth_demo_df["gender"] == "male")])
    bf = _var_rate(auth_demo_df[(auth_demo_df["race"] == "Black") & (auth_demo_df["gender"] == "female")])

    print(f"\n--- All Authority Variants vs Neutral Baseline ---")
    print(f"  {'Variant':<20} {'Deference':>10} {'vs Neutral':>11}")
    print(f"  {'-'*43}")
    print(f"  {'Neutral (baseline)':<20} {auth_neutral_rate:>10.1%} {'---':>11}")
    for lbl, val in [("Male", go_m), ("Female", go_f), ("White", ro_w), ("Black", ro_b),
                     ("White Male", wm), ("White Female", wf), ("Black Male", bm), ("Black Female", bf)]:
        print(f"  {lbl:<20} {val:>10.1%} {val - auth_neutral_rate:>+10.1%}p")

    _print_deference_breakdown(
        "Demographic Authority: Deference by Race x Gender",
        auth_demo_df, auth_neutral_rate,
        races=config.RACES, genders=config.GENDERS,
    )

    _print_deference_breakdown(
        "Gender-Only Authority: Deference by Gender",
        gender_only_df, auth_neutral_rate,
        genders=config.GENDERS, label_suffix=" (gender only)",
    )

    _print_deference_breakdown(
        "Race-Only Authority: Deference by Race",
        race_only_df, auth_neutral_rate,
        races=config.RACES, label_suffix=" (race only)",
    )

    person_demo_df = df[df["variant"].str.startswith("sycophancy_person_")]
    _print_deference_breakdown(
        "Person Demographic: Deference by Subject Demographics",
        person_demo_df, auth_neutral_rate,
        races=config.RACES, genders=config.GENDERS,
    )

    question_stats = []
    for qid in sorted(df["question_id"].unique()):
        qdf = df[df["question_id"] == qid]

        user_rows = qdf[qdf["variant"] == "sycophancy_user"]
        auth_rows = qdf[qdf["variant"] == "sycophancy_authority"]
        if len(user_rows) == 0 or len(auth_rows) == 0:
            continue

        q_user = user_rows["deferred"].iloc[0]
        q_auth = auth_rows["deferred"].iloc[0]
        q_demo_auth = qdf[qdf["variant"].str.startswith("sycophancy_authority_")]["deferred"]
        q_person = qdf[qdf["variant"].str.startswith("sycophancy_person_")]["deferred"]

        question_stats.append({
            "question_id": qid,
            "correct_answer": qdf["correct_answer"].iloc[0],
            "suggested_answer": qdf["suggested_answer"].iloc[0],
            "user_deferred": q_user,
            "authority_deferred": q_auth,
            "demo_authority_deference_rate": q_demo_auth.mean(),
            "person_demo_deference_rate": q_person.mean(),
            "total_deference_rate": qdf["deferred"].mean(),
        })

    q_df = pd.DataFrame(question_stats).sort_values("total_deference_rate", ascending=False)

    print(f"\n--- Per-Question Deference (top 10 most sycophantic) ---")
    print(f"  {'QID':>5} {'Correct':>8} {'Suggested':>10} "
          f"{'User':>6} {'Auth':>6} {'DemoAuth':>9} {'Person':>8} {'Total':>7}")
    print(f"  {'-'*67}")
    for _, row in q_df.head(10).iterrows():
        print(f"  {row['question_id']:>5} {row['correct_answer']:>8} "
              f"{row['suggested_answer']:>10} "
              f"{'Y' if row['user_deferred'] else 'N':>6} "
              f"{'Y' if row['authority_deferred'] else 'N':>6} "
              f"{row['demo_authority_deference_rate']:>9.2f} "
              f"{row['person_demo_deference_rate']:>8.2f} "
              f"{row['total_deference_rate']:>7.2f}")

    never_caved = (q_df["total_deference_rate"] == 0).sum()
    always_caved = (q_df["total_deference_rate"] == 1).sum()
    print(f"\n  Never caved (0% deference):   {never_caved}/{len(q_df)}")
    print(f"  Always caved (100% deference): {always_caved}/{len(q_df)}")

    def _print_cramers(label: str, subset_df: pd.DataFrame):
        if len(subset_df) == 0:
            return
        ct = pd.crosstab(subset_df["variant"], subset_df["deferred"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            v = config.cramers_v(ct.values)
            chi2, p_val, _, _ = chi2_contingency(ct.values)
            print(f"\n--- Cramer's V ({label}) ---")
            print(f"  V = {v:.4f}, chi2 = {chi2:.4f}, p = {p_val:.4f}")
            if p_val < 0.05:
                print(f"  ** Statistically significant (p < 0.05)")
            else:
                print(f"  Not significant (p >= 0.05)")

    _print_cramers("Demographic Authority: race x gender -> deference", auth_demo_df)
    _print_cramers("Gender-Only Authority: gender -> deference", gender_only_df)
    _print_cramers("Race-Only Authority: race -> deference", race_only_df)
    _print_cramers("Person Demographic: subject demographics -> deference", person_demo_df)

    basename = os.path.splitext(os.path.basename(filepath))[0]
    analysis_path = os.path.join(config.RESULTS_DIR, f"analysis_{basename}.csv")
    q_df.to_csv(analysis_path, index=False)
    print(f"\n  Per-question breakdown saved to {analysis_path}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_exp = subparsers.add_parser("experiment1", help="Analyze demographic counterfactual results")
    p_exp.add_argument("--file", type=str, default=None, help="Specific result file to analyze")

    p_syc = subparsers.add_parser("sycophancy", help="Analyze sycophancy deference results")
    p_syc.add_argument("--file", type=str, default=None, help="Specific result file to analyze")

    args = parser.parse_args()

    if args.command == "experiment1":
        if args.file:
            analyze_experiment1(args.file)
        else:
            result_files = _discover_files("together_ai_")
            if not result_files:
                print("No experiment1 result files found in data/results/")
                return
            for filepath in result_files:
                analyze_experiment1(filepath)

    elif args.command == "sycophancy":
        if args.file:
            analyze_sycophancy(args.file)
        else:
            result_files = _discover_files("sycophancy_")
            if not result_files:
                print("No sycophancy result files found in data/results/")
                return
            for filepath in result_files:
                analyze_sycophancy(filepath)


if __name__ == "__main__":
    main()
