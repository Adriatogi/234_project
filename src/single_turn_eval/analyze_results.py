"""
Statistical analysis of sycophancy and demographic bias results.

Usage:
    python src/single_turn_eval/analyze_results.py sycophancy --file data/results/single_turn/sycophancy_..._legal.jsonl
    python src/single_turn_eval/analyze_results.py sycophancy   # auto-discover all sycophancy result files
    python src/single_turn_eval/analyze_results.py experiment1   # auto-discover all experiment1 result files

Accepts both JSONL and CSV inputs (auto-detected by extension).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

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
    jsonl_files = glob.glob(os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"{pattern_prefix}*.jsonl"))
    csv_files = glob.glob(os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"{pattern_prefix}*.csv"))
    all_files = jsonl_files + csv_files
    all_files = [f for f in all_files if "analysis_" not in os.path.basename(f)]
    return sorted(all_files)


def _jsonl_prefix_for_direction(direction: str) -> str:
    """Return the JSONL filename prefix for a given direction."""
    if direction == "progressive":
        return "sycophancy_progressive_no_cot_"
    return "sycophancy_regressive_no_cot_"


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
    analysis_path = os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"analysis_{basename}.csv")
    if len(flip_df) > 0:
        flip_df.insert(0, "total_responses", total_raw)
        flip_df.insert(1, "invalid_count", n_invalid)
        flip_df.insert(2, "invalid_pct", round(n_invalid / total_raw * 100, 2))
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

    # BF vs WM two-proportion z-test
    bf_rows = auth_demo_df[(auth_demo_df["race"] == "Black") & (auth_demo_df["gender"] == "female")]
    wm_rows = auth_demo_df[(auth_demo_df["race"] == "White") & (auth_demo_df["gender"] == "male")]
    bf_wm_pval = float("nan")
    if len(bf_rows) > 0 and len(wm_rows) > 0:
        n1, n2 = len(bf_rows), len(wm_rows)
        p1, p2 = bf_rows["deferred"].mean(), wm_rows["deferred"].mean()
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        if 0 < p_pool < 1:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se
            bf_wm_pval = 2 * (1 - norm.cdf(abs(z)))

    print(f"\n--- All Authority Variants vs Neutral Baseline ---")
    print(f"  {'Variant':<20} {'Deference':>10} {'vs Neutral':>11}")
    print(f"  {'-'*43}")
    print(f"  {'Neutral (baseline)':<20} {auth_neutral_rate:>10.1%} {'---':>11}")
    for lbl, val in [("Male", go_m), ("Female", go_f), ("White", ro_w), ("Black", ro_b),
                     ("White Male", wm), ("White Female", wf), ("Black Male", bm), ("Black Female", bf)]:
        print(f"  {lbl:<20} {val:>10.1%} {val - auth_neutral_rate:>+10.1%}p")

    if not np.isnan(bf_wm_pval):
        sig = "***" if bf_wm_pval < 0.001 else ("**" if bf_wm_pval < 0.01 else ("*" if bf_wm_pval < 0.05 else "ns"))
        print(f"\n  BF vs WM: diff={bf - wm:+.4f} ({bf - wm:+.1%}), p={bf_wm_pval:.4f} {sig}")

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
    q_df.insert(0, "total_responses", total_raw)
    q_df.insert(1, "invalid_count", n_invalid)
    q_df.insert(2, "invalid_pct", round(n_invalid / total_raw * 100, 2))
    q_df.insert(3, "neutral_rate", round(auth_neutral_rate, 4))
    q_df.insert(4, "male_rate", round(go_m, 4))
    q_df.insert(5, "female_rate", round(go_f, 4))
    q_df.insert(6, "white_rate", round(ro_w, 4))
    q_df.insert(7, "black_rate", round(ro_b, 4))
    q_df.insert(8, "white_male_rate", round(wm, 4))
    q_df.insert(9, "white_female_rate", round(wf, 4))
    q_df.insert(10, "black_male_rate", round(bm, 4))
    q_df.insert(11, "black_female_rate", round(bf, 4))
    q_df.insert(12, "bf_vs_wm_pval", round(bf_wm_pval, 4) if not np.isnan(bf_wm_pval) else float("nan"))

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

    # Cramer's V for demo authority (race x gender) — saved to CSV
    cramers_v_demo = float("nan")
    cramers_v_demo_p = float("nan")
    if len(auth_demo_df) > 0:
        ct = pd.crosstab(auth_demo_df["variant"], auth_demo_df["deferred"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2_val, p_val, _, _ = chi2_contingency(ct.values)
            cramers_v_demo = config.cramers_v(ct.values)
            cramers_v_demo_p = p_val

    basename = os.path.splitext(os.path.basename(filepath))[0]
    analysis_path = os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"analysis_{basename}.csv")
    q_df.insert(13, "cramers_v_demo", round(cramers_v_demo, 4) if not np.isnan(cramers_v_demo) else float("nan"))
    q_df.insert(14, "cramers_v_demo_p", round(cramers_v_demo_p, 4) if not np.isnan(cramers_v_demo_p) else float("nan"))
    q_df.to_csv(analysis_path, index=False)
    print(f"\n  Per-question breakdown saved to {analysis_path}")


# ===================================================================
# Summary table (cross-model / cross-domain)
# ===================================================================

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
        df = config.load_jsonl(fpath)
        result[(model_safe, domain)] = df["is_correct"].mean()
    return result


def _parse_sycophancy_analysis_files(direction: str = "regressive") -> list[tuple[str, str, str]]:
    """Discover no-COT analysis CSVs and return [(filepath, model_safe, domain)]."""
    if direction == "progressive":
        csv_prefix = "analysis_sycophancy_progressive_no_cot_"
    else:
        csv_prefix = "analysis_sycophancy_regressive_no_cot_"

    no_cot_pattern = os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"{csv_prefix}*.csv")
    no_cot_files = sorted(glob.glob(no_cot_pattern))

    results = []
    for fpath in no_cot_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        remainder = basename.replace(csv_prefix, "", 1)
        parts = remainder.rsplit("_", 1)
        model_safe = parts[0] if len(parts) == 2 else remainder
        domain = parts[1] if len(parts) == 2 else "?"
        results.append((fpath, model_safe, domain))
    return results


_RATE_KEYS = [
    "neutral_rate", "male_rate", "female_rate", "white_rate", "black_rate",
    "white_male_rate", "white_female_rate", "black_male_rate", "black_female_rate",
    "bf_vs_wm_pval", "cramers_v_demo", "cramers_v_demo_p",
]


def _format_summary_row(r: dict) -> dict:
    """Format a raw numeric summary row into display strings."""
    neu = r["neutral_rate"]

    def _pct(v):
        return f"{v:.1%}" if not np.isnan(v) else "N/A"

    def _delta(v):
        return f"{v - neu:+.1%}" if not (np.isnan(v) or np.isnan(neu)) else "N/A"

    return {
        "Model": r["model_safe"],
        "Domain": r["domain"],
        "Base Acc": f"{r['base_acc']:.1%}" if not np.isnan(r["base_acc"]) else "N/A",
        "N": int(r["N"]),
        "Invalid": int(r["invalid"]),
        "Inv%": f"{r['invalid_pct']:.1f}%",
        "Neutral": _pct(neu),
        "Male": _pct(r["male_rate"]), "M vs Neu": _delta(r["male_rate"]),
        "Female": _pct(r["female_rate"]), "F vs Neu": _delta(r["female_rate"]),
        "White": _pct(r["white_rate"]), "W vs Neu": _delta(r["white_rate"]),
        "Black": _pct(r["black_rate"]), "B vs Neu": _delta(r["black_rate"]),
        "WM": _pct(r["white_male_rate"]), "WM vs Neu": _delta(r["white_male_rate"]),
        "WF": _pct(r["white_female_rate"]), "WF vs Neu": _delta(r["white_female_rate"]),
        "BM": _pct(r["black_male_rate"]), "BM vs Neu": _delta(r["black_male_rate"]),
        "BF": _pct(r["black_female_rate"]), "BF vs Neu": _delta(r["black_female_rate"]),
        "BF vs WM": f"{r['black_female_rate'] - r['white_male_rate']:+.1%}"
            if not (np.isnan(r["black_female_rate"]) or np.isnan(r["white_male_rate"])) else "N/A",
        "BF-WM p": f"{r['bf_vs_wm_pval']:.4f}" if not np.isnan(r["bf_vs_wm_pval"]) else "N/A",
        "sig": _sig_stars(r["bf_vs_wm_pval"]),
        "Cramer V": f"{r['cramers_v_demo']:.3f}" if not np.isnan(r["cramers_v_demo"]) else "N/A",
        "V p": f"{r['cramers_v_demo_p']:.4f}" if not np.isnan(r["cramers_v_demo_p"]) else "N/A",
    }


def print_summary_table(simple: bool = False, tsv: bool = False, combined: bool = False,
                        direction: str = "regressive", max_questions: int = 0):
    """Read all analysis_sycophancy_*.csv files and print a consolidated table."""
    all_files = _parse_sycophancy_analysis_files(direction=direction)
    if not all_files:
        print("No analysis CSVs found. Run evaluate_single_turn.sh first.")
        return

    baseline_acc = _load_baseline_accuracy()

    raw_rows = []
    for fpath, model_safe, domain in all_files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        row = df.iloc[0]
        if model_safe not in FOCUS_MODELS:
            continue
        raw = {
            "model_safe": model_safe,
            "domain": domain,
            "base_acc": baseline_acc.get((model_safe, domain), float("nan")),
            "N": int(row.get("total_responses", 0)),
            "invalid": int(row.get("invalid_count", 0)),
            "invalid_pct": row.get("invalid_pct", float("nan")),
        }
        for k in _RATE_KEYS:
            raw[k] = row.get(k, float("nan"))
        raw_rows.append(raw)

    if not raw_rows:
        print("No data rows found in analysis CSVs.")
        return

    if combined:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in raw_rows:
            groups[r["model_safe"]].append(r)
        raw_rows = []
        for model_safe, grp in groups.items():
            total_n = sum(r["N"] for r in grp)
            total_inv = sum(r["invalid"] for r in grp)
            merged = {
                "model_safe": model_safe,
                "domain": "combined",
                "N": total_n,
                "invalid": total_inv,
                "invalid_pct": (total_inv / (total_n + total_inv) * 100) if (total_n + total_inv) > 0 else float("nan"),
            }
            base_vals = [(r["base_acc"], r["N"]) for r in grp if not np.isnan(r["base_acc"])]
            merged["base_acc"] = (sum(v * n for v, n in base_vals) / sum(n for _, n in base_vals)) if base_vals else float("nan")
            for k in _RATE_KEYS:
                vals = [(r[k], r["N"]) for r in grp if not np.isnan(r[k])]
                merged[k] = (sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else float("nan")
            raw_rows.append(merged)

    rows = [_format_summary_row(r) for r in raw_rows]
    rows.sort(key=lambda r: r["Model"])

    if simple:
        _print_tables_by_domain(
            rows,
            "SINGLE-TURN SYCOPHANCY (NO-COT)",
            columns=["Model", "Base Acc", "Neutral", "White", "Black", "Female", "Male", "BF", "WM", "BF vs WM", "BF-WM p", "sig"],
            tsv=tsv,
        )
    else:
        _print_tables_by_domain(rows, "SINGLE-TURN SYCOPHANCY (NO-COT)", tsv=tsv)


# ===================================================================
# Accuracy table (cross-model / cross-domain)
# ===================================================================

def _accuracy_two_prop_z(n1: int, p1: float, n2: int, p2: float) -> float:
    """Two-proportion z-test on accuracy rates. Returns p-value."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return float("nan")
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return 2 * (1 - norm.cdf(abs(z)))


def print_accuracy_table(tsv: bool = False, combined: bool = False,
                         direction: str = "regressive", max_questions: int = 0,
                         split: str = "all"):
    """Accuracy per demographic from raw JSONL sycophancy results."""
    prefix = _jsonl_prefix_for_direction(direction)
    no_cot_pattern = os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"{prefix}*.jsonl")
    all_files = sorted(glob.glob(no_cot_pattern))

    if not all_files:
        print("No no-COT sycophancy JSONL files found.")
        return

    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for fpath in all_files:
        df = config.load_jsonl(fpath)
        if df.empty:
            continue
        invalid_mask = df["model_answer"].isin(["INVALID", "ERROR"])
        df = df[~invalid_mask].reset_index(drop=True)

        model = df["model"].iloc[0]
        domain = df["domain"].iloc[0]
        model_safe = model.replace("/", "_")
        if model_safe in _MODEL_SHORT:
            df = _apply_split(df, model_safe, direction, domain, split)
        df = _subsample_questions(df, max_questions)

        if model_safe not in FOCUS_MODELS:
            continue

        domain_label = "combined" if combined else domain
        groups.setdefault((model_safe, domain_label), []).append(df)

    def _pct(v):
        return f"{v:.1%}" if not np.isnan(v) else "N/A"

    def _diff(v, ref):
        if np.isnan(v) or np.isnan(ref):
            return "N/A"
        d = (v - ref) * 100
        return f"{d:.1f}pp" if d >= 0 else f"{d:.1f}pp"

    rows = []
    for (model_safe, domain_label), dfs in groups.items():
        df = pd.concat(dfs, ignore_index=True)

        auth_neutral = df[df["variant"] == "sycophancy_authority"]
        auth_demo = df[
            df["variant"].str.startswith("sycophancy_authority_")
            & ~df["variant"].str.startswith("sycophancy_authority_gender_")
            & ~df["variant"].str.startswith("sycophancy_authority_race_")
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

        bf_wm_p = _accuracy_two_prop_z(len(bf_df), bf_acc, len(wm_df), wm_acc)

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

    _print_tables_by_domain(rows, "SINGLE-TURN ACCURACY BY DEMOGRAPHIC (NO-COT)", tsv=tsv)


# ===================================================================
# Before / After comparison table
# ===================================================================

def _compute_comparison_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute deference metrics from a single-turn sycophancy DataFrame.

    Returns dict with keys: overall, neutral, male, female, white, black,
    wm, wf, bm, bf, bf_wm_gap, cramers_v, bf_wm_p.
    """
    auth_neutral = df[df["variant"] == "sycophancy_authority"]
    auth_demo = df[
        df["variant"].str.startswith("sycophancy_authority_")
        & ~df["variant"].str.startswith("sycophancy_authority_gender_")
        & ~df["variant"].str.startswith("sycophancy_authority_race_")
    ]
    gender_only = df[df["variant"].str.startswith("sycophancy_authority_gender_")]
    race_only = df[df["variant"].str.startswith("sycophancy_authority_race_")]
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
        p_pool = (bf_r * n_bf + wm_r * n_wm) / (n_bf + n_wm)
        if 0 < p_pool < 1:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_bf + 1 / n_wm))
            if se > 0:
                bf_wm_p = 2 * (1 - norm.cdf(abs((bf_r - wm_r) / se)))

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


def _print_comparison_sections(all_sections: list[tuple[str, list[dict]]], tsv: bool):
    """Print one or more model comparison sections.

    Each section is (model_label, metric_rows) where metric_rows is a list of
    dicts with keys: label, and then per-domain triplets like
    medical_before, medical_after, d_medical, etc.
    """
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
                           direction: str = "regressive", max_questions: int = 0,
                           split: str = "all"):
    """Before/after DPO comparison table for single-turn."""
    prefix = _jsonl_prefix_for_direction(direction)
    no_cot_pattern = os.path.join(config.SINGLE_TURN_RESULTS_DIR, f"{prefix}*.jsonl")
    all_files = sorted(glob.glob(no_cot_pattern))
    if not all_files:
        print(f"No {direction} sycophancy JSONL files found.")
        return

    data: dict[tuple[str, str], pd.DataFrame] = {}
    for fpath in all_files:
        df = config.load_jsonl(fpath)
        if df.empty:
            continue
        invalid_mask = df["model_answer"].isin(["INVALID", "ERROR"])
        df = df[~invalid_mask].reset_index(drop=True)
        model_safe = df["model"].iloc[0].replace("/", "_")
        domain = df["domain"].iloc[0]
        if model_safe in _MODEL_SHORT:
            df = _apply_split(df, model_safe, direction, domain, split)
        df = _subsample_questions(df, max_questions)
        data[(model_safe, domain)] = df

    domains = ["combined"] if combined else ["medical", "legal"]

    all_sections = []
    for base_safe, dpo_safe, label in MODEL_PAIRS:
        metric_rows = []
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

            m_before = _compute_comparison_metrics(before_df) if not before_df.empty else {k: float("nan") for _, k, _ in _METRIC_ROWS}
            m_after = _compute_comparison_metrics(after_df) if not after_df.empty else {k: float("nan") for _, k, _ in _METRIC_ROWS}
            domain_metrics[domain] = (m_before, m_after)

        if not domain_metrics:
            continue

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

    _print_comparison_sections(all_sections, tsv)


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

    p_sum = subparsers.add_parser("summary-table", help="Print cross-model summary table from all analysis CSVs")
    p_sum.add_argument("--simple", action="store_true", help="Print a simplified table (Model, White, Black, Female, Male, BF, WM, BF–WM)")
    p_sum.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")
    p_sum.add_argument("--combined", action="store_true", help="Pool medical + legal into a single combined table")
    p_sum.add_argument("--direction", default="regressive", choices=["regressive", "progressive"])
    p_sum.add_argument("--max-questions", type=int, default=0, help="Subsample to N questions per file (0 = no cap)")

    p_acc = subparsers.add_parser("accuracy-table", help="Print accuracy per demographic from raw sycophancy results")
    p_acc.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")
    p_acc.add_argument("--combined", action="store_true", help="Pool medical + legal into a single combined table")
    p_acc.add_argument("--direction", default="regressive", choices=["regressive", "progressive"])
    p_acc.add_argument("--max-questions", type=int, default=0, help="Subsample to N questions per file (0 = no cap)")
    p_acc.add_argument("--split", default="all", choices=["train", "test", "all"],
                        help="Filter questions: train=DPO training set, test=held-out, all=no filter")

    p_cmp = subparsers.add_parser("comparison-table", help="Before/after DPO comparison table")
    p_cmp.add_argument("--tsv", action="store_true", help="Output tab-separated values for pasting into Google Sheets")
    p_cmp.add_argument("--combined", action="store_true", help="Show only combined (medical+legal pooled) columns")
    p_cmp.add_argument("--direction", default="regressive", choices=["regressive", "progressive"])
    p_cmp.add_argument("--max-questions", type=int, default=0, help="Subsample to N questions per file (0 = no cap)")
    p_cmp.add_argument("--split", required=True, choices=["train", "test", "all"],
                        help="Filter questions: train=DPO training set, test=held-out, all=no filter")

    args = parser.parse_args()

    if args.command == "comparison-table":
        print_comparison_table(tsv=args.tsv, combined=args.combined,
                               direction=args.direction, max_questions=args.max_questions,
                               split=args.split)
        return

    if args.command == "summary-table":
        print_summary_table(simple=args.simple, tsv=args.tsv,
                            combined=args.combined, direction=args.direction,
                            max_questions=args.max_questions)
        return

    if args.command == "accuracy-table":
        print_accuracy_table(tsv=args.tsv, combined=args.combined,
                             direction=args.direction, max_questions=args.max_questions,
                             split=getattr(args, "split", "all"))
        return

    if args.command == "experiment1":
        if args.file:
            analyze_experiment1(args.file)
        else:
            result_files = _discover_files("together_ai_")
            if not result_files:
                print("No experiment1 result files found in data/results/single_turn/")
                return
            for filepath in result_files:
                analyze_experiment1(filepath)

    elif args.command == "sycophancy":
        if args.file:
            analyze_sycophancy(args.file)
        else:
            result_files = _discover_files("sycophancy_")
            if not result_files:
                print("No sycophancy result files found in data/results/single_turn/")
                return
            for filepath in result_files:
                analyze_sycophancy(filepath)


if __name__ == "__main__":
    main()
