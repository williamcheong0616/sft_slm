#!/usr/bin/env python3
"""
Analyze human evaluation results.

Computes:
  - Average scores per model variant and dimension
  - Inter-annotator agreement per dimension:
      Krippendorff's alpha (ordinal) — appropriate for 1–5 Likert ratings,
                                       uses every unit rated by ≥ 2 annotators
      Fleiss' kappa (nominal)        — kept for comparison, complete units only
  - Pairwise model comparisons: Wilcoxon signed-rank tests between variants per
    dimension (items = questions, scores averaged over annotators), with Holm
    correction over all variant pairs × dimensions

Run AFTER all annotators have saved their ratings:
  python human_eval_analyze.py

Reads:  eval_results/human_eval_*.json
Writes: eval_results/human_eval_summary.xlsx
        eval_results/human_eval_kappa.json      (kappa + alpha per dimension)
        eval_results/human_eval_wilcoxon.csv

Requirements:
  pip install pandas numpy scipy openpyxl
  pip install krippendorff        # optional; a built-in implementation is used otherwise
"""

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

EVAL_DIR   = "./eval_results"
DIMENSIONS = ["naturalness", "code_switching", "fluency", "relevance"]


# ─── Fleiss' kappa ────────────────────────────────────────────────────────────

def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' kappa from a (n_subjects × n_categories) count matrix.
    Each cell[i][j] = number of raters who assigned category j to subject i.
    All rows must sum to the same n_raters value.
    """
    n_subjects, n_cats = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())

    if n_raters < 2:
        return float("nan")

    # P_i = proportion of agreeing rater pairs per subject
    P_i    = np.sum(ratings_matrix * (ratings_matrix - 1), axis=1) / (n_raters * (n_raters - 1))
    P_bar  = P_i.mean()

    # p_j = proportion of all ratings in each category
    p_j    = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    P_e    = np.sum(p_j ** 2)

    if abs(1.0 - P_e) < 1e-10:
        return 1.0

    return round(float((P_bar - P_e) / (1.0 - P_e)), 4)


def kappa_interpretation(k: float) -> str:
    if np.isnan(k):    return "N/A (insufficient data)"
    if k < 0:          return "Poor (less than chance)"
    elif k < 0.20:     return "Slight"
    elif k < 0.40:     return "Fair"
    elif k < 0.60:     return "Moderate"
    elif k < 0.80:     return "Substantial"
    else:              return "Almost perfect"


# ─── Krippendorff's alpha (ordinal) ──────────────────────────────────────────

def _krippendorff_alpha_ordinal(data: np.ndarray) -> float:
    """
    Self-contained ordinal Krippendorff's alpha (Krippendorff 2011, "Computing
    Krippendorff's Alpha-Reliability"). data: (n_raters × n_units), NaN = missing.
    """
    values = np.unique(data[~np.isnan(data)])
    index  = {v: i for i, v in enumerate(values)}
    k      = len(values)

    # Coincidence matrix over pairable units (≥ 2 ratings)
    o = np.zeros((k, k))
    for unit in data.T:
        unit = unit[~np.isnan(unit)]
        m    = len(unit)
        if m < 2:
            continue
        counts = np.zeros(k)
        for v in unit:
            counts[index[v]] += 1
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)

    n_c = o.sum(axis=1)
    n   = n_c.sum()
    if n <= 1:
        return float("nan")

    # Ordinal metric: delta_ck = (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2
    cum   = np.concatenate([[0.0], np.cumsum(n_c)])
    delta = np.zeros((k, k))
    for c in range(k):
        for j in range(k):
            lo, hi      = min(c, j), max(c, j)
            delta[c, j] = (cum[hi + 1] - cum[lo] - (n_c[c] + n_c[j]) / 2) ** 2

    expected = np.sum(np.outer(n_c, n_c) * delta)
    if expected == 0:
        return float("nan")
    return float(1.0 - (n - 1) * np.sum(o * delta) / expected)


def krippendorff_alpha_ordinal(reliability_data) -> float:
    """
    Ordinal Krippendorff's alpha for a (n_raters × n_units) matrix with NaN for
    missing ratings. Uses the `krippendorff` package when installed, otherwise
    the built-in implementation. NaN if fewer than two distinct values or no
    unit has two ratings.
    """
    data = np.asarray(reliability_data, dtype=float)
    pairable = np.sum(~np.isnan(data), axis=0) >= 2
    if not pairable.any() or len(np.unique(data[:, pairable][~np.isnan(data[:, pairable])])) < 2:
        return float("nan")
    try:
        import krippendorff
    except ImportError:
        return round(_krippendorff_alpha_ordinal(data), 4)
    return round(float(krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")), 4)


def alpha_interpretation(a: float) -> str:
    # Krippendorff (2004): alpha >= .800 reliable, .667–.800 tentative conclusions only
    if np.isnan(a):    return "N/A (insufficient data)"
    if a >= 0.800:     return "Reliable (>= .800)"
    if a >= 0.667:     return "Tentative (.667–.800)"
    return "Unreliable (< .667)"


# ─── Wilcoxon signed-rank tests with Holm correction ─────────────────────────

def holm_correction(p_values: list) -> list:
    """Holm–Bonferroni adjusted p-values (NaNs are ignored and stay NaN)."""
    p     = np.asarray(p_values, dtype=float)
    valid = np.flatnonzero(~np.isnan(p))
    adj   = np.full(len(p), np.nan)
    m     = len(valid)
    running_max = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running_max = max(running_max, (m - rank) * p[i])
        adj[i]      = min(1.0, running_max)
    return adj.tolist()


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-sided Wilcoxon signed-rank test for every pair of model variants in
    every dimension. The unit is the question: scores are first averaged over
    annotators per (question, variant), then paired across questions rated for
    both variants. Holm correction is applied over the whole family of tests
    (all variant pairs × all dimensions).
    """
    from scipy.stats import wilcoxon

    item_means = (df.groupby(["dimension", "question_id", "model_variant"])["score"]
                    .mean().reset_index())
    variants   = sorted(df["model_variant"].unique())
    rows = []
    for dim in DIMENSIONS:
        wide = item_means[item_means["dimension"] == dim].pivot(
            index="question_id", columns="model_variant", values="score")
        for a, b in itertools.combinations(variants, 2):
            if a not in wide.columns or b not in wide.columns:
                continue
            pair = wide[[a, b]].dropna()
            if pair.empty:
                continue
            diff = pair[a] - pair[b]
            if (diff == 0).all():
                stat, p = 0.0, 1.0
            else:
                try:
                    res = wilcoxon(pair[a], pair[b], alternative="two-sided")
                    stat, p = float(res.statistic), float(res.pvalue)
                except ValueError:
                    stat, p = float("nan"), float("nan")
            rows.append({
                "dimension":   dim,
                "variant_a":   a,
                "variant_b":   b,
                "n_questions": len(pair),
                "n_nonzero":   int((diff != 0).sum()),   # wilcoxon drops zero differences
                "mean_a":      round(pair[a].mean(), 3),
                "mean_b":      round(pair[b].mean(), 3),
                "median_diff": round(diff.median(), 3),
                "W":           stat,
                "p_value":     p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"]         = holm_correction(out["p_value"].tolist())
        out["significant_05"] = out["p_holm"] < 0.05
    return out


# ─── Load all annotator files ─────────────────────────────────────────────────

def load_all_ratings() -> dict:
    """Returns {annotator_name: {str_qid: rating_record}}."""
    files = sorted(glob.glob(os.path.join(EVAL_DIR, "human_eval_*.json")))
    if not files:
        raise FileNotFoundError(
            f"No human_eval_*.json files found in {EVAL_DIR}/\n"
            "Make sure annotators have completed and saved their ratings."
        )
    all_data = {}
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotator = data["annotator"]
        all_data[annotator] = data.get("ratings", {})
        n = len(all_data[annotator])
        print(f"  Loaded: {annotator:<20}  ({n} questions rated)")
    return all_data


# ─── Flatten to a DataFrame ───────────────────────────────────────────────────

def build_flat_df(all_data: dict) -> pd.DataFrame:
    rows = []
    for annotator, ratings in all_data.items():
        for qid_str, record in ratings.items():
            model_order = record.get("model_order", {})   # {label: variant}
            for dim in DIMENSIONS:
                dim_scores = record.get("ratings", {}).get(dim, {})
                for label, score in dim_scores.items():
                    variant = model_order.get(label, label)
                    rows.append({
                        "annotator":     annotator,
                        "question_id":   int(qid_str),
                        "model_variant": variant,
                        "label":         label,
                        "dimension":     dim,
                        "score":         int(score),
                    })
    return pd.DataFrame(rows)


# ─── Compute Fleiss' kappa ────────────────────────────────────────────────────

def compute_kappa(df: pd.DataFrame, n_annotators: int) -> dict:
    """
    Returns {dimension: kappa_value}.
    Only uses (question, model_variant) pairs rated by ALL annotators.
    """
    results = {}
    CATEGORIES = [1, 2, 3, 4, 5]

    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        grouped = sub.groupby(["question_id", "model_variant"])["score"].apply(list)
        complete = grouped[grouped.apply(len) == n_annotators]

        if len(complete) == 0:
            results[dim] = float("nan")
            continue

        matrix = np.array([
            [scores.count(c) for c in CATEGORIES]
            for scores in complete
        ])
        results[dim] = fleiss_kappa(matrix)

    return results


def compute_alpha(df: pd.DataFrame) -> dict:
    """
    Returns {dimension: (alpha, n_units)} — ordinal Krippendorff's alpha over
    (question, model_variant) units; incomplete units are allowed.
    """
    results = {}
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            results[dim] = (float("nan"), 0)
            continue
        matrix = sub.pivot_table(index="annotator", columns=["question_id", "model_variant"],
                                 values="score", aggfunc="first")
        n_units = int((matrix.notna().sum(axis=0) >= 2).sum())
        results[dim] = (krippendorff_alpha_ordinal(matrix.to_numpy(dtype=float)), n_units)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n[Human Evaluation Analysis]")
    print(f"Reading from: {os.path.abspath(EVAL_DIR)}/\n")

    all_data    = load_all_ratings()
    n_annotators = len(all_data)
    annotators  = list(all_data.keys())
    print(f"\nAnnotators ({n_annotators}): {', '.join(annotators)}")

    df = build_flat_df(all_data)
    if df.empty:
        print("No ratings found in the files.")
        return

    # ── 1. Average scores per model variant × dimension ───────────────────────
    print("\n" + "─" * 65)
    print("  Average Scores per Model (1–5 scale)")
    print("─" * 65)
    agg   = df.groupby(["model_variant", "dimension"])["score"].mean().unstack()
    agg["Overall"] = agg[DIMENSIONS].mean(axis=1)
    print(agg.round(3).to_string())

    # ── 2. Inter-annotator agreement per dimension ────────────────────────────
    print("\n" + "─" * 65)
    print(f"  Krippendorff's α (ordinal)  (inter-annotator agreement, n={n_annotators} raters)")
    print("─" * 65)
    alpha_vals = compute_alpha(df)
    for dim in DIMENSIONS:
        a, n_units = alpha_vals[dim]
        a_str = f"{a:.4f}" if not np.isnan(a) else "  N/A "
        print(f"  {dim:<20}  α = {a_str}   ({alpha_interpretation(a)}, {n_units} units)")

    print("\n" + "─" * 65)
    print(f"  Fleiss' κ  (nominal; complete units only — reported for comparison)")
    print("─" * 65)
    kappa_vals = compute_kappa(df, n_annotators)
    for dim in DIMENSIONS:
        k = kappa_vals[dim]
        interp = kappa_interpretation(k)
        k_str = f"{k:.4f}" if not np.isnan(k) else "  N/A "
        print(f"  {dim:<20}  κ = {k_str}   ({interp})")

    # ── 2b. Wilcoxon signed-rank tests between variants ───────────────────────
    print("\n" + "─" * 65)
    print("  Wilcoxon signed-rank (two-sided; unit = question, scores averaged over")
    print("  annotators; Holm correction over all variant pairs × dimensions)")
    print("─" * 65)
    try:
        df_wilcoxon = wilcoxon_tests(df)
    except ImportError:
        print("  [WARN] scipy not installed — Wilcoxon tests skipped.")
        df_wilcoxon = pd.DataFrame()
    if df_wilcoxon.empty:
        print("  No variant pairs with shared questions.")
    else:
        print(f"  Family size (Holm): {len(df_wilcoxon)} tests")
        for _, r in df_wilcoxon.iterrows():
            print(f"  {r['dimension']:<15} {r['variant_a']} vs {r['variant_b']}  "
                  f"n={r['n_questions']} (non-zero {r['n_nonzero']})  Δmedian={r['median_diff']:+.2f}  "
                  f"p={r['p_value']:.4f}  p_holm={r['p_holm']:.4f}")

    # ── 3. Per-annotator averages (check for systematic bias) ─────────────────
    print("\n" + "─" * 65)
    print("  Per-Annotator Average Score (all models & dimensions)")
    print("─" * 65)
    per_ann = df.groupby("annotator")["score"].mean()
    for ann, avg in per_ann.items():
        print(f"  {ann:<25}  {avg:.3f}")

    # ── 4. Save outputs ────────────────────────────────────────────────────────
    xlsx_path     = os.path.join(EVAL_DIR, "human_eval_summary.xlsx")
    kappa_path    = os.path.join(EVAL_DIR, "human_eval_kappa.json")
    wilcoxon_path = os.path.join(EVAL_DIR, "human_eval_wilcoxon.csv")

    # Save agreement JSON
    kappa_out = {
        dim: {
            "kappa":                      kappa_vals[dim] if not np.isnan(kappa_vals[dim]) else None,
            "interpretation":             kappa_interpretation(kappa_vals[dim]),
            "krippendorff_alpha_ordinal": alpha_vals[dim][0] if not np.isnan(alpha_vals[dim][0]) else None,
            "alpha_interpretation":       alpha_interpretation(alpha_vals[dim][0]),
            "alpha_n_units":              alpha_vals[dim][1],
        }
        for dim in DIMENSIONS
    }
    with open(kappa_path, "w") as f:
        json.dump(kappa_out, f, indent=2)
    print(f"\n  Saved: {kappa_path}")

    if not df_wilcoxon.empty:
        df_wilcoxon.to_csv(wilcoxon_path, index=False)
        print(f"  Saved: {wilcoxon_path}")

    try:
        # Per-annotator per-model table (useful for the paper appendix)
        per_ann_model = (
            df.groupby(["annotator", "model_variant", "dimension"])["score"]
            .mean()
            .unstack(level="dimension")
            .round(3)
        )

        # Agreement table for Excel
        kappa_rows = [
            {"dimension": dim,
             "krippendorff_alpha_ordinal": alpha_vals[dim][0],
             "alpha_interpretation":       alpha_interpretation(alpha_vals[dim][0]),
             "alpha_n_units":              alpha_vals[dim][1],
             "fleiss_kappa":               kappa_vals[dim],
             "interpretation":             kappa_interpretation(kappa_vals[dim])}
            for dim in DIMENSIONS
        ]
        df_kappa = pd.DataFrame(kappa_rows)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(          writer, sheet_name="Raw Ratings",          index=False)
            agg.round(3).to_excel(writer, sheet_name="Avg Scores by Model")
            per_ann_model.to_excel(writer, sheet_name="Per Annotator Scores")
            df_kappa.to_excel(    writer, sheet_name="Agreement",            index=False)
            if not df_wilcoxon.empty:
                df_wilcoxon.to_excel(writer, sheet_name="Wilcoxon (Holm)",   index=False)
        print(f"  Saved: {xlsx_path}")

    except ImportError:
        csv_path = os.path.join(EVAL_DIR, "human_eval_raw.csv")
        df.to_csv(csv_path, index=False)
        print(f"  openpyxl not found — saved raw data as {csv_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
