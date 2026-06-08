#!/usr/bin/env python3
"""
Analyze human evaluation results.

Computes:
  - Average scores per model variant and dimension
  - Fleiss' kappa (inter-annotator agreement) per dimension
  - Pairwise model comparisons

Run AFTER all annotators have saved their ratings:
  python human_eval_analyze.py

Reads:  eval_results/human_eval_*.json
Writes: eval_results/human_eval_summary.xlsx
        eval_results/human_eval_kappa.json

Requirements:
  pip install pandas numpy openpyxl
"""

import glob
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

    # ── 2. Fleiss' kappa per dimension ────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  Fleiss' κ  (inter-annotator agreement, n={n_annotators} raters)")
    print("─" * 65)
    kappa_vals = compute_kappa(df, n_annotators)
    for dim in DIMENSIONS:
        k = kappa_vals[dim]
        interp = kappa_interpretation(k)
        k_str = f"{k:.4f}" if not np.isnan(k) else "  N/A "
        print(f"  {dim:<20}  κ = {k_str}   ({interp})")

    # ── 3. Per-annotator averages (check for systematic bias) ─────────────────
    print("\n" + "─" * 65)
    print("  Per-Annotator Average Score (all models & dimensions)")
    print("─" * 65)
    per_ann = df.groupby("annotator")["score"].mean()
    for ann, avg in per_ann.items():
        print(f"  {ann:<25}  {avg:.3f}")

    # ── 4. Save outputs ────────────────────────────────────────────────────────
    xlsx_path  = os.path.join(EVAL_DIR, "human_eval_summary.xlsx")
    kappa_path = os.path.join(EVAL_DIR, "human_eval_kappa.json")

    # Save kappa JSON
    kappa_out = {
        dim: {
            "kappa":          kappa_vals[dim] if not np.isnan(kappa_vals[dim]) else None,
            "interpretation": kappa_interpretation(kappa_vals[dim]),
        }
        for dim in DIMENSIONS
    }
    with open(kappa_path, "w") as f:
        json.dump(kappa_out, f, indent=2)
    print(f"\n  Saved: {kappa_path}")

    try:
        # Per-annotator per-model table (useful for the paper appendix)
        per_ann_model = (
            df.groupby(["annotator", "model_variant", "dimension"])["score"]
            .mean()
            .unstack(level="dimension")
            .round(3)
        )

        # Kappa table for Excel
        kappa_rows = [
            {"dimension": dim, "fleiss_kappa": kappa_vals[dim], "interpretation": kappa_interpretation(kappa_vals[dim])}
            for dim in DIMENSIONS
        ]
        df_kappa = pd.DataFrame(kappa_rows)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(          writer, sheet_name="Raw Ratings",          index=False)
            agg.round(3).to_excel(writer, sheet_name="Avg Scores by Model")
            per_ann_model.to_excel(writer, sheet_name="Per Annotator Scores")
            df_kappa.to_excel(    writer, sheet_name="Fleiss Kappa",         index=False)
        print(f"  Saved: {xlsx_path}")

    except ImportError:
        csv_path = os.path.join(EVAL_DIR, "human_eval_raw.csv")
        df.to_csv(csv_path, index=False)
        print(f"  openpyxl not found — saved raw data as {csv_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
