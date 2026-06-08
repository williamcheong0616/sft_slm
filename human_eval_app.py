#!/usr/bin/env python3
"""
Human evaluation interface for Bahasa Rojak model responses.

Reads benchmark_raw.json produced by eval_metrics.py and presents each
question with up to 3 model responses labelled Model A / B / C (blind —
the label-to-model mapping is hidden during rating).

Run:
  pip install streamlit
  streamlit run human_eval_app.py

Each annotator:
  1. Enters their name on the login page.
  2. Rates each response 1–5 on four dimensions.
  3. Progress is auto-saved after every question.

Output: eval_results/human_eval_<name>.json
        (run human_eval_analyze.py after all annotators are done)
"""

import hashlib
import json
import os
import random
from datetime import datetime

import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

EVAL_DIR       = "./eval_results"
BENCHMARK_FILE = os.path.join(EVAL_DIR, "benchmark_raw.json")
N_EVAL_ITEMS   = 50   # how many of the 75 samples each annotator rates

DIMENSIONS = {
    "naturalness":    "Naturalness",
    "code_switching": "Code-Switching",
    "fluency":        "Fluency",
    "relevance":      "Relevance",
}

DIM_HELP = {
    "naturalness":    "Sounds like authentic Bahasa Rojak a Malaysian would naturally say",
    "code_switching": "Mixing of Malay, English, and particles (lah, mah, kan, wey) feels natural",
    "fluency":        "Response is grammatically coherent, easy to follow, and not repetitive",
    "relevance":      "Response actually addresses what the question asked",
}

SCALE_LABELS = {1: "1 — Very Poor", 2: "2 — Poor", 3: "3 — Average", 4: "4 — Good", 5: "5 — Excellent"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_model_order(question_id: int, annotator_name: str, variants: list) -> list:
    """Deterministic shuffle per (question_id, annotator) pair — blind evaluation."""
    seed_str = f"{question_id}_{annotator_name}"
    seed     = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 31)
    rng      = random.Random(seed)
    order    = variants.copy()
    rng.shuffle(order)
    return order


def save_progress(annotator_name: str, ratings: dict, started_at: str) -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in annotator_name)
    path = os.path.join(EVAL_DIR, f"human_eval_{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "annotator":  annotator_name,
            "started":    started_at,
            "last_saved": datetime.now().isoformat(),
            "ratings":    ratings,
        }, f, ensure_ascii=False, indent=2)
    return path


def load_progress(annotator_name: str) -> dict:
    safe_name = "".join(c if c.isalnum() else "_" for c in annotator_name)
    path = os.path.join(EVAL_DIR, f"human_eval_{safe_name}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_benchmark() -> dict:
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Page: Login ──────────────────────────────────────────────────────────────

def page_login():
    st.title("Bahasa Rojak — Model Evaluation")

    st.markdown("""
### Selamat datang! / Welcome!

Thank you for helping evaluate AI-generated Bahasa Rojak responses.

**What you will do:**
- Read a question written in Bahasa Rojak (Malaysian code-mixed language).
- Read **three responses** labelled **Model A, B, C** — you will NOT know which AI produced which.
- Rate each response on **4 dimensions** using a 1–5 scale.

**Rating scale:**  `1 = Very Poor`  `2 = Poor`  `3 = Average`  `4 = Good`  `5 = Excellent`

**Dimensions explained:**

| Dimension | What to look for |
|---|---|
| **Naturalness** | Does it sound like real Bahasa Rojak a Malaysian would naturally say? |
| **Code-Switching** | Is the Malay/English/particle mixing (lah, mah, kan, wey) natural and fluent? |
| **Fluency** | Is it easy to read, coherent, and not repetitive or garbled? |
| **Relevance** | Does it actually answer what was asked? |

**Estimated time:** ~30–40 minutes for all questions.
Your progress is **auto-saved** after every question. You can close and resume anytime.
    """)

    st.divider()

    if not os.path.exists(BENCHMARK_FILE):
        st.error(
            f"Benchmark file not found: `{BENCHMARK_FILE}`\n\n"
            "Please run `eval_metrics.py` first to generate model responses."
        )
        return

    name = st.text_input(
        "Your name (used to save your progress):",
        placeholder="e.g., Ahmad Faiz",
        help="Use your real name so your ratings can be matched with other annotators.",
    )

    benchmark = load_benchmark()
    all_variants = benchmark["metadata"]["model_variants"]
    ft_variants  = [v for v in all_variants if "finetuned"  in v]
    zs_variants  = [v for v in all_variants if "zeroshot"   in v]

    st.markdown("**Which models to compare?**")
    use_ft = st.checkbox("Fine-tuned models (recommended)", value=True)
    use_zs = st.checkbox("Zero-shot baselines", value=False)

    selected = []
    if use_ft: selected += ft_variants
    if use_zs: selected += zs_variants
    selected = selected[:3]   # cap at 3 so the UI stays readable

    if not selected:
        st.warning("Please select at least one model type.")

    col1, col2 = st.columns([1, 3])
    with col1:
        start = st.button(
            "Start Evaluation",
            disabled=not name.strip() or not selected,
            type="primary",
        )

    if start:
        existing = load_progress(name.strip())
        if existing and existing.get("ratings"):
            n_done = len(existing["ratings"])
            st.info(f"Resuming previous session — {n_done} question(s) already rated.")
            st.session_state.ratings    = existing["ratings"]
            st.session_state.started_at = existing.get("started", datetime.now().isoformat())
        else:
            st.session_state.ratings    = {}
            st.session_state.started_at = datetime.now().isoformat()

        samples  = benchmark["samples"][:N_EVAL_ITEMS]
        rated_ids = set(st.session_state.ratings.keys())
        first_unrated = next(
            (i for i, q in enumerate(samples) if str(q["id"]) not in rated_ids),
            len(samples) - 1,
        )

        st.session_state.annotator_name  = name.strip()
        st.session_state.benchmark       = benchmark
        st.session_state.questions       = samples
        st.session_state.model_variants  = selected
        st.session_state.current_q       = first_unrated
        st.session_state.page            = "eval"
        st.rerun()


# ─── Page: Evaluation ─────────────────────────────────────────────────────────

def page_eval():
    questions       = st.session_state.questions
    model_variants  = st.session_state.model_variants
    annotator_name  = st.session_state.annotator_name
    ratings         = st.session_state.ratings
    current_q       = st.session_state.current_q
    n_total         = len(questions)
    n_rated         = len(ratings)

    # ── Sidebar: navigation ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"**Annotator:** {annotator_name}")
        st.progress(n_rated / n_total, text=f"{n_rated} / {n_total} rated")
        st.divider()
        st.markdown("**Jump to question:**")
        for i, q in enumerate(questions):
            is_rated   = str(q["id"]) in ratings
            is_current = (i == current_q)
            icon = "✅" if is_rated else ("▶" if is_current else "○")
            label = f"{icon} Q{i+1}"
            if is_current:
                label = f"**{label}**"
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state.current_q = i
                st.rerun()
        st.divider()
        if st.button("Save & Exit", use_container_width=True):
            path = save_progress(annotator_name, ratings, st.session_state.started_at)
            st.success(f"Saved to:\n`{path}`")

    # ── Completion screen ──────────────────────────────────────────────────────
    if current_q >= n_total:
        st.balloons()
        st.title("All done! Thank you!")
        path = save_progress(annotator_name, ratings, st.session_state.started_at)
        st.success(f"Your {n_rated} ratings have been saved to:\n`{path}`")
        st.markdown("Please let William know you have finished so he can collect your results.")
        if st.button("Review from Q1"):
            st.session_state.current_q = 0
            st.rerun()
        return

    question = questions[current_q]
    qid      = question["id"]

    # Deterministic blind shuffle for this (question, annotator) pair
    model_order = get_model_order(qid, annotator_name, model_variants)
    labels      = ["Model A", "Model B", "Model C"][:len(model_order)]

    # Load any previously saved ratings for this question
    saved = ratings.get(str(qid), {})
    saved_dim_ratings = saved.get("ratings", {})   # {dim: {label: score}}

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"### Question {current_q + 1} of {n_total}")

    # Question box
    st.info(f"**{question['question']}**")

    # Optional reference (hidden by default to avoid bias)
    with st.expander("Show reference answer (toggle carefully — may influence ratings)"):
        st.markdown(f"*{question.get('reference', 'N/A')}*")

    st.divider()

    # ── Model responses ────────────────────────────────────────────────────────
    st.markdown("**Read all three responses before rating:**")
    resp_cols = st.columns(len(model_order))
    for col, label, variant in zip(resp_cols, labels, model_order):
        with col:
            st.markdown(f"**{label}**")
            response = question.get(f"{variant}_response", "*(no response generated)*")
            st.text_area(
                label, value=response, height=220,
                disabled=True, label_visibility="collapsed",
            )

    st.divider()
    st.markdown("**Rate each response — 1 = Very Poor, 5 = Excellent:**")

    # ── Rating grid ────────────────────────────────────────────────────────────
    collected = {}   # {dim: {label: score}}

    for dim_key, dim_name in DIMENSIONS.items():
        st.markdown(f"**{dim_name}** — *{DIM_HELP[dim_key]}*")
        dim_cols  = st.columns(len(model_order))
        dim_vals  = {}
        prev_dim  = saved_dim_ratings.get(dim_key, {})

        for col, label, variant in zip(dim_cols, labels, model_order):
            with col:
                st.markdown(f"<center><small>{label}</small></center>", unsafe_allow_html=True)
                default = prev_dim.get(label, 3)
                val = st.select_slider(
                    f"{dim_key}_{label}",
                    options=[1, 2, 3, 4, 5],
                    value=default,
                    format_func=lambda x: SCALE_LABELS[x],
                    key=f"q{qid}_{dim_key}_{label}",
                    label_visibility="collapsed",
                )
                st.markdown(
                    f"<center><b>{'⭐' * val}</b></center>",
                    unsafe_allow_html=True,
                )
                dim_vals[label] = val

        collected[dim_key] = dim_vals
        st.markdown("")

    st.divider()

    # ── Navigation buttons ─────────────────────────────────────────────────────
    col_prev, col_spacer, col_next = st.columns([1, 2, 1])

    with col_prev:
        if current_q > 0:
            if st.button("◄ Previous", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

    with col_next:
        label_next = "Save & Finish ✓" if current_q == n_total - 1 else "Save & Next ►"
        if st.button(label_next, type="primary", use_container_width=True):
            ratings[str(qid)] = {
                "question_id": qid,
                "model_order": {lbl: var for lbl, var in zip(labels, model_order)},
                "ratings":     collected,
            }
            st.session_state.ratings = ratings
            save_progress(annotator_name, ratings, st.session_state.started_at)
            st.session_state.current_q = current_q + 1
            st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Bahasa Rojak Evaluation",
        page_icon="🇲🇾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "page" not in st.session_state:
        st.session_state.page = "login"

    for key in ("ratings", "annotator_name", "benchmark", "questions",
                "model_variants", "current_q", "started_at"):
        if key not in st.session_state:
            st.session_state[key] = {} if key == "ratings" else (
                0 if key == "current_q" else (
                    datetime.now().isoformat() if key == "started_at" else None
                )
            )

    if st.session_state.page == "login":
        page_login()
    elif st.session_state.page == "eval":
        page_eval()


if __name__ == "__main__":
    main()
