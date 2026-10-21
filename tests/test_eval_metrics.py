"""CPU-only smoke tests for eval_metrics.py (no model loading)."""

import json

import numpy as np
import pytest

import eval_metrics as em


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _chat(q, a, key="messages", system=None):
    turns = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if system:
        turns.insert(0, {"role": "system", "content": system})
    return {key: turns}


# ─── messages / conversations filter ─────────────────────────────────────────

def test_extract_qa_accepts_messages_and_conversations():
    assert em.extract_qa(_chat("q1", "a1", key="messages")) == {"question": "q1", "reference": "a1"}
    assert em.extract_qa(_chat("q2", "a2", key="conversations")) == {"question": "q2", "reference": "a2"}


def test_extract_qa_keeps_leading_system_and_rejects_unusable():
    assert em.extract_qa(_chat("q", "a", system="sys")) == {"question": "q", "reference": "a", "system": "sys"}
    assert em.extract_qa({"messages": [{"role": "assistant", "content": "a"},
                                       {"role": "user", "content": "q"}]}) is None
    assert em.extract_qa(_chat("q", "   ")) is None
    assert em.extract_qa({"messages": [{"role": "user", "content": "q"}]}) is None
    assert em.extract_qa({"text": "no turns"}) is None


def test_sample_test_set_reads_prepare_data_messages_format(tmp_path):
    test_file = tmp_path / "eval.jsonl"
    _write_jsonl(test_file, [_chat(f"q{i}", f"a{i}") for i in range(10)])
    samples = em.sample_test_set(str(test_file), 5, 42, str(tmp_path))
    assert len(samples) == 5
    assert [s["id"] for s in samples] == list(range(5))
    assert all(s["question"].startswith("q") and s["reference"].startswith("a") for s in samples)
    cached = json.loads((tmp_path / "test_samples.json").read_text())
    assert cached["source"]["sha256"] == em.file_sha256(str(test_file))


def test_sample_test_set_raises_when_too_few_valid_records(tmp_path):
    test_file = tmp_path / "eval.jsonl"
    _write_jsonl(test_file, [_chat(f"q{i}", f"a{i}") for i in range(3)] + [{"text": "bad"}])
    with pytest.raises(em.EvalDataError, match="Only 3 usable"):
        em.sample_test_set(str(test_file), 5, 42, str(tmp_path))


def test_sample_test_set_refuses_cache_from_other_source(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    file_a, file_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(file_a, [_chat(f"qa{i}", f"aa{i}") for i in range(6)])
    _write_jsonl(file_b, [_chat(f"qb{i}", f"ab{i}") for i in range(6)])

    first = em.sample_test_set(str(file_a), 4, 42, str(out))
    assert em.sample_test_set(str(file_a), 4, 42, str(out)) == first          # same source: reused
    with pytest.raises(em.EvalDataError, match="different source"):
        em.sample_test_set(str(file_b), 4, 42, str(out))
    with pytest.raises(em.EvalDataError, match="different source"):
        em.sample_test_set(str(file_a), 3, 42, str(out))
    rebuilt = em.sample_test_set(str(file_b), 4, 42, str(out), resample=True)
    assert all(s["question"].startswith("qb") for s in rebuilt)


def test_sample_test_set_refuses_legacy_list_cache(tmp_path):
    test_file = tmp_path / "eval.jsonl"
    _write_jsonl(test_file, [_chat(f"q{i}", f"a{i}") for i in range(6)])
    (tmp_path / "test_samples.json").write_text(json.dumps([{"id": 0, "question": "q", "reference": "r"}]))
    with pytest.raises(em.EvalDataError, match="no source record"):
        em.sample_test_set(str(test_file), 4, 42, str(tmp_path))


def test_check_overlap_exact_and_normalised(tmp_path):
    train_file = tmp_path / "train.jsonl"
    _write_jsonl(train_file, [_chat("Apa khabar?", "Baik lah"), _chat("Hello   World", "Other answer")])
    samples = [
        {"id": 0, "question": "Apa khabar?", "reference": "new"},       # exact question
        {"id": 1, "question": "hello world", "reference": "BAIK  LAH"},  # normalised q + ref
        {"id": 2, "question": "unseen", "reference": "unseen"},
    ]
    report = em.check_overlap(samples, str(train_file))
    assert report["checked"]
    assert report["question_exact"] == [0]
    assert report["question_normalised"] == [0, 1]
    assert report["reference_exact"] == []
    assert report["reference_normalised"] == [1]
    assert em.check_overlap(samples, str(tmp_path / "missing.jsonl"))["checked"] is False


def test_stale_reason_detects_responses_for_other_questions(tmp_path):
    samples = [{"id": 0, "question": "q0"}, {"id": 1, "question": "q1"}]
    path = tmp_path / "responses.json"
    path.write_text(json.dumps([{"id": 0, "question": "q0", "response": "r"},
                                {"id": 1, "question": "q1", "response": "r"}]))
    assert em.stale_reason(str(path), samples) is None
    path.write_text(json.dumps({"metadata": {}, "responses": [{"id": 0, "question": "OLD", "response": "r"}]}))
    assert em.stale_reason(str(path), samples) == "1 question(s) differ, 1 missing"


def test_parse_model_overrides():
    assert em.parse_model_overrides(["qwen=Org/Model-8B"]) == {"qwen": "Org/Model-8B"}
    for bad in (["qwen"], ["mistral=x/y"], ["qwen="]):
        with pytest.raises(ValueError):
            em.parse_model_overrides(bad)


# ─── CMI ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("saya suka the food lah", 33.33),   # M: saya, lah | E: the | u: suka, food → 100*(1-2/3)
    ("saya the", 50.0),                  # 1 M, 1 E
    ("the cat is on the mat", 0.0),      # E only (u: cat, mat)
    ("nasi goreng sedap", 0.0),          # no lexicon hits → n == u → 0
    ("", 0.0),
    ("Saya THE kita", 33.33),            # case-insensitive: 2 M, 1 E
])
def test_cmi_standard_hand_made(text, expected):
    assert em.compute_cmi_standard(text) == pytest.approx(expected, abs=0.01)


def test_cmi_heuristic_unchanged():
    # Old formula divides by all tokens and returns 50 when nothing matches.
    assert em.compute_cmi_heuristic("saya suka the food lah") == 60.0
    assert em.compute_cmi_heuristic("nasi goreng sedap") == 50.0


# ─── <think> stripping ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, clean, kind", [
    ("Jawapan biasa lah.", "Jawapan biasa lah.", None),
    ("<think>let me reason</think>\n\nOk lah boss.", "Ok lah boss.", "block"),
    ("reasoning from a template-opened tag</think>Answer here", "Answer here", "orphan"),
    ("Partial answer <think>ran out of tokens", "Partial answer", "unclosed"),
    ("<think>\n</think>\n\nEmpty think block", "Empty think block", "block"),
])
def test_strip_think(raw, clean, kind):
    assert em.strip_think(raw) == (clean, kind)


# ─── bootstrap ───────────────────────────────────────────────────────────────

def test_bootstrap_ci_constant_and_deterministic():
    const = np.full(30, 7.0)
    assert em.bootstrap_ci(lambda idx: const[idx].mean(), 30) == (7.0, 7.0)

    values = np.random.default_rng(0).normal(50, 10, size=75)
    ci_1 = em.bootstrap_ci(lambda idx: values[idx].mean(), 75, n_resamples=1000, seed=1)
    ci_2 = em.bootstrap_ci(lambda idx: values[idx].mean(), 75, n_resamples=1000, seed=1)
    assert ci_1 == ci_2
    lo, hi = ci_1
    assert lo < values.mean() < hi
    # roughly mean ± 1.96 * SE
    se = values.std(ddof=1) / np.sqrt(75)
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_bootstrap_indices_shape_and_range():
    idx = em.bootstrap_indices(10, n_resamples=200, seed=3)
    assert idx.shape == (200, 10)
    assert idx.min() >= 0 and idx.max() <= 9


def test_paired_bootstrap_pvalue():
    values = np.random.default_rng(0).normal(0, 1, size=100)
    idx    = em.bootstrap_indices(100, n_resamples=1000, seed=0)
    dist_a = em.bootstrap_distribution(lambda i: values[i].mean(), idx)
    dist_b = em.bootstrap_distribution(lambda i: (values[i] + 5).mean(), idx)
    # identical systems → p = 1; constant shift of 5 → smallest possible p
    assert em.paired_bootstrap_pvalue(dist_a, dist_a, 0.0) == 1.0
    assert em.paired_bootstrap_pvalue(dist_b, dist_a, 5.0) == pytest.approx(1 / 1001)


def test_bootstrap_bleu_matches_full_recomputation():
    sacrebleu = pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import BLEU, CHRF
    hyps = ["saya suka makan nasi lemak lah", "the weather today very hot one", "ok boss",
            "jom pergi mamak tonight", "I think can lah"]
    refs = ["saya suka nasi lemak lah", "the weather today is very hot", "ok lah boss",
            "jom lepak mamak tonight", "I think boleh lah"]
    for cls, fn in ((BLEU, sacrebleu.corpus_bleu), (CHRF, sacrebleu.corpus_chrf)):
        stat = em._corpus_stat_fn(cls, hyps, refs)
        assert stat(np.arange(5)) == pytest.approx(fn(hyps, [refs]).score)
        for idx in em.bootstrap_indices(5, n_resamples=20, seed=0):
            assert stat(idx) == pytest.approx(
                fn([hyps[i] for i in idx], [[refs[i] for i in idx]]).score)
    lo, hi = em.bootstrap_ci(em._corpus_stat_fn(BLEU, refs, refs), 5, n_resamples=50)
    assert lo == hi == pytest.approx(100.0)
