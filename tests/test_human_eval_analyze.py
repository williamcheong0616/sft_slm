"""CPU-only smoke tests for human_eval_analyze.py."""

import numpy as np
import pandas as pd
import pytest

import human_eval_analyze as hea

nan = np.nan

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers × 12 units.
# Published values: nominal .743, ordinal .815, interval .849.
KRIPPENDORFF_2011 = np.array([
    [1,   2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
    [1,   2, 3, 3, 2, 2, 4, 1, 2, 5,   nan, 3],
    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5,   1,   nan],
    [1,   2, 3, 3, 2, 4, 4, 1, 2, 5,   1,   nan],
])


def test_alpha_builtin_matches_published_example():
    assert hea._krippendorff_alpha_ordinal(KRIPPENDORFF_2011) == pytest.approx(0.815, abs=5e-4)


def test_alpha_perfect_agreement_and_degenerate_cases():
    perfect = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, nan]], dtype=float)
    assert hea._krippendorff_alpha_ordinal(perfect) == pytest.approx(1.0)
    assert hea.krippendorff_alpha_ordinal(perfect) == pytest.approx(1.0)
    assert np.isnan(hea.krippendorff_alpha_ordinal([[3, 3, 3], [3, 3, 3]]))    # one value only
    assert np.isnan(hea.krippendorff_alpha_ordinal([[1, nan], [nan, 2]]))      # nothing pairable


def test_alpha_builtin_matches_krippendorff_package():
    krippendorff = pytest.importorskip("krippendorff")
    rng = np.random.default_rng(0)
    for _ in range(10):
        data = rng.integers(1, 6, size=(3, 40)).astype(float)
        data[rng.random(data.shape) < 0.2] = nan
        expected = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        assert hea._krippendorff_alpha_ordinal(data) == pytest.approx(expected, abs=1e-9)


def test_holm_correction():
    assert hea.holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adjusted = hea.holm_correction([0.2, nan, 0.9])
    assert adjusted[0] == pytest.approx(0.4) and np.isnan(adjusted[1]) and adjusted[2] == pytest.approx(0.9)


def test_wilcoxon_tests_smoke():
    pytest.importorskip("scipy")
    rows = []
    for annotator in ("ann1", "ann2"):
        for q in range(12):
            for dim in hea.DIMENSIONS:
                rows.append({"annotator": annotator, "question_id": q, "model_variant": "gemma_finetuned",
                             "label": "Model A", "dimension": dim, "score": 5})
                rows.append({"annotator": annotator, "question_id": q, "model_variant": "gemma_zeroshot",
                             "label": "Model B", "dimension": dim, "score": 2 if q % 3 else 3})
    out = hea.wilcoxon_tests(pd.DataFrame(rows))
    assert len(out) == len(hea.DIMENSIONS)
    assert (out["n_questions"] == 12).all()
    assert (out["p_holm"] >= out["p_value"]).all()
    assert out["significant_05"].all()
