import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from wete import reliability as R


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.random(120) < 0.4
        s = np.round(rng.random(120) + 0.5 * y, 1)
        assert R.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(R.auroc(np.ones(5), np.arange(5)))


def test_weighted_means_and_concordance():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(R.weighted_means(S, np.array([1.0, 0.0])), [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(R.scenario_concordance(S), [1.0, 0.0])
    np.testing.assert_allclose(R.scenario_concordance(np.zeros((4, 3))), [0.5, 0.5, 0.5])


def _planted(n=300, seed=0):
    """Families whose per-scenario reliability depends on feature x."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    fams = {}
    for b in ("b0", "b1", "b2"):
        for d in ("d0", "d1", "d2"):
            true = np.array([0.0, -0.3, -0.6, -0.9])[:, None] + 0.1 * rng.normal(size=(4, n))
            noise = rng.normal(size=(4, n)) * (3.0 * x)[None]   # unreliable where x is large
            fams[(b, d)] = true + noise
    return fams, x[:, None]


def test_heldout_cv_recovers_planted_reliability():
    fams, X = _planted()
    res = R.heldout_cv(fams, X, B=50, R=50)
    s = R.summarize_cv(res)
    # X is the TRUE planted variable, yet per-scenario labels from 6 level
    # pairs are noisy: AUROC ~0.67. Compare with the split-half label ceiling.
    assert s["auroc_ci_over_families"][0] > 0.6
    assert s["gain_z_ci"][0] > 0
    ceil = R.label_ceiling_auroc(fams)
    assert s["auroc_mean"] <= ceil["ceiling_auroc"] + 0.1


def test_heldout_cv_negative_control_coverage():
    """95 % CI for a pure-noise feature should cover 0.5 in most replicates."""
    fams, _ = _planted()
    cover = 0
    for seed in range(12):
        noise = np.random.default_rng(100 + seed).normal(size=(300, 1))
        lo, hi = R.summarize_cv(R.heldout_cv(fams, noise, B=30, R=30, seed=seed))["auroc_ci_over_families"]
        cover += lo <= 0.5 <= hi
    assert cover >= 9
