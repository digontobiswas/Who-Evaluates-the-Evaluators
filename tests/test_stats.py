import itertools

import numpy as np
import pytest
from scipy import stats as sps

from wete import stats as St
from wete.degradations import calibrate_levels


def test_order_tau_matches_scipy_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = np.round(rng.normal(size=4), 1)          # rounding creates ties
        ref = sps.kendalltau(m, -np.arange(4)).statistic
        ref = 0.0 if np.isnan(ref) else ref
        assert St.order_tau(m) == pytest.approx(ref, abs=1e-12)


def test_order_tau_extremes():
    assert St.order_tau(np.array([4.0, 3, 2, 1])) == pytest.approx(1.0)
    assert St.order_tau(np.array([1.0, 2, 3, 4])) == pytest.approx(-1.0)
    assert St.order_tau(np.array([1.0, 1, 1, 1])) == 0.0


def test_tau_b_batch_matches_scipy():
    rng = np.random.default_rng(1)
    a = np.round(rng.normal(size=7), 1)
    B = np.round(rng.normal(size=(50, 7)), 1)
    got = St.tau_b_batch(a, B)
    for k in range(50):
        r = sps.kendalltau(a, B[k]).statistic
        assert got[k] == pytest.approx(0.0 if np.isnan(r) else r, abs=1e-12)


def test_mmrv_hand_examples():
    R = np.array([0.9, 0.5, 0.1])
    assert St.mmrv(R, R) == 0.0
    assert St.mmrv(R, R[::-1]) == pytest.approx((0.8 + 0.4 + 0.8) / 3)


def test_permutation_exact_n4():
    a = np.arange(4.0)
    assert St.permutation_pvalue_tau(a, a) == pytest.approx(1 / 24)


def test_monotonicity_type1_error_single_pair():
    """Under H0 (no drop) a single adjacent pair should pass at about 5 %."""
    rng = np.random.default_rng(2)
    passes = 0
    for i in range(400):
        S = rng.normal(size=(2, 80))
        passes += St.monotonicity_test(S, B=400, rng=np.random.default_rng(i)).passed
    rate = passes / 400
    assert 0.015 <= rate <= 0.10, rate


def test_monotonicity_conjunction_is_conservative_and_powerful():
    rng = np.random.default_rng(3)
    null = sum(St.monotonicity_test(rng.normal(size=(4, 80)), B=300,
                                    rng=np.random.default_rng(i)).passed for i in range(200))
    assert null / 200 <= 0.05
    drops = np.array([0, -0.7, -1.4, -2.1])[:, None]   # z ~ 4.4 per pair
    power = sum(St.monotonicity_test(drops + rng.normal(size=(4, 80)), B=300,
                                     rng=np.random.default_rng(i)).passed for i in range(50))
    assert power / 50 >= 0.9


def test_ordering_recovery_null_ci_covers_zero_mostly():
    rng = np.random.default_rng(4)
    cover = 0
    for i in range(60):
        S = [rng.normal(size=(4, 100)) for _ in range(3)]
        r = St.ordering_recovery(S, B=300, rng=np.random.default_rng(i))
        cover += r.tau_lo <= 0 <= r.tau_hi
    assert cover / 60 >= 0.85


def test_verdict_three_way():
    assert St.verdict_greater(np.full(100, 0.5) + np.linspace(-0.1, 0.1, 100))["verdict"] == "supported"
    assert St.verdict_greater(np.linspace(-0.05, 0.05, 100), margin=0.1)["verdict"] == "contradicted"
    assert St.verdict_greater(np.linspace(-0.5, 0.5, 100), margin=0.1)["verdict"] == "inconclusive"


def test_calibrate_levels():
    sweep = {0.0: 1.0, 0.1: 0.95, 0.2: 0.85, 0.3: 0.7, 0.5: 0.5, 0.8: 0.3, 1.0: 0.25}
    lv = calibrate_levels(sweep, 0.0, 3)
    assert lv == sorted(lv) and len(set(lv)) == 3
    # flat sweep -> not calibratable
    assert calibrate_levels({0.0: 1.0, 0.5: 1.0, 1.0: 1.0}, 0.0, 3) is None
    # non-monotone start (mild degradation helps) is skipped, not chosen
    lv2 = calibrate_levels({0.0: 0.7, 1.0: 0.75, 2.0: 0.65, 3.0: 0.55, 4.0: 0.5, 6.0: 0.4}, 0.0, 3)
    assert 1.0 not in lv2
    # decreasing parameter (history kept fraction)
    lv3 = calibrate_levels({1.0: 0.9, 0.75: 0.8, 0.5: 0.6, 0.25: 0.4, 0.0: 0.2}, 1.0, 3)
    assert lv3 == sorted(lv3, reverse=True)


def test_sample_efficiency_monotone_in_n():
    rng = np.random.default_rng(5)
    S = [np.array([0.0, -0.2, -0.4, -0.6])[:, None] + rng.normal(size=(4, 400)) for _ in range(2)]
    c = St.sample_efficiency(S, [10, 50, 200], R=200, rng=rng)
    assert c[10]["p_full_order"] < c[50]["p_full_order"] < c[200]["p_full_order"] + 1e-9
