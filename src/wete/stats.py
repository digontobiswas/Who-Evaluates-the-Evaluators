"""Ranking statistics with scenario-level uncertainty.

Shapes
------
A *family* is one (evaluator, base policy, degradation type). Its scores are an
array S of shape (L, n): L degradation levels (level 0 first) x n scenarios,
already sign-adjusted so that HIGHER IS BETTER (use `orient`). The known order
is S-mean decreasing in level.

All bootstrap routines resample scenarios (columns) with replacement, jointly
across levels and across base policies that share a scenario set, because the
levels are evaluated on the same scenarios (paired design). Evaluators with
different scenario sets (navtest vs navhard vs HUGSIM scenes) are resampled
independently; differences between such evaluators combine independent draws.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------

def orient(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) * (1.0 if higher_is_better else -1.0)


def percentile_ci(draws: np.ndarray, alpha: float = 0.05, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=axis)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=axis)
    return lo, hi


def _pair_signs(means: np.ndarray) -> np.ndarray:
    """means: (L, ...) -> signs (P, ...) of means[i] - means[j] for i<j.
    +1 = correctly ordered (lower level scores higher)."""
    L = means.shape[0]
    pairs = list(itertools.combinations(range(L), 2))
    return np.stack([np.sign(means[i] - means[j]) for i, j in pairs], axis=0)


def order_tau(means: np.ndarray) -> np.ndarray:
    """Kendall tau-b between level means and the known order (-level).

    means: (L,) or (L, B). Ties in the means count as neither concordant nor
    discordant and shrink the denominator (tau-b). If all means are tied the
    evaluator does not discriminate at all and we return 0 (scipy returns nan).
    """
    means = np.asarray(means, dtype=np.float64)
    s = _pair_signs(means)
    n0 = s.shape[0]
    C = (s > 0).sum(0)
    D = (s < 0).sum(0)
    # ties among y (means). x (levels) has no ties.
    T = (s == 0).sum(0)
    # tau-b denominator: sqrt(n0 * (n0 - n_tied_pairs_in_y)) where tied pairs
    # in y are counted from tie groups; for small L, counting tied pairs
    # directly is equivalent.
    denom = np.sqrt(n0 * (n0 - T).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(denom > 0, (C - D) / denom, 0.0)
    return tau


def pairwise_accuracy(means: np.ndarray) -> np.ndarray:
    """Fraction of level pairs ordered correctly; ties count 1/2."""
    s = _pair_signs(np.asarray(means, dtype=np.float64))
    return ((s > 0) + 0.5 * (s == 0)).mean(0)


def full_order_correct(means: np.ndarray) -> np.ndarray:
    s = _pair_signs(np.asarray(means, dtype=np.float64))
    return (s > 0).all(0).astype(np.float64)


# --------------------------------------------------------------------------
# Ordering recovery with bootstrap
# --------------------------------------------------------------------------

@dataclass
class OrderingResult:
    tau: float
    tau_lo: float
    tau_hi: float
    pair_acc: float
    pair_acc_lo: float
    pair_acc_hi: float
    p_full_order: float          # bootstrap probability that the full order is recovered
    n_scenarios: int
    n_bases: int
    tau_draws: np.ndarray        # (B,) kept for downstream difference tests


def _aligned(S_list: Sequence[np.ndarray]) -> np.ndarray:
    """Stack bases -> (K, L, n), dropping scenarios with any NaN."""
    arr = np.stack([np.asarray(S, dtype=np.float64) for S in S_list], axis=0)
    keep = ~np.isnan(arr).any(axis=(0, 1))
    return arr[:, :, keep]


def ordering_recovery(S_list: Sequence[np.ndarray], B: int = 1000,
                      rng: Optional[np.random.Generator] = None,
                      alpha: float = 0.05) -> OrderingResult:
    """One cell of the discriminative-power matrix.

    S_list: one (L, n) array per base policy, same scenario columns.
    The statistic is the mean over base policies of tau(level means).
    """
    rng = rng or np.random.default_rng(0)
    arr = _aligned(S_list)                 # (K, L, n)
    K, L, n = arr.shape
    if n == 0:
        raise ValueError("no complete scenarios")
    point_means = arr.mean(-1)             # (K, L)
    tau = order_tau(point_means.T).mean()
    pa = pairwise_accuracy(point_means.T).mean()

    idx = rng.integers(0, n, size=(B, n))
    # (K, L, B): mean over resampled scenarios
    bm = arr[:, :, idx].mean(-1)
    tau_d = np.stack([order_tau(bm[k]) for k in range(K)]).mean(0)        # (B,)
    pa_d = np.stack([pairwise_accuracy(bm[k]) for k in range(K)]).mean(0)
    full_d = np.stack([full_order_correct(bm[k]) for k in range(K)]).min(0)
    tlo, thi = percentile_ci(tau_d, alpha)
    plo, phi = percentile_ci(pa_d, alpha)
    return OrderingResult(float(tau), float(tlo), float(thi), float(pa), float(plo),
                          float(phi), float(full_d.mean()), n, K, tau_d)


# --------------------------------------------------------------------------
# Monotonicity test (protocol step: is the degradation a valid ground truth?)
# --------------------------------------------------------------------------

@dataclass
class MonotonicityResult:
    passed: bool
    pair_prob: List[float]       # bootstrap P(mean drop > 0) per adjacent pair
    pair_mean_drop: List[float]
    pair_ci: List[Tuple[float, float]]
    pair_t_p: List[float]        # one-sided paired t-test p-value, for reference
    threshold: float


def monotonicity_test(S: np.ndarray, B: int = 1000, threshold: float = 0.95,
                      rng: Optional[np.random.Generator] = None) -> MonotonicityResult:
    """Paired bootstrap test that every adjacent level pair shows a drop.

    Pass iff P_boot(mean(S[l] - S[l+1]) > 0) > threshold for all l.
    This is an intersection-union test: requiring ALL adjacent pairs to pass,
    each at level 1-threshold, controls the type-I error of the conjunction at
    1-threshold without multiplicity correction (Berger 1982). It is
    conservative, not anti-conservative; do not add a Bonferroni factor.
    Note: the bootstrap P(mean>0) approximates 1 - one-sided p only for
    moderate n; the t-test p is reported alongside as a sanity check.
    """
    rng = rng or np.random.default_rng(0)
    S = np.asarray(S, dtype=np.float64)
    S = S[:, ~np.isnan(S).any(0)]
    L, n = S.shape
    idx = rng.integers(0, n, size=(B, n))
    probs, drops, cis, tps = [], [], [], []
    for l in range(L - 1):
        d = S[l] - S[l + 1]
        bd = d[idx].mean(-1)
        probs.append(float((bd > 0).mean()))
        drops.append(float(d.mean()))
        lo, hi = percentile_ci(bd)
        cis.append((float(lo), float(hi)))
        if np.allclose(d, d[0]):
            tps.append(0.0 if d[0] > 0 else 1.0)
        else:
            tps.append(float(sps.ttest_1samp(d, 0.0, alternative="greater").pvalue))
    return MonotonicityResult(all(p > threshold for p in probs), probs, drops, cis, tps, threshold)


# --------------------------------------------------------------------------
# Agreement between evaluators (Populations B, C)
# --------------------------------------------------------------------------

def kendall(a: np.ndarray, b: np.ndarray) -> float:
    t = sps.kendalltau(a, b)[0]
    return 0.0 if np.isnan(t) else float(t)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    r = sps.spearmanr(a, b)[0]
    return 0.0 if np.isnan(r) else float(r)


def mmrv(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean Maximum Rank Violation (Li et al., SIMPLER, CoRL 2024).

    MMRV = (1/N) sum_i max_j |R_i - R_j| * 1[(R_i < R_j) != (C_i < C_j)]
    R: reference performance values (e.g. closed-loop score), C: candidate.
    Lower is better; 0 = no rank violations. Units are those of R, so R should
    be on a fixed scale (e.g. [0, 1]) for MMRV to be comparable across studies.
    """
    R = np.asarray(reference, dtype=np.float64)
    C = np.asarray(candidate, dtype=np.float64)
    viol = (R[:, None] < R[None, :]) != (C[:, None] < C[None, :])
    w = np.abs(R[:, None] - R[None, :]) * viol
    return float(w.max(1).mean())


def tau_b_batch(a: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Kendall tau-b between a (n,) and each row of Bm (P, n); vectorised.
    Matches scipy.stats.kendalltau (variant 'b'); undefined -> 0."""
    a = np.asarray(a, float)
    Bm = np.atleast_2d(np.asarray(Bm, float))
    i, j = np.triu_indices(len(a), 1)
    sa = np.sign(a[i] - a[j])                      # (n_pairs,)
    sb = np.sign(Bm[:, i] - Bm[:, j])              # (P, n_pairs)
    num = (sa[None] * sb).sum(1)
    n1 = (sa != 0).sum()
    n2 = (sb != 0).sum(1)
    den = np.sqrt(n1 * n2.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def permutation_pvalue_tau(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                           rng: Optional[np.random.Generator] = None) -> float:
    """One-sided p-value for tau(a, b) > 0. Exact for n <= 8."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = float(tau_b_batch(a, b[None])[0])
    n = len(a)
    if n <= 8:
        perms = np.array(list(itertools.permutations(range(n))))
        vals = tau_b_batch(a, b[perms])
    else:
        rng = rng or np.random.default_rng(0)
        perms = np.argsort(rng.random((n_perm, n)), axis=1)
        vals = np.append(tau_b_batch(a, b[perms]), obs)  # include observed (valid MC p)
    return float((vals >= obs - 1e-12).mean())


def bootstrap_agreement(policy_scores: Dict[str, np.ndarray], B: int = 1000,
                        reference: Optional[str] = None,
                        rng: Optional[np.random.Generator] = None) -> Dict[str, Dict]:
    """Pairwise tau, rho (and MMRV against `reference`) between evaluators.

    policy_scores: evaluator -> (P, n_e) oriented score matrix, rows = the same
    P policies in the same order for every evaluator; scenario sets may differ,
    so each evaluator is resampled independently.
    """
    rng = rng or np.random.default_rng(0)
    names = list(policy_scores)
    means = {e: np.nanmean(policy_scores[e], 1) for e in names}
    draws = {}
    for e in names:
        M = policy_scores[e]
        n = M.shape[1]
        idx = rng.integers(0, n, size=(B, n))
        draws[e] = np.nanmean(M[:, idx], -1)       # (P, B)
    out: Dict[str, Dict] = {}
    for a, b in itertools.combinations(names, 2):
        td = np.array([kendall(draws[a][:, k], draws[b][:, k]) for k in range(B)])
        rd = np.array([spearman(draws[a][:, k], draws[b][:, k]) for k in range(B)])
        out[f"{a}|{b}"] = {
            "tau": kendall(means[a], means[b]), "tau_ci": tuple(map(float, percentile_ci(td))),
            "rho": spearman(means[a], means[b]), "rho_ci": tuple(map(float, percentile_ci(rd))),
            "tau_perm_p": permutation_pvalue_tau(means[a], means[b], rng=rng),
        }
    if reference is not None:
        for e in names:
            if e == reference:
                continue
            md = np.array([mmrv(draws[reference][:, k], draws[e][:, k]) for k in range(B)])
            out.setdefault(f"MMRV|{e}", {})
            out[f"MMRV|{e}"].update({"mmrv": mmrv(means[reference], means[e]),
                                     "mmrv_ci": tuple(map(float, percentile_ci(md)))})
    return out


# --------------------------------------------------------------------------
# Sample efficiency
# --------------------------------------------------------------------------

def sample_efficiency(S_list: Sequence[np.ndarray], grid: Sequence[int], R: int = 300,
                      rng: Optional[np.random.Generator] = None) -> Dict[int, Dict[str, float]]:
    """For each n in grid, subsample n scenarios WITHOUT replacement R times and
    report mean tau and P(full order correct for every base)."""
    rng = rng or np.random.default_rng(0)
    arr = _aligned(S_list)
    K, L, N = arr.shape
    out = {}
    for n in grid:
        if n > N:
            continue
        taus, full = [], []
        for _ in range(R):
            cols = rng.choice(N, size=n, replace=False)
            m = arr[:, :, cols].mean(-1)            # (K, L)
            taus.append(order_tau(m.T).mean())
            full.append(full_order_correct(m.T).min())
        out[int(n)] = {"tau": float(np.mean(taus)), "p_full_order": float(np.mean(full))}
    return out


def scenarios_to_stability(curve: Dict[int, Dict[str, float]], target: float = 0.9) -> Optional[int]:
    """Smallest n with P(full order) >= target, or None."""
    for n in sorted(curve):
        if curve[n]["p_full_order"] >= target:
            return n
    return None


# --------------------------------------------------------------------------
# Three-way verdicts for pre-registered hypotheses
# --------------------------------------------------------------------------

def verdict_greater(diff_draws: np.ndarray, margin: float = 0.0) -> Dict[str, object]:
    """Test 'difference > 0' from bootstrap draws of the difference.

    supported     : 95% CI lower bound > 0
    contradicted  : 95% CI upper bound < margin (difference is at most a
                    negligible `margin`, i.e. the effect the hypothesis needs is
                    ruled out -- an equivalence-style falsification)
    inconclusive  : otherwise (CI too wide; NOT evidence of absence)
    """
    lo, hi = percentile_ci(np.asarray(diff_draws))
    if lo > 0:
        v = "supported"
    elif hi < margin:
        v = "contradicted"
    else:
        v = "inconclusive"
    return {"diff": float(np.nanmean(diff_draws)), "ci": (float(lo), float(hi)),
            "verdict": v, "margin": margin}
