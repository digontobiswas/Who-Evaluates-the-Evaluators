"""Reliability-weighted scoring (proposal: Method, "Reliability-weighted score").

Pipeline for one evaluator E
----------------------------
1. Labels   y_E(s): does E, restricted to scenario s, order the Population A
            levels correctly? (pooled over TRAINING base policies and TRAINING
            degradation types only).
2. Predictor r_E(s) = sigmoid(w^T phi(s)) from cheap per-scenario features.
3. Score    S_w(pi) = sum_s r(s) E(pi, s) / sum_s r(s).
4. Held-out validation: leave-one-base-out x leave-one-degradation-out, AND a
   scenario split, so neither the family nor the scenario used for testing was
   seen when fitting r. Without the scenario split a flexible r could memorise
   per-scenario labels, which would test memorisation, not prediction.

Known confound (must be discussed in the paper)
-----------------------------------------------
A scenario can be "unreliable" for two different reasons:
  (a) the evaluator is wrong there (rendering breaks, world model hallucinates);
  (b) the degradation has no effect there (camera dropout on an empty road:
      a PERFECT evaluator correctly gives tied scores).
Mode "absolute" mixes both. Mode "effect_conditioned" drops (family, scenario)
cells where the reference evaluator E* itself sees no ordering, so labels
reflect (a) only -- at the cost of depending on E*, which is under study.
Report both.

Ceiling effect (must be handled in the paper)
---------------------------------------------
With hundreds of scenarios, Kendall tau of LEVEL MEANS saturates at 1 for any
evaluator that is sensitive at all, so "tau weighted vs uniform" (proposal H4)
can be uninformative. We therefore also report two ceiling-free metrics:
  * sep_z  : mean over adjacent level pairs of (weighted mean drop / its SE);
  * p_full@n: probability of recovering the full order from n subsampled
              scenarios (n small, e.g. 25).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .stats import order_tau, full_order_correct, percentile_ci

Family = Tuple[str, str]   # (base_policy, degradation)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def scenario_concordance(S: np.ndarray) -> np.ndarray:
    """S: (L, n) oriented scores. Per scenario, fraction of level pairs ordered
    correctly (ties = 1/2). Returns (n,)."""
    L = S.shape[0]
    vals = []
    for i, j in itertools.combinations(range(L), 2):
        d = np.sign(S[i] - S[j])
        vals.append((d > 0) + 0.5 * (d == 0))
    return np.mean(vals, axis=0)


def make_labels(families: Dict[Family, np.ndarray], use: Sequence[Family],
                theta: float = 0.5, mode: str = "absolute",
                reference: Optional[Dict[Family, np.ndarray]] = None,
                ref_min_conc: float = 0.5,
                behavior: Optional[Dict[Family, np.ndarray]] = None,
                eps: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Per-scenario reliability labels from the families in `use`.

    Returns (y, valid): y in {0,1}, valid marks scenarios with >= 1 usable cell.

    mode='absolute'            every (family, scenario) cell counts. Confounded
                               by per-scenario effect size (see module doc).
    mode='effect_conditioned'  a cell counts only if the REFERENCE evaluator's
                               concordance on s is > ref_min_conc. Needs E* on
                               the same scenarios (rarely available across
                               benchmarks) and inherits E*'s blind spots.
    mode='behavior_conditioned' a cell counts only if the degradation changes
                               the POLICY'S OUTPUT on s: behavior[fam][s] =
                               displacement (m) between the level-3 and clean
                               plans from the same input; cell used iff > eps.
                               Evaluator-independent; available for any
                               benchmark where plans can be logged.
    """
    num = None
    den = None
    for fam in use:
        c = scenario_concordance(families[fam])
        w = np.ones_like(c)
        if mode == "effect_conditioned":
            if reference is None:
                raise ValueError("effect_conditioned labels need a reference evaluator")
            w = (scenario_concordance(reference[fam]) > ref_min_conc).astype(float)
        elif mode == "behavior_conditioned":
            if behavior is None:
                raise ValueError("behavior_conditioned labels need per-scenario plan deltas")
            w = (np.asarray(behavior[fam]) > eps).astype(float)
        elif mode != "absolute":
            raise ValueError(mode)
        num = c * w if num is None else num + c * w
        den = w if den is None else den + w
    valid = den > 0
    conc = np.where(valid, num / np.maximum(den, 1e-12), np.nan)
    y = (conc > theta).astype(int)
    return y, valid


# --------------------------------------------------------------------------
# Predictor and weighted score
# --------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U / (n_pos n_neg)), ties get average
    rank. Equals sklearn.metrics.roc_auc_score but ~100x cheaper per call,
    which matters inside bootstraps. nan if only one class is present."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_model(kind: str = "logreg", seed: int = 0):
    if kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=500, early_stopping=True,
                            n_iter_no_change=20, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, kind: str = "logreg") -> np.ndarray:
    counts = np.bincount(np.asarray(y_tr, int), minlength=2) if len(y_tr) else np.zeros(2, int)
    if counts.min() < 5:
        # degenerate: (almost) all train scenarios reliable, or none -> a
        # classifier cannot be fitted meaningfully; use uniform weights
        return np.full(len(X_te), float(y_tr.mean()) if len(y_tr) else 0.5)
    m = make_model(kind)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def weighted_means(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """S (L, n), w (n,) -> (L,) reliability-weighted level means (eq. S_w)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 1e-6, None)
    return (S * w).sum(-1) / w.sum()


def separation_z(S: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Ceiling-free sensitivity: mean over adjacent pairs of weighted mean drop
    divided by its standard error (Kish effective sample size)."""
    L, n = S.shape
    w = np.ones(n) if w is None else np.clip(np.asarray(w, float), 1e-6, None)
    wn = w / w.sum()
    n_eff = 1.0 / (wn ** 2).sum()
    zs = []
    for l in range(L - 1):
        d = S[l] - S[l + 1]
        m = (wn * d).sum()
        var = (wn * (d - m) ** 2).sum() * n_eff / max(n_eff - 1, 1)
        se = np.sqrt(var / n_eff)
        zs.append(m / se if se > 0 else (np.sign(m) * 10.0))
    return float(np.mean(zs))


def p_full_at_n(S: np.ndarray, w: Optional[np.ndarray], n_sub: int, R: int,
                rng: np.random.Generator) -> float:
    L, n = S.shape
    if n_sub >= n:
        return float(full_order_correct(weighted_means(S, np.ones(n) if w is None else w)))
    # R subsets without replacement, vectorised: argsort of uniforms
    cols = np.argsort(rng.random((R, n)), axis=1)[:, :n_sub]            # (R, n_sub)
    ww = np.ones((R, n_sub)) if w is None else np.clip(np.asarray(w, float), 1e-6, None)[cols]
    m = (S[:, cols] * ww[None]).sum(-1) / ww.sum(-1)[None]              # (L, R)
    return float(full_order_correct(m).mean())


# --------------------------------------------------------------------------
# Held-out cross-validation
# --------------------------------------------------------------------------

@dataclass
class FoldResult:
    test_family: Family
    auroc: float
    tau_uniform: float
    tau_weighted: float
    z_uniform: float
    z_weighted: float
    pfull_uniform: float
    pfull_weighted: float
    n_test: int
    gain_tau_draws: np.ndarray = field(repr=False)
    gain_z_draws: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    r_test: np.ndarray = field(repr=False)


def heldout_cv(families: Dict[Family, np.ndarray], features: np.ndarray,
               theta: float = 0.5, mode: str = "absolute",
               reference: Optional[Dict[Family, np.ndarray]] = None,
               behavior: Optional[Dict[Family, np.ndarray]] = None, eps: float = 0.1,
               kind: str = "logreg", n_scen_folds: int = 2, n_sub: int = 25,
               B: int = 500, R: int = 200, seed: int = 0) -> List[FoldResult]:
    """Leave-one-base x leave-one-degradation x scenario-fold CV.

    families: (base, deg) -> (L, n) oriented scores on a SHARED scenario axis.
    features: (n, d) cheap per-scenario features (policy-agnostic).
    """
    rng = np.random.default_rng(seed)
    bases = sorted({b for b, _ in families})
    degs = sorted({d for _, d in families})
    if len(bases) < 2 or len(degs) < 2:
        raise ValueError("held-out CV needs >= 2 base policies and >= 2 degradation types")
    n = features.shape[0]
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_scen_folds)
    results: List[FoldResult] = []
    for b, k in itertools.product(bases, degs):
        if (b, k) not in families:
            continue
        train_fams = [f for f in families if f[0] != b and f[1] != k]
        if not train_fams:
            continue
        ys_all, rs_all = [], []
        test_cols_all = []
        r_full = np.empty(n)
        for f_idx, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            y_tr, v_tr = make_labels({f: families[f][:, tr] for f in train_fams}, train_fams,
                                     theta, mode,
                                     None if reference is None else {f: reference[f][:, tr] for f in train_fams},
                                     behavior=None if behavior is None else
                                     {f: np.asarray(behavior[f])[tr] for f in train_fams}, eps=eps)
            r_full[te] = fit_predict(features[tr][v_tr], y_tr[v_tr], features[te], kind)
            test_cols_all.append(te)
        S = families[(b, k)]
        y_te, v_te = make_labels({(b, k): S}, [(b, k)], theta, mode,
                                 None if reference is None else {(b, k): reference[(b, k)]},
                                 behavior=None if behavior is None else {(b, k): behavior[(b, k)]}, eps=eps)
        au = auroc(y_te[v_te], r_full[v_te])
        # ordering on the held-out family; every scenario is out-of-fold here
        tu = float(order_tau(weighted_means(S, np.ones(n))))
        tw = float(order_tau(weighted_means(S, r_full)))
        zu, zw = separation_z(S), separation_z(S, r_full)
        pu = p_full_at_n(S, None, n_sub, R, rng)
        pw = p_full_at_n(S, r_full, n_sub, R, rng)
        idx = rng.integers(0, n, size=(B, n))
        gt = np.array([order_tau(weighted_means(S[:, i], r_full[i])) - order_tau(weighted_means(S[:, i], np.ones(n)))
                       for i in idx])
        gz = np.array([separation_z(S[:, i], r_full[i]) - separation_z(S[:, i]) for i in idx])
        results.append(FoldResult((b, k), au, tu, tw, zu, zw, pu, pw, n, gt, gz,
                                  np.where(v_te, y_te, -1), r_full))
    return results


def summarize_cv(results: List[FoldResult]) -> Dict[str, object]:
    au = np.array([r.auroc for r in results])
    gt = np.stack([r.gain_tau_draws for r in results]).mean(0)
    gz = np.stack([r.gain_z_draws for r in results]).mean(0)
    # AUROC CI: bootstrap over SCENARIOS, jointly for all folds (predictions
    # r held fixed). Folds share scenarios and features, so treating folds as
    # independent units gives CIs that are far too narrow (seen on the toy:
    # a pure-noise feature got CI [0.505, 0.530]).
    rng = np.random.default_rng(0)
    valid = au[~np.isnan(au)]
    if len(valid):
        n = len(results[0].y_test)
        draws = []
        for _ in range(500):
            idx = rng.integers(0, n, n)
            vals = []
            for r in results:
                y, s = r.y_test[idx], r.r_test[idx]
                ok = y >= 0                       # -1 = no usable label
                y, s = y[ok], s[ok]
                a_ = auroc(y, s)
                if not np.isnan(a_):
                    vals.append(a_)
            if vals:
                draws.append(np.mean(vals))
        au_ci = tuple(map(float, percentile_ci(np.array(draws))))
    else:
        au_ci = (float("nan"), float("nan"))
    return {
        "n_folds": len(results),
        "auroc_mean": float(np.nanmean(au)) if len(valid) else float("nan"),
        "auroc_ci_over_families": au_ci,
        "tau_uniform": float(np.mean([r.tau_uniform for r in results])),
        "tau_weighted": float(np.mean([r.tau_weighted for r in results])),
        "gain_tau": float(gt.mean()), "gain_tau_ci": tuple(map(float, percentile_ci(gt))),
        "z_uniform": float(np.mean([r.z_uniform for r in results])),
        "z_weighted": float(np.mean([r.z_weighted for r in results])),
        "gain_z": float(gz.mean()), "gain_z_ci": tuple(map(float, percentile_ci(gz))),
        "pfull_uniform": float(np.mean([r.pfull_uniform for r in results])),
        "pfull_weighted": float(np.mean([r.pfull_weighted for r in results])),
        "_gain_tau_draws": gt, "_gain_z_draws": gz,
    }


# --------------------------------------------------------------------------
# Label-noise ceiling for H3
# --------------------------------------------------------------------------

def label_ceiling_auroc(families: Dict[Family, np.ndarray], theta: float = 0.5,
                        n_splits: int = 50, seed: int = 0) -> Dict[str, float]:
    """Split-half estimate of how predictable the per-scenario label is AT ALL.

    Families are split at random into halves A and B. The continuous
    concordance on A is used as a 'predictor' of the binary label on B. Its
    AUROC estimates the ceiling attainable by any scenario-level predictor
    given label noise (with ~half the families; with all families labels are
    less noisy, so this is a conservative-low estimate of the ceiling).

    Why it matters: with 4 levels a scenario label rests on only 6 pairwise
    comparisons per family, and on the toy world even the TRUE planted
    reliability variable reaches only AUROC ~0.67-0.70. A fixed H3 threshold
    of 0.7 must therefore be read against this ceiling.
    """
    rng = np.random.default_rng(seed)
    keys = list(families)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(len(keys))
        A = [keys[i] for i in perm[: len(keys) // 2]]
        B = [keys[i] for i in perm[len(keys) // 2:]]
        cA = np.mean([scenario_concordance(families[k]) for k in A], axis=0)
        yB, vB = make_labels(families, B, theta)
        a = auroc(yB[vB], cA[vB])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return {"ceiling_auroc": float("nan"), "ci": (float("nan"), float("nan"))}
    return {"ceiling_auroc": float(np.mean(vals)), "ci": tuple(map(float, percentile_ci(np.array(vals))))}


# --------------------------------------------------------------------------
# Fallback: discriminative scenario subset
# --------------------------------------------------------------------------

def discriminative_subset_cv(families: Dict[Family, np.ndarray],
                             fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
                             seed: int = 0) -> Dict[str, object]:
    """Rank scenarios by training-family concordance, keep the top fraction,
    and test on the held-out family whether the subset orders it as well with
    fewer scenarios. Selection uses training families only."""
    bases = sorted({b for b, _ in families})
    degs = sorted({d for _, d in families})
    rows = []
    for b, k in itertools.product(bases, degs):
        if (b, k) not in families:
            continue
        train = [f for f in families if f[0] != b and f[1] != k]
        if not train:
            continue
        conc = np.mean([scenario_concordance(families[f]) for f in train], axis=0)
        order = np.argsort(-conc)
        S = families[(b, k)]
        n = S.shape[1]
        for frac in fractions:
            m = max(4, int(round(frac * n)))
            cols = order[:m]
            zs, zf = separation_z(S[:, cols]), separation_z(S)
            # z grows like sqrt(n); the per-scenario effect size d = z/sqrt(n)
            # is the fair comparison between a subset and the full set.
            rows.append({"family": f"{b}/{k}", "frac": frac, "m": m,
                         "z_subset": zs, "z_full": zf,
                         "d_subset": zs / np.sqrt(m), "d_full": zf / np.sqrt(n),
                         "tau_subset": float(order_tau(S[:, cols].mean(1))),
                         "tau_full": float(order_tau(S.mean(1)))})
    df = pd.DataFrame(rows)
    agg = df.groupby("frac")[["d_subset", "d_full", "z_subset", "z_full", "tau_subset", "tau_full"]].mean()
    return {"per_family": df, "by_fraction": agg}
