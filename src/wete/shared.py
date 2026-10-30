"""Evaluators compared on the SAME scenarios.

BridgeSim names a NavHard scenario ``sd_<token>``, where <token> is the NAVSIM token of the scene's
current frame (bridgesim converters/openscene, commit ea5d45f). 341 of the 421 NavHard scenarios
are also navtest scenarios, so NAVSIM v1 PDMS and open-loop L2 already exist for them. Restricting
every evaluator to that shared set removes "different scenarios" as an explanation for different
rankings.

Reads the long score table written by ``wete.collect`` and writes, under --out:

    shared_agreement.csv    Population B: rank agreement between evaluators on the shared set
                            (paired bootstrap: the same scenario resample for every evaluator)
    family_contrasts.csv    Population B: family A minus family B (seed-averaged), per evaluator,
                            on the shared set and on each evaluator's full set; positive = A better
    contrast_correlation.csv  per-scenario correlation of those contrasts between evaluators
    popA_shared.csv         Population A and the specificity arm at level 3: mean drop per
                            evaluator on the shared set, and the per-scenario correlation of the
                            drop with the reference evaluator's drop
    shared_report.json      everything above plus the shared scenario count

    python -m wete.shared --scores <analysis dir>/scores.csv --meta configs/meta.bridgesim.json \
        --out <analysis dir>/shared [--B 1000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import stats as St

SEED_RE = re.compile(r"_seed\d+$")


def normalise_scenario(s: str) -> str:
    """BridgeSim 'sd_<token>' -> '<token>'; other ids unchanged."""
    s = str(s)
    return s[3:] if s.startswith("sd_") else s


def family_of(policy: str) -> str:
    return SEED_RE.sub("", policy)


def _matrix(df: pd.DataFrame, policies: Sequence[str], scenarios: Sequence[str]) -> np.ndarray:
    piv = df.pivot_table(index="policy", columns="scenario", values="score", aggfunc="mean")
    return piv.reindex(index=list(policies), columns=list(scenarios)).values.astype(float)


def population_b(scores: pd.DataFrame, meta: Dict[str, Dict], B: int,
                 rng: np.random.Generator) -> Dict:
    dfb = scores[scores.population == "B"]
    evals = [e for e in meta if e in set(dfb.evaluator)]
    if len(evals) < 2:
        return {}
    pols = sorted(set.intersection(*[set(dfb[dfb.evaluator == e].policy) for e in evals]))
    full = {}
    for e in evals:
        de = dfb[(dfb.evaluator == e) & dfb.policy.isin(pols)]
        sc = sorted(de.scenario.unique())
        M = _matrix(de, pols, sc)
        keep = ~np.isnan(M).any(axis=0)                 # drop scenarios invalid for any policy
        full[e] = (St.orient(M[:, keep], meta[e]["higher_is_better"]), [s for s, k in zip(sc, keep) if k])
    shared = sorted(set.intersection(*[set(sc) for _, sc in full.values()]))
    out: Dict = {"policies": pols, "evaluators": evals, "n_shared": len(shared),
                 "n_full": {e: len(sc) for e, (_, sc) in full.items()}}
    if len(shared) < 20:
        out["skipped"] = "fewer than 20 shared scenarios"
        return out
    mats = {}
    for e, (M, sc) in full.items():
        pos = {s: i for i, s in enumerate(sc)}
        mats[e] = M[:, [pos[s] for s in shared]]
    n = len(shared)
    idx = rng.integers(0, n, size=(B, n))               # one resample shared by all evaluators
    means = {e: M.mean(1) for e, M in mats.items()}
    draws = {e: M[:, idx].mean(-1) for e, M in mats.items()}   # (P, B)

    agree = []
    for a, b in itertools.combinations(evals, 2):
        td = np.array([St.kendall(draws[a][:, k], draws[b][:, k]) for k in range(B)])
        rd = np.array([St.spearman(draws[a][:, k], draws[b][:, k]) for k in range(B)])
        lo, hi = St.percentile_ci(td)
        rlo, rhi = St.percentile_ci(rd)
        agree.append({"pair": f"{a}|{b}", "n_scenarios": n, "tau": St.kendall(means[a], means[b]),
                      "tau_lo": float(lo), "tau_hi": float(hi), "rho": St.spearman(means[a], means[b]),
                      "rho_lo": float(rlo), "rho_hi": float(rhi)})
    out["agreement"] = agree
    out["policy_means"] = {e: {p: float(v) for p, v in zip(pols, means[e] * (1 if meta[e]["higher_is_better"] else -1))}
                           for e in evals}

    fams = sorted({family_of(p) for p in pols})
    rows_of = {f: [i for i, p in enumerate(pols) if family_of(p) == f] for f in fams}
    contrasts, corr = [], []
    per_scenario = {}
    for fa, fb in itertools.combinations(fams, 2):
        for e in evals:
            for scope, (M, nsc) in (("shared", (mats[e], n)), ("full", (full[e][0], len(full[e][1])))):
                d = M[rows_of[fa]].mean(0) - M[rows_of[fb]].mean(0)        # per scenario, oriented
                j = rng.integers(0, nsc, size=(B, nsc))
                dd = d[j].mean(1)
                lo, hi = St.percentile_ci(dd)
                contrasts.append({"contrast": f"{fa} - {fb}", "evaluator": e, "scope": scope,
                                  "n_scenarios": nsc, "mean": float(d.mean()), "lo": float(lo),
                                  "hi": float(hi), "p_positive": float((dd > 0).mean()),
                                  "units": "oriented: positive = first family better"})
                if scope == "shared":
                    per_scenario[(fa, fb, e)] = d
        for a, b in itertools.combinations(evals, 2):
            da, db = per_scenario[(fa, fb, a)], per_scenario[(fa, fb, b)]
            both = (da != 0) & (db != 0)
            corr.append({"contrast": f"{fa} - {fb}", "pair": f"{a}|{b}", "n_scenarios": n,
                         "spearman": St.spearman(da, db),
                         "sign_agreement": float((np.sign(da[both]) == np.sign(db[both])).mean()) if both.any() else np.nan,
                         "n_both_nonzero": int(both.sum())})
    out["family_contrasts"] = contrasts
    out["contrast_correlation"] = corr
    return out


def population_a(scores: pd.DataFrame, meta: Dict[str, Dict], level: int = 3) -> List[Dict]:
    """Level-`level` drop (oriented clean - degraded) per scenario on the shared set."""
    dfa = scores[scores.population == "A"]
    evals = [e for e in meta if e in set(dfa.evaluator)]
    refs = [e for e in evals if meta[e].get("reference")]
    rows = []
    for base in sorted(dfa.base.unique()):
        degs = sorted(set(dfa[(dfa.base == base) & (dfa.level == level)].degradation))
        for d in degs:
            drops = {}
            for e in evals:
                de = dfa[(dfa.evaluator == e) & (dfa.base == base)]
                c = de[de.level == 0].groupby("scenario").score.mean()
                x = de[(de.degradation == d) & (de.level == level)].groupby("scenario").score.mean()
                if c.empty or x.empty:
                    continue
                s = St.orient((c - x).dropna().values, meta[e]["higher_is_better"])
                drops[e] = pd.Series(s, index=(c - x).dropna().index)
            if len(drops) < 2:
                continue
            shared = sorted(set.intersection(*[set(v.index) for v in drops.values()]))
            if len(shared) < 20:
                continue
            for e, v in drops.items():
                row = {"base": base, "degradation": d, "level": level, "evaluator": e,
                       "n_shared": len(shared), "mean_drop_shared": float(v.loc[shared].mean()),
                       "mean_drop_full": float(v.mean())}
                for r in refs:
                    if r != e and r in drops:
                        row[f"spearman_with_{r}"] = St.spearman(v.loc[shared].values, drops[r].loc[shared].values)
                rows.append(row)
    return rows


def run(scores: pd.DataFrame, meta: Dict[str, Dict], out_dir: str, B: int = 1000, seed: int = 0) -> Dict:
    scores = scores.copy()
    scores["scenario"] = scores["scenario"].map(normalise_scenario)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    report: Dict = {"B": B, "seed": seed}
    pb = population_b(scores, meta, B, rng)
    report["population_b"] = pb
    if pb.get("agreement"):
        pd.DataFrame(pb["agreement"]).to_csv(out / "shared_agreement.csv", index=False)
        pd.DataFrame(pb["family_contrasts"]).to_csv(out / "family_contrasts.csv", index=False)
        pd.DataFrame(pb["contrast_correlation"]).to_csv(out / "contrast_correlation.csv", index=False)
    pa = population_a(scores, meta)
    report["population_a_level3"] = pa
    if pa:
        pd.DataFrame(pa).to_csv(out / "popA_shared.csv", index=False)
    (out / "shared_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    scores = pd.read_csv(a.scores, dtype={"scenario": str})
    meta = json.loads(Path(a.meta).read_text())
    rep = run(scores, meta, a.out, a.B, a.seed)
    pb = rep["population_b"]
    print(json.dumps({"n_shared": pb.get("n_shared"), "n_full": pb.get("n_full"),
                      "agreement": pb.get("agreement")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
