"""Analyses requested in the review of the first submission (Nov 2026), from per-scenario scores.

    python -m wete.cli.revision --scores <analysis dir>/scores.csv --results results --out <dir>

Reads the scores.csv that ``python -m wete.collect`` wrote for every run on disk, and the
per-scenario BridgeSim files under results/. Writes one CSV per item and summary.json:

  1  projection_specificity.csv  benign-to-harmful ratio of the projected NAVSIM score and L2
     projection_agreement.csv    Kendall tau of full / projected scores against held-out BridgeSim
     projection_decomposition.csv  each family contrast: full = projected part + tail part
  2  head_tail.csv               drop(head) + drop(tail) against drop(full lateral noise)
  3  rr10.csv                    benign twins with replanning every 1.0 s: TOST within +-1 point
  4  late_braking.csv            closed-loop metrics per late-braking level (why DS stays flat)
  5  contrasts.csv               every family contrast, NAVSIM v1 vs held-out BridgeSim
  6  h1.csv                      tau at fixed scene counts, closed loop minus each proxy

Scenes. Population A measurements use the scenes outside NavHard (the selection set of the main
analysis): navtest minus NavHard for NAVSIM and L2, the held-out scenes for BridgeSim. Item 3
uses NavHard itself (its own protocol variant). Population B agreement uses every scene each
evaluator scored, as in the paper; contrasts are also given on the scenes shared with BridgeSim.
Every interval is a percentile bootstrap over scenes (B resamples, paired wherever two numbers
come from the same scenes).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from wete import stats as St

HIGHER_IS_BETTER = {"open_l2": False, "open_l2_proj": False}       # every other evaluator: True
BASES = ("transfuser_seed0", "ltf_seed0", "diffusiondrive_seed0")
EQUIV = 0.01                     # benign margin: 1 point of a 0-1 score (DECISIONS item 10)
PENALISE_SHARE = 0.25            # "penalises" if P(drop>0) > 0.95 and drop >= 25% of the twin's
TWIN = {"tail_lateral": "lateral_noise", "tail_stretch": "late_braking"}


def key(s: str) -> str:
    s = str(s)
    return s[3:] if s.startswith("sd_") else s


def family(policy: str) -> str:
    return str(policy).rsplit("_seed", 1)[0]


def oriented(df: pd.DataFrame) -> pd.Series:
    """Score with higher = better for every evaluator."""
    sign = df.evaluator.map(lambda e: 1.0 if HIGHER_IS_BETTER.get(e, True) else -1.0)
    return df.score * sign


def boot_mean(x: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B bootstrap means of x (scenes resampled)."""
    n = len(x)
    if n == 0:
        return np.full(B, np.nan)
    return x[rng.integers(0, n, size=(B, n))].mean(axis=1)


def ci(draws: np.ndarray, level: float = 0.95):
    lo, hi = np.nanpercentile(draws, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


class Scores:
    """scores.csv with oriented scores and helpers to pivot a run against its clean run."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["key"] = df.scenario.map(key)
        df["o"] = oriented(df)
        self.df = df
        sel = df[df.evaluator == "bridgesim"]
        self.navhard = set(sel.key)                      # selection set of the main analysis

    def vector(self, evaluator: str, population: str, policy: str, outside_navhard: bool) -> pd.Series:
        d = self.df[(self.df.evaluator == evaluator) & (self.df.population == population) & (self.df.policy == policy)]
        if outside_navhard:
            d = d[~d.key.isin(self.navhard)]
        return d.drop_duplicates("key").set_index("key").o.dropna()

    def run(self, evaluator: str, base: str, degradation: str, level: int, outside_navhard: bool = True) -> pd.Series:
        pol = base if degradation == "none" or level == 0 else f"{base}:{degradation}@L{level}"
        d = self.df[(self.df.evaluator == evaluator) & (self.df.population == "A") & (self.df.base == base)]
        d = d[(d.degradation == degradation) & (d.level == level)] if level else d[d.level == 0]
        if outside_navhard:
            d = d[~d.key.isin(self.navhard)]
        if d.empty:
            return pd.Series(dtype=float, name=pol)
        return d.drop_duplicates("key").set_index("key").o.dropna()


def paired_drop(clean: pd.Series, deg: pd.Series) -> np.ndarray:
    """Per-scene drop (clean - degraded, oriented: positive = worse) on the scenes both scored."""
    common = clean.index.intersection(deg.index)
    return (clean.loc[common] - deg.loc[common]).values


# ------------------------------------------------------------------------------------------------
# 1. Projection
# ------------------------------------------------------------------------------------------------
def projection_specificity(S: Scores, B: int, rng) -> pd.DataFrame:
    rows = []
    for e in ("navsim_v1", "navsim_v1_proj", "open_l2", "open_l2_proj"):
        for base in BASES:
            clean = S.run(e, base, "none", 0)
            if clean.empty:
                continue
            for benign, harmful in TWIN.items():
                for lvl in (1, 2, 3):
                    b, h = S.run(e, base, benign, lvl), S.run(e, base, harmful, lvl)
                    if b.empty:
                        continue
                    db = paired_drop(clean, b)
                    draws = boot_mean(db, B, rng)
                    lo, hi = ci(draws)
                    row = {"evaluator": e, "base": base, "perturbation": benign, "level": lvl,
                           "n_scenarios": len(db), "benign_drop": db.mean(), "benign_lo": lo, "benign_hi": hi,
                           "p_benign_positive": float((draws > 0).mean())}
                    if not h.empty:
                        dh = paired_drop(clean, h)
                        row.update({"harmful_drop": dh.mean(),
                                    "ratio": db.mean() / dh.mean() if abs(dh.mean()) > 1e-12 else np.nan})
                    rows.append(row)
    return pd.DataFrame(rows)


def agreement(S: Scores, pairs: Sequence[tuple], B: int, rng, shared_only: bool = False) -> pd.DataFrame:
    """Kendall tau between evaluators' policy means (Population B), scenes bootstrapped per evaluator."""
    dB = S.df[S.df.population == "B"]
    rows = []
    for a, b in pairs:
        pa, pb = set(dB[dB.evaluator == a].policy), set(dB[dB.evaluator == b].policy)
        pols = sorted(pa & pb)
        if len(pols) < 3:
            continue
        mats = {}
        keys = None
        if shared_only:
            keys = set(dB[dB.evaluator == a].key) & set(dB[dB.evaluator == b].key)
        for e in (a, b):
            d = dB[(dB.evaluator == e) & dB.policy.isin(pols)]
            if keys is not None:
                d = d[d.key.isin(keys)]
            piv = d.pivot_table(index="policy", columns="key", values="o").loc[pols]
            mats[e] = piv.dropna(axis=1).values
        if shared_only:      # identical scenes: one paired resample for both
            n = mats[a].shape[1]
            idx = rng.integers(0, n, size=(B, n))
            da, db = mats[a][:, idx].mean(-1), mats[b][:, idx].mean(-1)
        else:
            da = mats[a][:, rng.integers(0, mats[a].shape[1], size=(B, mats[a].shape[1]))].mean(-1)
            db = mats[b][:, rng.integers(0, mats[b].shape[1], size=(B, mats[b].shape[1]))].mean(-1)
        taus = np.array([St.kendall(da[:, k], db[:, k]) for k in range(B)])
        lo, hi = ci(taus)
        rows.append({"pair": f"{a}|{b}", "scenes": "shared" if shared_only else "all",
                     "n_policies": len(pols), "n_scenes_a": mats[a].shape[1], "n_scenes_b": mats[b].shape[1],
                     "tau": St.kendall(mats[a].mean(1), mats[b].mean(1)), "tau_lo": lo, "tau_hi": hi,
                     "policies": " ".join(pols)})
    return pd.DataFrame(rows)


def family_matrix(S: Scores, evaluator: str, keys: Optional[set], outside_navhard: bool) -> pd.DataFrame:
    """Population B: rows = family, columns = scene key, value = mean over the family's seeds."""
    d = S.df[(S.df.evaluator == evaluator) & (S.df.population == "B")]
    if outside_navhard:
        d = d[~d.key.isin(S.navhard)]
    if keys is not None:
        d = d[d.key.isin(keys)]
    piv = d.pivot_table(index="policy", columns="key", values="o").dropna(axis=1)
    return piv.groupby(piv.index.map(family)).mean()


def decomposition(S: Scores, B: int, rng) -> pd.DataFrame:
    """Full NAVSIM contrast = projected contrast + tail part, per family pair."""
    rows = []
    shared = set(S.df[S.df.evaluator == "bridgesim_ext"].key)
    for scope, keys, outside in (("navtest_outside_navhard", None, True), ("shared_heldout", shared, False)):
        full = family_matrix(S, "navsim_v1", keys, outside)
        proj = family_matrix(S, "navsim_v1_proj", keys, outside)
        ref = family_matrix(S, "bridgesim_ext", keys, False) if keys is not None else None
        common = full.columns.intersection(proj.columns)
        fams = sorted(set(full.index) & set(proj.index))
        for fa, fb in itertools.combinations(fams, 2):
            f = (full.loc[fa, common] - full.loc[fb, common]).values
            p = (proj.loc[fa, common] - proj.loc[fb, common]).values
            t = f - p
            idx = rng.integers(0, len(common), size=(B, len(common)))
            row = {"contrast": f"{fa} - {fb}", "scenes": scope, "n_scenes": len(common)}
            for name, v in (("full", f), ("projected", p), ("tail_part", t)):
                draws = v[idx].mean(1)
                row.update({name: v.mean(), f"{name}_lo": ci(draws)[0], f"{name}_hi": ci(draws)[1]})
            row["tail_share"] = t.mean() / f.mean() if abs(f.mean()) > 1e-12 else np.nan
            if ref is not None and fa in ref.index and fb in ref.index:
                r = (ref.loc[fa] - ref.loc[fb]).values
                draws = r[rng.integers(0, len(r), size=(B, len(r)))].mean(1)
                row.update({"bridgesim": r.mean(), "bridgesim_lo": ci(draws)[0], "bridgesim_hi": ci(draws)[1]})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# 2. Head + tail against full
# ------------------------------------------------------------------------------------------------
def head_tail(S: Scores, B: int, rng) -> pd.DataFrame:
    rows = []
    for e in ("navsim_v1", "navsim_v2", "open_l2"):
        for base in BASES:
            clean = S.run(e, base, "none", 0)
            if clean.empty:
                continue
            for lvl in (1, 2, 3):
                runs = {k: S.run(e, base, k, lvl) for k in ("head_lateral", "tail_lateral", "lateral_noise")}
                if any(r.empty for r in runs.values()):
                    continue
                common = clean.index
                for r in runs.values():
                    common = common.intersection(r.index)
                drops = {k: (clean.loc[common] - r.loc[common]).values for k, r in runs.items()}
                idx = rng.integers(0, len(common), size=(B, len(common)))
                d = {k: v[idx].mean(1) for k, v in drops.items()}
                gap = drops["head_lateral"] + drops["tail_lateral"] - drops["lateral_noise"]
                g = gap[idx].mean(1)
                full = drops["lateral_noise"].mean()
                rows.append({
                    "evaluator": e, "base": base, "level": lvl, "n_scenarios": len(common),
                    "drop_head": drops["head_lateral"].mean(), "head_lo": ci(d["head_lateral"])[0], "head_hi": ci(d["head_lateral"])[1],
                    "drop_tail": drops["tail_lateral"].mean(), "tail_lo": ci(d["tail_lateral"])[0], "tail_hi": ci(d["tail_lateral"])[1],
                    "drop_full": full, "full_lo": ci(d["lateral_noise"])[0], "full_hi": ci(d["lateral_noise"])[1],
                    "head_plus_tail_minus_full": gap.mean(), "gap_lo": ci(g)[0], "gap_hi": ci(g)[1],
                    "additive": bool(ci(g)[0] <= 0 <= ci(g)[1]),
                    "head_share": drops["head_lateral"].mean() / full if abs(full) > 1e-12 else np.nan,
                    "tail_share": drops["tail_lateral"].mean() / full if abs(full) > 1e-12 else np.nan,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# 3. Replanning every 1.0 s
# ------------------------------------------------------------------------------------------------
def rr10(S: Scores, B: int, rng) -> pd.DataFrame:
    rows = []
    for e, cut in (("bridgesim_rr10", 1.5), ("bridgesim_rr10_t20", 2.0)):
        for base in ("transfuser_seed0", "ltf_seed0"):
            clean = S.run(e, base, "none", 0, outside_navhard=False)
            if clean.empty:
                continue
            for pert in TWIN:
                for lvl in (1, 2, 3):
                    r = S.run(e, base, pert, lvl, outside_navhard=False)
                    if r.empty:
                        continue
                    dr = paired_drop(clean, r)
                    draws = boot_mean(dr, B, rng)
                    lo90, hi90 = ci(draws, 0.90)
                    rows.append({"evaluator": e, "tail_t_cut_s": cut, "base": base, "perturbation": pert,
                                 "level": lvl, "n_scenarios": len(dr), "mean_drop": dr.mean(),
                                 "ci90_lo": lo90, "ci90_hi": hi90, "p_drop_positive": float((draws > 0).mean()),
                                 "equivalent": bool(-EQUIV < lo90 and hi90 < EQUIV)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# 4. Late braking in closed loop
# ------------------------------------------------------------------------------------------------
LB_METRICS = ("DS", "RC", "EPDMS_no_ep", "NC", "DAC", "TTC", "EC", "HC", "LK", "TL")


def late_braking(results: Path, B: int, rng) -> pd.DataFrame:
    rows = []
    for base in ("transfuser_seed0", "ltf_seed0"):
        root = results / "popA_bridgesim" / base
        clean_f = root / "none__L0" / "per_scenario.csv"
        if not clean_f.is_file():
            continue
        clean = pd.read_csv(clean_f).set_index("scenario")
        for lvl in (1, 2, 3):
            f = root / f"late_braking__L{lvl}" / "per_scenario.csv"
            if not f.is_file():
                continue
            deg = pd.read_csv(f).set_index("scenario")
            common = clean.index.intersection(deg.index)
            row = {"base": base, "level": lvl, "n_scenarios": len(common)}
            for mtr in LB_METRICS:
                if mtr not in clean.columns:
                    continue
                c, d = clean.loc[common, mtr].values, deg.loc[common, mtr].values
                diff = d - c
                lo, hi = ci(boot_mean(diff, B, rng))
                row.update({f"{mtr}_clean": c.mean(), f"{mtr}": d.mean(), f"{mtr}_change": diff.mean(),
                            f"{mtr}_change_lo": lo, f"{mtr}_change_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# 5. Every family contrast, NAVSIM v1 against held-out BridgeSim
# ------------------------------------------------------------------------------------------------
def contrasts(S: Scores, B: int, rng) -> pd.DataFrame:
    shared = set(S.df[S.df.evaluator == "bridgesim_ext"].key)
    mats = {e: family_matrix(S, e, shared, False) for e in ("open_l2", "navsim_v1", "navsim_v1_proj", "bridgesim_ext")}
    mats = {e: m for e, m in mats.items() if not m.empty}
    common = None
    for m in mats.values():
        common = m.columns if common is None else common.intersection(m.columns)
    fams = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    rows = []
    idx = rng.integers(0, len(common), size=(B, len(common)))

    def resolved(lo, hi):
        return 1 if lo > 0 else (-1 if hi < 0 else 0)

    for fa, fb in itertools.combinations(fams, 2):
        row = {"contrast": f"{fa} - {fb}", "n_scenes": len(common)}
        per = {}
        for e, m in mats.items():
            v = (m.loc[fa, common] - m.loc[fb, common]).values
            per[e] = v
            lo, hi = ci(v[idx].mean(1))
            row.update({e: v.mean(), f"{e}_lo": lo, f"{e}_hi": hi, f"{e}_sign": resolved(lo, hi)})
        if "navsim_v1" in per and "bridgesim_ext" in per:
            row["spearman_navsim_v1_bridgesim"] = St.spearman(per["navsim_v1"], per["bridgesim_ext"])
            a, b = row["navsim_v1_sign"], row["bridgesim_ext_sign"]
            row["pattern"] = ("reversed" if a * b < 0 else "agree" if a == b != 0 else
                              "neither resolved" if a == b == 0 else
                              "only NAVSIM resolved" if b == 0 else "only closed loop resolved")
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# 6. H1 at fixed scene counts
# ------------------------------------------------------------------------------------------------
def _level_matrix(S: Scores, evaluator: str, base: str, fam: str) -> Optional[np.ndarray]:
    """(4, n) oriented scores of levels 0-3 on the measurement scenes, scenes common to all levels."""
    runs = [S.run(evaluator, base, "none", 0)] + [S.run(evaluator, base, fam, lvl) for lvl in (1, 2, 3)]
    if any(r.empty for r in runs):
        return None
    common = runs[0].index
    for r in runs[1:]:
        common = common.intersection(r.index)
    return np.vstack([r.loc[common].values for r in runs])


def h1(S: Scores, families: Sequence[str], ns: Sequence[int], R: int, rng,
       closed: str = "bridgesim_ext", proxies: Sequence[str] = ("open_l2", "navsim_v1", "navsim_v2")) -> pd.DataFrame:
    """Mean Kendall tau between the true level order and the evaluator's, over R random subsets of n
    scenes (bases averaged), for the closed-loop reference and each proxy; the difference
    closed - proxy uses independent subsets (the evaluators score different scenes), and its
    interval is the 2.5/97.5 percentile of the R differences. Bases: those every compared
    evaluator ran (TF, LTF and DD seed 0; NAVSIM v2 has no DD)."""
    true_order = -np.arange(4.0)                        # level 0 is best
    rows = []
    for fam in families:
        for proxy in proxies:
            bases = [b for b in BASES if _level_matrix(S, closed, b, fam) is not None
                     and _level_matrix(S, proxy, b, fam) is not None]
            if not bases:
                continue
            mats = {e: [_level_matrix(S, e, b, fam) for b in bases] for e in (closed, proxy)}
            for n in ns:
                draws = {}
                for e, ms in mats.items():
                    taus = np.zeros(R)
                    for m in ms:
                        k = min(n, m.shape[1])
                        sub = np.stack([rng.choice(m.shape[1], size=k, replace=False) for _ in range(R)])
                        means = m[:, sub].mean(-1).T          # (R, 4)
                        taus += np.array([St.kendall(true_order, mu) for mu in means])
                    draws[e] = taus / len(ms)
                diff = draws[closed] - draws[proxy]
                lo, hi = ci(diff)
                rows.append({"family": fam, "proxy": proxy, "n": n, "bases": " ".join(bases),
                             "tau_closed": draws[closed].mean(), "tau_proxy": draws[proxy].mean(),
                             "diff": diff.mean(), "diff_lo": lo, "diff_hi": hi,
                             "p_closed_better": float((diff > 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="scores.csv written by python -m wete.collect")
    ap.add_argument("--results", default="results", help="results/ (per-scenario BridgeSim files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--R", type=int, default=1000, help="subsets per scene count for item 6")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(list(argv) if argv is not None else None)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = Scores(pd.read_csv(a.scores))
    print(f"[revision] {len(S.df)} score rows; evaluators {sorted(S.df.evaluator.unique())}; "
          f"{len(S.navhard)} NavHard scenes excluded from Population A measurements")

    pairs = [("navsim_v1", "bridgesim_ext"), ("navsim_v1_proj", "bridgesim_ext"), ("open_l2", "bridgesim_ext"),
             ("open_l2_proj", "bridgesim_ext"), ("navsim_v1", "navsim_v1_proj"), ("bridgesim", "bridgesim_ext")]
    tables = {
        "projection_specificity": projection_specificity(S, a.B, rng),
        "projection_agreement": pd.concat([agreement(S, pairs, a.B, rng),
                                           agreement(S, pairs[:4], a.B, rng, shared_only=True)], ignore_index=True),
        "projection_decomposition": decomposition(S, a.B, rng),
        "head_tail": head_tail(S, a.B, rng),
        "rr10": rr10(S, a.B, rng),
        "late_braking": late_braking(Path(a.results), a.B, rng),
        "contrasts": contrasts(S, a.B, rng),
        "h1": h1(S, ("latency", "lateral_noise"), (25, 50, 100, 200), a.R, rng),
    }
    summary = {}
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
        summary[name] = {"rows": int(len(t))}
        print(f"[revision] {name}: {len(t)} rows")
    (out / "summary.json").write_text(json.dumps({"B": a.B, "R": a.R, "seed": a.seed,
                                                  "navhard_scenes": len(S.navhard), "tables": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
