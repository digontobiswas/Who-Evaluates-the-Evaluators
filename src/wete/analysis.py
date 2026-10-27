"""End-to-end analysis of a long-format score table.

Input schema (CSV/parquet), one row per (evaluator, policy, scenario):
    evaluator    str   e.g. "navsim_v1", "hugsim"
    population   str   "A" (degraded), "B" (natural), "C" (checkpoints)
    base         str   base policy for A; policy name for B; run name for C
    degradation  str   "none" for level 0 / B / C
    level        int   0..3 for A; epoch index for C; 0 for B
    policy       str   unique policy id
    scenario     str   scenario token / scene id
    score        float raw evaluator score (orientation given in meta)

Evaluator meta (JSON/YAML): name -> {higher_is_better: bool,
    family: open_loop|nonreactive|pseudo_sim|closed_loop|world_model|carla,
    reference: bool (the most trustworthy closed-loop evaluator, E*)}

Optional features table: evaluator, scenario, <feature columns...>.

Usage:
    python -m wete.analysis --scores scores.csv --meta meta.json \
        [--features features.csv] --out results/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import degradations as D
from . import reliability as R
from . import stats as St

DEFAULT_CFG = {
    "B": 1000,                 # bootstrap resamples
    "mono_threshold": 0.95,
    "mono_require_all_refs": True,
    "h_margin": 0.10,          # equivalence margin in tau for 'contradicted'
    "h3_auroc": 0.70,
    "theta": 0.5,              # reliability label threshold
    "behavior_eps": 0.1,       # m; plan change that counts as 'degradation has an effect'
    "n_sub": 25,               # p_full@n
    "eff_grid": [10, 25, 50, 100, 200, 400, 800],
    "seed": 0,
    # True only for a provisional summary when no closed-loop reference has been run yet:
    # monotonicity is then NOT tested, nothing counts as retained, and H1-H4 are skipped.
    "allow_no_reference": False,
    # Specificity arm (benign tail perturbations), fixed before any specificity run (28 Oct 2026):
    "spec_margin": 0.01,       # equivalence margin on the reference score scale (1 DS point)
    "spec_ratio": 0.25,        # a penalty counts if >= this fraction of the harmful twin's drop
    "spec_threshold": 0.95,    # P(drop > 0) needed for a penalty to count
}


# --------------------------------------------------------------------------
# Table helpers
# --------------------------------------------------------------------------

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    need = {"evaluator", "population", "base", "degradation", "level", "policy", "scenario", "score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"score table missing columns {missing}")
    df["scenario"] = df["scenario"].astype(str)
    df["level"] = df["level"].astype(int)
    return df


def load_meta(path: str) -> Dict[str, Dict]:
    txt = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def family_matrix(dfA: pd.DataFrame, evaluator: str, base: str, deg: str,
                  hib: bool) -> Tuple[Optional[np.ndarray], List[str]]:
    """(L, n) oriented scores for one family; level 0 is the base's clean row."""
    e = dfA[(dfA.evaluator == evaluator) & (dfA.base == base)]
    l0 = e[(e.level == 0)]
    ld = e[(e.degradation == deg) & (e.level > 0)]
    if l0.empty or ld.empty:
        return None, []
    sub = pd.concat([l0.assign(level=0), ld])
    piv = sub.pivot_table(index="level", columns="scenario", values="score", aggfunc="mean")
    piv = piv.dropna(axis=1)
    if piv.shape[0] < 2:
        return None, []
    return St.orient(piv.values, hib), list(piv.columns)


def families_for(dfA: pd.DataFrame, evaluator: str, degs: Sequence[str], hib: bool,
                 shared: bool = True) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
    """All families of one evaluator on a SHARED scenario axis (intersection)."""
    bases = sorted(dfA[dfA.evaluator == evaluator].base.unique())
    raw = {}
    for b in bases:
        for d in degs:
            M, sc = family_matrix(dfA, evaluator, b, d, hib)
            if M is not None:
                raw[(b, d)] = (M, sc)
    if not raw:
        return {}, []
    common = sorted(set.intersection(*[set(sc) for _, sc in raw.values()])) if shared else None
    out = {}
    for k, (M, sc) in raw.items():
        pos = {s: i for i, s in enumerate(sc)}
        out[k] = M[:, [pos[s] for s in common]]
    return out, common


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------

def run(scores: pd.DataFrame, meta: Dict[str, Dict], out_dir: str,
        features: Optional[pd.DataFrame] = None, cfg: Optional[Dict] = None,
        behavior: Optional[pd.DataFrame] = None) -> Dict:
    """behavior (optional): columns base, degradation, scenario, delta (m) --
    per-scenario displacement between the level-3 and clean plans from the
    same input. Enables the evaluator-independent 'behavior_conditioned'
    reliability labels."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rng = np.random.default_rng(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: Dict = {"config": cfg}

    dfA = scores[scores.population == "A"]
    evaluators = [e for e in meta if e in set(scores.evaluator)]
    refs = [e for e in evaluators if meta[e].get("reference")]
    if not refs and not cfg["allow_no_reference"]:
        raise ValueError("meta must mark at least one evaluator as reference (E*)")
    if not refs:
        report["provisional"] = ("No reference evaluator: monotonicity was not tested, so no degradation "
                                 "is validated as ground truth. Matrix cells are descriptive only.")
    degs_all = [d for d in D.LEVELS if d in set(dfA.degradation) and d not in D.BENIGN]

    # ---- 1. Monotonicity on reference evaluator(s) -----------------------
    mono_rows = []
    keep = {}
    for d in degs_all:
        ok_all = []
        for ref in refs:
            fams, _ = families_for(dfA, ref, [d], meta[ref]["higher_is_better"])
            for (b, _), S in fams.items():
                m = St.monotonicity_test(S, B=cfg["B"], threshold=cfg["mono_threshold"], rng=rng)
                mono_rows.append({"reference": ref, "base": b, "degradation": d, "passed": m.passed,
                                  **{f"p_{l}{l+1}": p for l, p in enumerate(m.pair_prob)},
                                  **{f"drop_{l}{l+1}": x for l, x in enumerate(m.pair_mean_drop)},
                                  **{f"t_p_{l}{l+1}": x for l, x in enumerate(m.pair_t_p)}})
                ok_all.append(m.passed)
        keep[d] = bool(ok_all) and all(ok_all)
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(out / "monotonicity.csv", index=False)
    retained = [d for d in degs_all if keep[d]]
    report["monotonicity"] = {"tested": bool(refs), "references": refs, "retained": retained,
                              "discarded": [d for d in degs_all if not keep[d]] if refs else []}

    # ---- 2. Discriminative-power matrix ---------------------------------
    cells = []
    draws: Dict[Tuple[str, str], np.ndarray] = {}
    for e in evaluators:
        hib = meta[e]["higher_is_better"]
        for d in degs_all:
            if d in D.CLOSED_LOOP_ONLY and meta[e]["family"] not in ("closed_loop", "world_model", "carla"):
                continue
            fams, sc = families_for(dfA, e, [d], hib)
            if not fams:
                continue
            res = St.ordering_recovery(list(fams.values()), B=cfg["B"], rng=rng)
            draws[(e, d)] = res.tau_draws
            cells.append({"evaluator": e, "degradation": d, "failure_class": D.FAILURE_CLASS[d],
                          "retained": d in retained, "tau": res.tau, "tau_lo": res.tau_lo,
                          "tau_hi": res.tau_hi, "pair_acc": res.pair_acc,
                          "p_full_order": res.p_full_order, "n_scenarios": res.n_scenarios,
                          "n_bases": res.n_bases})
    matrix = pd.DataFrame(cells)
    matrix.to_csv(out / "discriminative_matrix.csv", index=False)

    # ---- 3. Pre-registered hypotheses H1, H2 -----------------------------
    def group_draws(evals: Sequence[str], dgs: Sequence[str]) -> Optional[np.ndarray]:
        arr = [draws[(e, d)] for e in evals for d in dgs if (e, d) in draws]
        return np.mean(arr, axis=0) if arr else None

    fam_of = {e: meta[e]["family"] for e in evaluators}
    openish = [e for e in evaluators if fam_of[e] in ("open_loop", "nonreactive")]
    closed = [e for e in evaluators if fam_of[e] == "closed_loop"]
    timing_mem = [d for d in retained if D.FAILURE_CLASS[d] in ("timing", "memory")
                  and d not in D.CLOSED_LOOP_ONLY]   # must be measurable by both groups
    hyp = {}
    a, b = group_draws(closed, timing_mem), group_draws(openish, timing_mem)
    if a is not None and b is not None:
        hyp["H1"] = {"statement": "closed-loop orders timing/memory degradations better than "
                                  "open-loop + non-reactive evaluators",
                     "groups": {"closed": closed, "open_nonreactive": openish},
                     "degradations": timing_mem,
                     **St.verdict_greater(a - b, cfg["h_margin"])}
    wms = [e for e in evaluators if fam_of[e] == "world_model"]
    perc = [d for d in retained if D.FAILURE_CLASS[d] == "perception"]
    ctrl = [d for d in retained if D.FAILURE_CLASS[d] == "control"]
    a, b = group_draws(wms, perc), group_draws(wms, ctrl)
    if a is not None and b is not None:
        hyp["H2"] = {"statement": "world model orders perception degradations better than control",
                     "degradations": {"perception": perc, "control": ctrl},
                     **St.verdict_greater(a - b, cfg["h_margin"])}

    # ---- 4. Sample efficiency -------------------------------------------
    eff = {}
    eff_degs = retained if refs else degs_all
    for e in evaluators:
        hib = meta[e]["higher_is_better"]
        eff[e] = {}
        for d in eff_degs:
            fams, _ = families_for(dfA, e, [d], hib)
            if not fams:
                continue
            curve = St.sample_efficiency(list(fams.values()), cfg["eff_grid"], R=200, rng=rng)
            eff[e][d] = {"curve": curve, "n_for_p90": St.scenarios_to_stability(curve, 0.9)}
    report["sample_efficiency"] = eff

    # ---- 4b. Specificity arm --------------------------------------------
    spec_rows, spec_summary = specificity(dfA, meta, evaluators, refs, cfg, rng)
    if spec_rows:
        pd.DataFrame(spec_rows).to_csv(out / "specificity.csv", index=False)
    report["specificity"] = spec_summary

    # ---- 5. Agreement on Populations B and C -----------------------------
    agreement = {}
    for pop in ("B", "C"):
        dfp = scores[scores.population == pop]
        if dfp.empty:
            continue
        pols = sorted(set.intersection(*[set(dfp[dfp.evaluator == e].policy) for e in evaluators
                                         if not dfp[dfp.evaluator == e].empty]))
        mats = {}
        for e in evaluators:
            de = dfp[(dfp.evaluator == e) & dfp.policy.isin(pols)]
            if de.empty:
                continue
            piv = de.pivot_table(index="policy", columns="scenario", values="score").loc[pols]
            mats[e] = St.orient(piv.values, meta[e]["higher_is_better"])
        if len(mats) >= 2 and len(pols) >= 3:
            agreement[pop] = {"policies": pols,
                              "pairs": St.bootstrap_agreement(mats, B=cfg["B"], reference=refs[0]
                                                              if refs[0] in mats else None, rng=rng)}
            if pop == "C":
                # checkpoints: approximate known order = training progress (level)
                lvl = dfp.drop_duplicates("policy").set_index("policy").loc[pols, "level"].values
                agreement[pop]["vs_epoch_tau"] = {e: St.kendall(np.nanmean(M, 1), lvl)
                                                  for e, M in mats.items()}
    report["agreement"] = agreement

    # ---- 6. Reliability-weighted scoring (H3, H4) ------------------------
    rel = {}
    if features is not None:
        features = features.copy()
        features["scenario"] = features["scenario"].astype(str)
        for e in evaluators:
            fe = features[features.evaluator == e].drop(columns=["evaluator"]).set_index("scenario")
            fe = fe.dropna(axis=1, how="all")
            if fe.empty or fe.shape[1] == 0:
                continue
            fams, all_common = families_for(dfA, e, retained, meta[e]["higher_is_better"])
            common = [s for s in all_common if s in fe.index]
            if len(common) < 20 or len({b for b, _ in fams}) < 2 or len({d for _, d in fams}) < 2:
                rel[e] = {"skipped": "needs >=2 bases, >=2 retained degradations, >=20 scenarios"}
                continue
            sc_pos = {s: i for i, s in enumerate(all_common)}
            cols = [sc_pos[s] for s in common]
            fams = {k: v[:, cols] for k, v in fams.items()}
            X = fe.loc[common].values.astype(float)
            entry = {"features": list(fe.columns), "n_scenarios": len(common),
                     "label_ceiling": R.label_ceiling_auroc(fams, cfg["theta"], seed=cfg["seed"])}
            beh = None
            if behavior is not None and not behavior.empty:
                bpiv = behavior.assign(scenario=behavior.scenario.astype(str)) \
                    .pivot_table(index=["base", "degradation"], columns="scenario", values="delta")
                beh = {k: (bpiv.loc[k, common].values if k in bpiv.index and set(common) <= set(bpiv.columns)
                           else np.full(len(common), np.nan)) for k in fams}
            for mode in ("absolute", "effect_conditioned", "behavior_conditioned"):
                ref_fams = None
                if mode == "behavior_conditioned" and beh is None:
                    entry[mode] = {"skipped": "no behaviour deltas supplied"}
                    continue
                if mode == "effect_conditioned":
                    ref = refs[0]
                    if ref == e:
                        entry[mode] = {"skipped": "evaluator is the reference (circular)"}
                        continue
                    rf, rcommon = families_for(dfA, ref, retained, meta[ref]["higher_is_better"])
                    if not set(common) <= set(rcommon):
                        entry[mode] = {"skipped": "reference evaluator does not share this scenario set"}
                        continue
                    rpos = {s: i for i, s in enumerate(rcommon)}
                    ref_fams = {k: rf[k][:, [rpos[s] for s in common]] for k in fams if k in rf}
                    if set(ref_fams) != set(fams):
                        entry[mode] = {"skipped": "reference lacks some families"}
                        continue
                for kind in ("logreg", "mlp"):
                    cv = R.heldout_cv(fams, X, theta=cfg["theta"], mode=mode, reference=ref_fams,
                                      behavior=beh if mode == "behavior_conditioned" else None,
                                      eps=cfg["behavior_eps"],
                                      kind=kind, n_sub=cfg["n_sub"], B=min(cfg["B"], 300),
                                      seed=cfg["seed"])
                    s = R.summarize_cv(cv)
                    h3 = {"auroc": s["auroc_mean"], "ci": s["auroc_ci_over_families"]}
                    lo, hi = h3["ci"]
                    h3["verdict"] = ("supported" if lo > cfg["h3_auroc"] else
                                     "contradicted" if hi <= cfg["h3_auroc"] else "inconclusive")
                    h4_tau = St.verdict_greater(s["_gain_tau_draws"], 0.0)
                    h4_z = St.verdict_greater(s["_gain_z_draws"], 0.0)
                    entry.setdefault(mode, {})[kind] = {
                        **{k: v for k, v in s.items() if not k.startswith("_")},
                        "H3": h3, "H4_tau": h4_tau, "H4_sepz": h4_z}
            sub = R.discriminative_subset_cv(fams)
            entry["fallback_subset"] = sub["by_fraction"].reset_index().to_dict(orient="records")
            rel[e] = entry
    report["reliability"] = rel
    report["hypotheses"] = hyp

    # ---- 7. Outputs ------------------------------------------------------
    write_latex(matrix, rel, out, retained)
    try:
        plot_matrix(matrix, out / "discriminative_matrix.png", retained)
        plot_efficiency(eff, out / "sample_efficiency.png")
    except Exception as ex:  # plotting must never kill the analysis
        report["plot_error"] = repr(ex)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    return report


def specificity(dfA: pd.DataFrame, meta: Dict[str, Dict], evaluators: Sequence[str],
                refs: Sequence[str], cfg: Dict, rng: np.random.Generator) -> Tuple[List[Dict], Dict]:
    """Benign perturbations (D.BENIGN): does an evaluator penalise a change of the plan that does
    not change what the car executes?

    Per (evaluator, base, benign perturbation, level), on scenarios shared with the clean run and
    the harmful twin: paired drop = clean - perturbed (oriented, higher = better), bootstrapped
    over scenarios (B draws, same indices for the twin).

    * Reference evaluator: the perturbation is VALIDATED as benign if, for every base and level,
      the 90% bootstrap CI of the mean drop lies inside +-spec_margin (TOST at alpha 0.05).
    * Every evaluator: a level is a FALSE ALARM if P(drop > 0) > spec_threshold and the drop is at
      least spec_ratio of the harmful twin's drop at the same level. Verdict per (evaluator,
      perturbation): 'penalises benign' if any base/level is a false alarm, else 'specific'.
      Only interpreted for perturbations validated on the reference."""
    rows: List[Dict] = []
    benign = [d for d in D.BENIGN if d in set(dfA.degradation)]
    if not benign:
        return rows, {}
    for e in evaluators:
        hib = meta[e]["higher_is_better"]
        for d in benign:
            twin = D.BENIGN[d]
            for b in sorted(dfA[dfA.evaluator == e].base.unique()):
                S, sc = family_matrix(dfA, e, b, d, hib)
                if S is None:
                    continue
                T, tsc = family_matrix(dfA, e, b, twin, hib)
                common = [x for x in sc if x in set(tsc)] if T is not None else list(sc)
                spos = {x: i for i, x in enumerate(sc)}
                S = S[:, [spos[x] for x in common]]
                if T is not None:
                    tpos = {x: i for i, x in enumerate(tsc)}
                    T = T[:, [tpos[x] for x in common]]
                n = S.shape[1]
                idx = rng.integers(0, n, size=(cfg["B"], n))
                for lvl in range(1, S.shape[0]):
                    drop = S[0] - S[lvl]
                    draws = drop[idx].mean(axis=1)
                    lo90, hi90 = np.quantile(draws, [0.05, 0.95])
                    row = {"evaluator": e, "reference": e in refs, "base": b, "perturbation": d,
                           "twin": twin, "level": lvl, "n_scenarios": n, "mean_drop": float(drop.mean()),
                           "ci90_lo": float(lo90), "ci90_hi": float(hi90),
                           "p_drop_pos": float((draws > 0).mean()),
                           "equivalent": bool(-cfg["spec_margin"] < lo90 and hi90 < cfg["spec_margin"]),
                           "twin_mean_drop": np.nan, "ratio": np.nan, "ratio_lo": np.nan, "ratio_hi": np.nan}
                    if T is not None and lvl < T.shape[0]:
                        tdrop = T[0] - T[lvl]
                        tdraws = tdrop[idx].mean(axis=1)
                        row["twin_mean_drop"] = float(tdrop.mean())
                        if tdrop.mean() > 0:
                            with np.errstate(divide="ignore", invalid="ignore"):
                                rdraws = np.where(tdraws > 0, draws / tdraws, np.nan)
                            row["ratio"] = float(drop.mean() / tdrop.mean())
                            if np.isfinite(rdraws).any():
                                row["ratio_lo"], row["ratio_hi"] = (float(v) for v in np.nanquantile(rdraws, [0.025, 0.975]))
                    row["false_alarm"] = bool(row["p_drop_pos"] > cfg["spec_threshold"]
                                              and np.isfinite(row["ratio"]) and row["ratio"] >= cfg["spec_ratio"])
                    rows.append(row)
    df = pd.DataFrame(rows)
    summary: Dict = {"margin": cfg["spec_margin"], "ratio": cfg["spec_ratio"],
                     "threshold": cfg["spec_threshold"], "validated": {}, "verdicts": {}}
    for d in benign:
        r = df[(df.perturbation == d) & df.reference]
        summary["validated"][d] = bool(len(r)) and bool(r.equivalent.all()) if refs else None
    for (e, d), g in df.groupby(["evaluator", "perturbation"]):
        if g.ratio.isna().all():
            v = "no harmful twin to compare"
        else:
            v = "penalises benign" if g.false_alarm.any() else "specific"
        summary["verdicts"].setdefault(e, {})[d] = {
            "verdict": v, "validated_on_reference": summary["validated"].get(d),
            "max_ratio": float(g.ratio.max()) if g.ratio.notna().any() else None}
    return rows, summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


# --------------------------------------------------------------------------
# LaTeX + figures
# --------------------------------------------------------------------------

PRETTY = {"camera_dropout": "Camera dropout", "image_noise": "Image noise",
          "stale_obs": "Stale obs.", "latency": "Latency", "history_removal": "History removal",
          "lateral_noise": "Lateral noise", "late_braking": "Late braking"}


def write_latex(matrix: pd.DataFrame, rel: Dict, out: Path, retained: Sequence[str]) -> None:
    if matrix.empty:
        return
    degs = [d for d in D.LEVELS if d in set(matrix.degradation)]
    evs = list(dict.fromkeys(matrix.evaluator))
    lines = [r"\begin{tabular}{@{}l" + "c" * len(degs) + "@{}}", r"\toprule",
             " & " + " & ".join(PRETTY[d] + ("" if d in retained else r"$^{\ast}$") for d in degs) + r" \\",
             r"\midrule"]
    for e in evs:
        row = [e.replace("_", r"\_")]
        for d in degs:
            c = matrix[(matrix.evaluator == e) & (matrix.degradation == d)]
            row.append("n/a" if c.empty else
                       f"{c.tau.iloc[0]:.2f}\\,{{\\scriptsize[{c.tau_lo.iloc[0]:.2f},{c.tau_hi.iloc[0]:.2f}]}}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $^{\ast}$ degradation failed the monotonicity test on the reference evaluator;"
              r" shown for completeness, excluded from H1/H2 and reliability."]
    (out / "tab_matrix.tex").write_text("\n".join(lines) + "\n")

    rl = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
          r"Evaluator & AUROC of $r_E$ & $\tau$ unif. & $\tau$ weight. & $z$ unif. & $z$ weight. \\",
          r"\midrule"]
    for e, ent in rel.items():
        s = ent.get("absolute", {}).get("logreg") if isinstance(ent, dict) else None
        if not s:
            continue
        lo, hi = s["auroc_ci_over_families"]
        rl.append(f"{e.replace('_', chr(92) + '_')} & {s['auroc_mean']:.2f} [{lo:.2f},{hi:.2f}] & "
                  f"{s['tau_uniform']:.2f} & {s['tau_weighted']:.2f} & {s['z_uniform']:.1f} & {s['z_weighted']:.1f} \\\\")
    rl += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_reliability.tex").write_text("\n".join(rl) + "\n")


def plot_matrix(matrix: pd.DataFrame, path: Path, retained: Sequence[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    degs = [d for d in D.LEVELS if d in set(matrix.degradation)]
    evs = list(dict.fromkeys(matrix.evaluator))
    M = np.full((len(evs), len(degs)), np.nan)
    for i, e in enumerate(evs):
        for j, d in enumerate(degs):
            c = matrix[(matrix.evaluator == e) & (matrix.degradation == d)]
            if not c.empty:
                M[i, j] = c.tau.iloc[0]
    fig, ax = plt.subplots(figsize=(1.3 * len(degs) + 2, 0.6 * len(evs) + 1.5))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(degs)), [PRETTY[d] + ("" if d in retained else "*") for d in degs],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(evs)), evs)
    for i in range(len(evs)):
        for j in range(len(degs)):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Kendall τ vs known order")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_efficiency(eff: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for e, per in eff.items():
        if not per:
            continue
        grid = sorted(set.intersection(*[set(v["curve"]) for v in per.values()]))
        y = [np.mean([per[d]["curve"][n]["p_full_order"] for d in per]) for n in grid]
        ax.plot(grid, y, marker="o", label=e)
    ax.set_xscale("log")
    ax.set_xlabel("scenarios used")
    ax.set_ylabel("P(full order recovered), mean over degradations")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--features")
    ap.add_argument("--behavior", help="CSV base,degradation,scenario,delta (plan displacement, m)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=DEFAULT_CFG["B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-reference", action="store_true",
                    help="provisional run without a reference evaluator (monotonicity untested)")
    a = ap.parse_args(argv)
    feats = pd.read_csv(a.features) if a.features else None
    beh = pd.read_csv(a.behavior) if a.behavior else None
    rep = run(load_table(a.scores), load_meta(a.meta), a.out, feats,
              {"B": a.B, "seed": a.seed, "allow_no_reference": a.no_reference}, beh)
    print(json.dumps({"provisional": rep.get("provisional"), "monotonicity": rep["monotonicity"],
                      "specificity": rep.get("specificity"), "hypotheses": rep["hypotheses"]},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    main()
