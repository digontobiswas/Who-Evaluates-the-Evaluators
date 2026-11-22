"""Write every number, table and figure of the paper from results/ (never typed by hand).

    python -m wete.cli.paper_numbers [--results results] [--paper paper]

Reads only files written by the cluster jobs (results/analysis/*/report.json and CSVs,
results/tables/*.csv) and writes:

    paper/numbers.tex           \\num{Name} macros used in the text
    paper/tables/*.tex          table bodies (\\input inside the table environments of sec/5_results)
    paper/figs/*.pdf            figures

Main analysis: heldout_matched (degradations and benign perturbations validated on all 421 NavHard
scenes; every evaluator measured on scenes outside NavHard, on the same three bases). NAVSIM v2
comes from heldout_v2_matched. Robustness rows come from final, final_seed0, final_all and
final_epdms. Nothing is recomputed except means of numbers already in those files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MAIN = "heldout_matched"
MAIN_V2 = "heldout_v2_matched"
BASES = ("transfuser_seed0", "ltf_seed0", "diffusiondrive_seed0")
BASE_LABEL = {"transfuser_seed0": "TF", "ltf_seed0": "LTF", "diffusiondrive_seed0": "DD"}
EVAL_LABEL = {"open_l2": "Open-loop L2", "navsim_v1": "NAVSIM v1 (PDMS)", "navsim_v2": "NAVSIM v2 (EPDMS)$^\\dagger$",
              "bridgesim_ext": "BridgeSim (DS)"}
FAMILY_LABEL = {"camera_dropout": "Camera dropout", "image_noise": "Image noise", "stale_obs": "Stale observation",
                "latency": "Latency", "lateral_noise": "Lateral noise", "late_braking": "Late braking"}
POLICY_LABEL = {"transfuser": "TransFuser", "ltf": "LTF", "ego_status_mlp": "Ego MLP", "diffusiondrive": "DiffusionDrive"}
# reference palette slots 1-4 (validated: dataviz validate_palette.js, light mode)
COLORS = {"navsim_v1": "#2a78d6", "navsim_v2": "#eb6834", "open_l2": "#1baf7a", "bridgesim_ext": "#eda100"}
HATCH = {"navsim_v1": "", "navsim_v2": "//", "open_l2": "..", "bridgesim_ext": "xx"}


# ------------------------------------------------------------------------------------------------
def _json(p: Path) -> Dict:
    return json.loads(p.read_text())


def f(x: float, nd: int = 1, signed: bool = False) -> str:
    """Number for LaTeX text: fixed decimals, proper minus sign."""
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def ci(lo: float, hi: float, nd: int = 1, scale: float = 1.0) -> str:
    return f"[{f(lo * scale, nd)}, {f(hi * scale, nd)}]"


class Macros:
    def __init__(self) -> None:
        self.items: Dict[str, str] = {}

    def __setitem__(self, name: str, value) -> None:
        if not name.isalpha():
            raise ValueError(f"macro names must be letters only: {name}")
        self.items[name] = str(value)

    def tex(self) -> str:
        lines = ["% Written by python -m wete.cli.paper_numbers from results/. Do not edit."]
        lines += [f"\\expandafter\\def\\csname wete@{k}\\endcsname{{{v}}}" for k, v in sorted(self.items.items())]
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
def scenario_counts(m: Macros, rep: Dict, results: Path) -> None:
    h = rep["heldout"]
    m["NavhardScenarios"] = h["n_selection_scenarios"]
    m["HeldoutScenarios"] = h["scenarios"]["bridgesim_ext"]
    m["NavtestHeldout"] = f"{h['scenarios']['navsim_v1']:,}".replace(",", "{,}")
    st = pd.read_csv(results / "tables/stage1_navsim_navtest.csv")
    m["NavtestScenarios"] = f"{int(st.num_valid.max()):,}".replace(",", "{,}")
    m["SharedNavhard"] = _json(results / "analysis/final/shared/shared_report.json")["population_b"]["n_shared"]
    m["SharedHeldout"] = _json(results / f"analysis/heldout/shared_ext/shared_report.json")["population_b"]["n_shared"]


def reproduction(m: Macros, results: Path) -> None:
    st = pd.read_csv(results / "tables/stage1_navsim_navtest.csv")
    st["family"] = st.policy.str.replace(r"_seed\d+$", "", regex=True)
    means = st.groupby("family").pdms.mean()
    pub = st.groupby("family").published_pdms.first()
    for fam, key in (("transfuser", "TF"), ("ltf", "LTF"), ("ego_status_mlp", "MLP")):
        m[f"Pdms{key}"], m[f"Pdms{key}Published"] = f(means[fam], 1), f(pub[fam], 1)
    pa = pd.read_csv(results / "tables/popA_navsim_navtest.csv")
    dd = pa[(pa.policy == "diffusiondrive_seed0") & (pa.degradation == "none")].iloc[0]
    m["PdmsDD"] = f(dd.pdms, 2)
    m["PdmsDDPublished"] = f(_json(results / "popA/diffusiondrive_seed0/none__L0/summary.json")["comparison"]["reference_pdms"], 1)
    bs = pd.read_csv(results / "tables/bridgesim_navhard.csv")
    clean = bs[bs.extra_evaluator_args == "none"].drop_duplicates("policy").set_index("policy")
    for pol, key in (("transfuser_seed0", "TF"), ("ltf_seed0", "LTF")):
        m[f"Ds{key}Navhard"] = f(clean.loc[pol, "ds"], 1)
        m[f"Ds{key}Published"] = f(clean.loc[pol, "published_ds"], 1)
    pab = pd.read_csv(results / "tables/popA_bridgesim_navhard.csv")
    ddb = pab[(pab.policy == "diffusiondrive_seed0") & (pab.degradation == "none")].iloc[0]
    m["DsDDNavhard"] = f(ddb.ds, 1)
    m["DsDDPublished"] = f(_json(results / "popA_bridgesim/diffusiondrive_seed0/none__L0/summary.json")["comparison"]["reference"]["ds"], 1)


def monotonicity_table(m: Macros, rep: Dict, results: Path) -> str:
    mono = pd.read_csv(results / f"analysis/{MAIN}/monotonicity.csv")
    retained = rep["monotonicity"]["retained"]
    m["RetainedCount"] = {0: "none", 1: "one", 2: "two", 3: "three"}.get(len(retained), len(retained))
    m["RetainedFamilies"] = " and ".join(FAMILY_LABEL[d].lower() for d in retained)
    m["MonoBases"] = mono[mono.degradation.isin(retained)].base.nunique()
    rows = []
    order = ["latency", "lateral_noise", "camera_dropout", "image_noise", "stale_obs", "late_braking"]
    for d in order:
        g = mono[mono.degradation == d]
        if g.empty:
            continue
        ps = [g[f"p_{a}{a+1}"].min() for a in range(3)]
        mark = "\\checkmark" if d in retained else "--"
        rows.append(f"{FAMILY_LABEL[d]} & {g.base.nunique()} & " + " & ".join(f(p, 3) for p in ps) + f" & {mark} \\\\")
    return "\n".join(rows) + "\n"


def robustness_table(m: Macros, results: Path) -> str:
    def ret(name: str) -> str:
        r = _json(results / f"analysis/{name}/report.json")["monotonicity"]["retained"]
        return ", ".join({"latency": "latency", "lateral_noise": "lateral"}.get(d, d) for d in r) or "none"

    spec = lambda name: _json(results / f"analysis/{name}/report.json")["specificity"]["validated"]  # noqa: E731
    yes = lambda d: "yes" if all(d.values()) else "no"  # noqa: E731
    rows = [
        ("All, 6 bases (main)", "DS", ret(MAIN), yes(spec(MAIN))),
        ("Half, 6 bases", "DS", ret("final"), yes(spec("final"))),
        ("Half, seed-0 bases", "DS", ret("final_seed0"), yes(spec("final_seed0"))),
        ("Half, 6 bases", "EPDMS", ret("final_epdms"), yes(spec("final_epdms"))),
    ]
    mono = pd.read_csv(results / "analysis/final/monotonicity.csv")
    lat = mono[mono.degradation == "latency"]
    worst = lat.loc[lat[["p_01", "p_12", "p_23"]].min(axis=1).idxmin()]
    m["SplitWorstLatencyP"] = f(min(worst.p_01, worst.p_12, worst.p_23), 2)
    m["SplitWorstLatencyBase"] = {"transfuser": "TransFuser", "ltf": "LTF"}[worst.base.rsplit("_seed", 1)[0]] + \
        f" seed~{worst.base.rsplit('_seed', 1)[1]}"
    return "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows) + "\n"


def efficiency_table(m: Macros, rep: Dict, rep2: Dict) -> str:
    rows = []
    for e in ("open_l2", "navsim_v1", "navsim_v2", "bridgesim_ext"):
        r = rep2 if e == "navsim_v2" else rep
        eff = r["sample_efficiency"].get(e, {})
        spec = r["specificity"]["verdicts"].get(e, {}).get("tail_lateral", {})
        cells = []
        for d, key in (("latency", "Latency"), ("lateral_noise", "Lateral")):
            x = eff.get(d)
            if not x:
                cells += ["--", "--"]
                continue
            n90, p100 = x["n_for_p90"], x["curve"]["100"]["p_full_order"]
            cells += [str(n90), f(p100, 2)]
            m[f"Eff{key}{_key(e)}"] = n90
            m[f"Pfull{key}{_key(e)}"] = f(p100, 2)
        ratio = spec.get("max_ratio")
        verdict = {"penalises benign": "penalises", "specific": "specific"}.get(spec.get("verdict"), "--")
        m[f"SpecRatio{_key(e)}"] = f(ratio, 2) if ratio is not None else "--"
        m[f"SpecPct{_key(e)}"] = f"{round(100 * ratio)}\\%" if ratio is not None else "--"
        rows.append(f"{EVAL_LABEL[e]} & " + " & ".join(cells) + f" & {f(ratio, 2) if ratio is not None else '--'} & {verdict} \\\\")
    for e in ("open_l2", "navsim_v1"):
        st = rep["specificity"]["verdicts"].get(e, {}).get("tail_stretch", {})
        if st.get("max_ratio") is not None:
            m[f"StretchPct{_key(e)}"] = f"{round(100 * st['max_ratio'])}\\%"
    h1 = rep["hypotheses"].get("H1", {})
    m["HOneVerdict"] = h1.get("verdict", "--")
    return "\n".join(rows) + "\n"


def _key(e: str) -> str:
    return {"open_l2": "Ltwo", "navsim_v1": "NavsimVone", "navsim_v2": "NavsimVtwo", "bridgesim_ext": "Bridgesim"}[e]


def specificity_table(m: Macros, results: Path) -> str:
    s1 = pd.read_csv(results / f"analysis/{MAIN}/specificity.csv")
    s2 = pd.read_csv(results / f"analysis/{MAIN_V2}/specificity.csv")
    s = pd.concat([s1, s2[s2.evaluator == "navsim_v2"]])
    s = s[(s.perturbation == "tail_lateral") & (s.level == 3) & s.base.isin(BASES)]
    rows = []
    unit = {"open_l2": (1.0, 2, "L2 (m)"), "navsim_v1": (100.0, 1, "NAVSIM v1"), "navsim_v2": (100.0, 1, "NAVSIM v2$^\\dagger$"),
            "bridgesim_ext": (100.0, 1, "BridgeSim")}
    short = {"open_l2": "Open-loop L2", "navsim_v1": "NAVSIM v1", "navsim_v2": "NAVSIM v2$^\\dagger$",
             "bridgesim_ext": "BridgeSim"}
    for e in ("open_l2", "navsim_v1", "navsim_v2", "bridgesim_ext"):
        g = s[s.evaluator == e].set_index("base")
        sc, nd, u = unit[e]
        benign = [f(g.loc[b, "mean_drop"] * sc, nd) if b in g.index else "--" for b in BASES]
        harm = [f(g.loc[b, "twin_mean_drop"] * sc, nd) if b in g.index else "--" for b in BASES]
        rows.append(f"{u} & " + " & ".join(benign) + " & " + " & ".join(harm) + " \\\\")
        if e in ("navsim_v1", "bridgesim_ext"):
            vals = (g.mean_drop * sc).values
            m[f"TailDropMax{_key(e)}"] = f(np.max(np.abs(vals)) if e == "bridgesim_ext" else np.max(vals), 1)
            m[f"TailDropMin{_key(e)}"] = f(np.min(vals), 1)
    return "\n".join(rows) + "\n"


def specificity_figure(results: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1 = pd.read_csv(results / f"analysis/{MAIN}/specificity.csv")
    s2 = pd.read_csv(results / f"analysis/{MAIN_V2}/specificity.csv")
    s = pd.concat([s1, s2[s2.evaluator == "navsim_v2"]])
    s = s[(s.perturbation == "tail_lateral") & s.base.isin(BASES)]
    evals = ["navsim_v1", "navsim_v2", "open_l2", "bridgesim_ext"]
    names = {"navsim_v1": "NAVSIM v1", "navsim_v2": "NAVSIM v2", "open_l2": "Open-loop L2", "bridgesim_ext": "BridgeSim"}
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    w = 0.19
    for i, e in enumerate(evals):
        g = s[s.evaluator == e]
        xs, mean, lo, hi = [], [], [], []
        for lvl in (1, 2, 3):
            r = g[g.level == lvl].ratio.dropna()
            if r.empty:
                continue
            xs.append(lvl + (i - 1.5) * w)
            mean.append(r.mean()); lo.append(r.min()); hi.append(r.max())
        mean, lo, hi = map(np.array, (mean, lo, hi))
        ax.bar(xs, mean, width=w - 0.02, color=COLORS[e], hatch=HATCH[e], edgecolor="white", linewidth=0.4,
               label=names[e], zorder=2)
        ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="#52514e", elinewidth=0.6,
                    capsize=1.5, zorder=3)
    ax.axhline(0.25, color="#52514e", lw=0.6, ls="--", zorder=1, label="25% threshold")
    ax.axhline(0, color="#52514e", lw=0.5)
    ax.set_xticks([1, 2, 3], ["L1 (0.3 m)", "L2 (0.8 m)", "L3 (1.5 m)"])
    ax.set_ylabel("benign drop / harmful drop")
    ax.set_ylim(-0.12, 0.92)
    ax.grid(axis="y", color="#e5e4e0", lw=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.7)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def efficiency_figure(rep: Dict, rep2: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"navsim_v1": "NAVSIM v1", "navsim_v2": "NAVSIM v2$^\\dagger$", "open_l2": "Open-loop L2", "bridgesim_ext": "BridgeSim"}
    markers = {"navsim_v1": "o", "navsim_v2": "s", "open_l2": "^", "bridgesim_ext": "D"}
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
    for ax, d, title in zip(axes, ("latency", "lateral_noise"), ("Latency", "Lateral noise")):
        for e in ("navsim_v1", "navsim_v2", "open_l2", "bridgesim_ext"):
            r = rep2 if e == "navsim_v2" else rep
            x = r["sample_efficiency"].get(e, {}).get(d)
            if not x:
                continue
            ns = sorted(int(k) for k in x["curve"])
            ax.plot(ns, [x["curve"][str(n)]["p_full_order"] for n in ns], marker=markers[e], ms=3, lw=1.2,
                    color=COLORS[e], label=names[e])
        ax.axhline(0.9, color="#52514e", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 800], ["10", "100", "800"])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("scenarios")
        ax.grid(color="#e5e4e0", lw=0.4)
    axes[0].set_ylabel("P(full order)")
    axes[1].legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def agreement(m: Macros, rep: Dict, rep2: Dict, results: Path) -> None:
    pairs, pairs2 = rep["agreement"]["B"]["pairs"], rep2["agreement"]["B"]["pairs"]

    def put(key: str, p: Dict) -> None:
        m[f"Tau{key}"] = f(p["tau"], 2)
        m[f"Tau{key}Ci"] = ci(*p["tau_ci"], nd=2)
        m[f"Tau{key}N"] = p["n_policies"]

    put("NavsimBridgesim", pairs["navsim_v1|bridgesim_ext"])
    put("LtwoBridgesim", pairs["open_l2|bridgesim_ext"])
    put("LtwoNavsim", pairs["open_l2|navsim_v1"])
    put("NavhardHeldout", pairs["bridgesim|bridgesim_ext"])
    put("NavsimNavhard", pairs["navsim_v1|bridgesim"])
    put("NavsimVtwoBridgesim", pairs2["navsim_v2|bridgesim_ext"])
    rt = pd.read_csv(results / "analysis/final/shared/retest.csv")
    sc = rt[rt.what == "score"].spearman
    m["RetestMin"], m["RetestMax"] = f(sc.min(), 2), f(sc.max(), 2)
    m["RetestContrast"] = f(rt[rt.what == "contrast"].spearman.iloc[0], 2)
    cc = pd.read_csv(results / "analysis/heldout/shared_ext/contrast_correlation.csv")
    cc = cc[cc.pair == "navsim_v1|bridgesim_ext"].spearman
    m["ContrastCorrMin"], m["ContrastCorrMax"] = f(cc.min(), 2), f(cc.max(), 2)


def contrasts_table(m: Macros, results: Path) -> str:
    c = pd.read_csv(results / "analysis/heldout/shared_ext/family_contrasts.csv")
    c = c[c.scope == "shared"]
    want = [("diffusiondrive - ltf", "DD $-$ LTF", "DDLTF"),
            ("diffusiondrive - ego_status_mlp", "DD $-$ MLP", "DDMLP"),
            ("ego_status_mlp - transfuser", "MLP $-$ TF", "MLPTF"),
            ("ltf - transfuser", "LTF $-$ TF", "LTFTF")]
    rows = []
    for key, label, mk in want:
        g = c[c.contrast == key].set_index("evaluator")
        cells = []
        for e, sc in (("open_l2", 1.0), ("navsim_v1", 100.0), ("bridgesim_ext", 100.0)):
            r = g.loc[e]
            nd = 2 if e == "open_l2" else 1
            # L2 is shown as the point estimate only (its sign is the point); the score columns carry intervals
            cells.append(f(r["mean"] * sc, nd, signed=True) if e == "open_l2"
                         else f"{f(r['mean'] * sc, nd, signed=True)} {ci(r.lo, r.hi, nd, sc)}")
            if e != "open_l2":
                m[f"Contrast{mk}{_key(e)}"] = f(r["mean"] * sc, 1, signed=True)
                m[f"Contrast{mk}{_key(e)}Ci"] = ci(r.lo, r.hi, 1, sc)
                m[f"Contrast{mk}{_key(e)}Abs"] = f(abs(r["mean"]) * sc, 1)
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def popb_table(m: Macros, results: Path) -> str:
    t = results / "tables"
    v1 = pd.read_csv(t / "popB_navsim_navtest.csv").set_index("policy").pdms
    v2 = pd.read_csv(t / "popB_navsim_v2_navtest.csv").set_index("policy").pdms
    nh = pd.read_csv(t / "popB_bridgesim_navhard.csv").set_index("policy").ds
    ho = pd.read_csv(t / "popB_bridgesim_heldout.csv").set_index("policy").ds
    rows = []
    for fam in ("ego_status_mlp", "transfuser", "ltf", "diffusiondrive"):
        pols = sorted(p for p in v1.index if p.startswith(fam + "_seed"))
        for p in pols:
            seed = p.rsplit("_seed", 1)[1]
            label = f"{POLICY_LABEL[fam]}" + (f" s{seed}" if fam != "diffusiondrive" else "")
            cell = lambda s: f(s[p], 1) if p in s.index else "--"  # noqa: E731
            rows.append(f"{label} & {cell(v1)} & {cell(v2)} & {cell(nh)} & {cell(ho)} \\\\")
        if fam != "diffusiondrive":
            rows.append("\\addlinespace[1pt]")
    for fam, key in (("ego_status_mlp", "MLP"), ("transfuser", "TF"), ("ltf", "LTF"), ("diffusiondrive", "DD")):
        sel = [p for p in ho.index if p.startswith(fam + "_seed")]
        m[f"DsHeldout{key}"] = f(ho[sel].mean(), 1)
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args(argv)
    results, paper = Path(a.results), Path(a.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figs").mkdir(parents=True, exist_ok=True)

    rep = _json(results / f"analysis/{MAIN}/report.json")
    rep2 = _json(results / f"analysis/{MAIN_V2}/report.json")
    m = Macros()
    scenario_counts(m, rep, results)
    reproduction(m, results)
    tables = {
        "monotonicity": monotonicity_table(m, rep, results),
        "robustness": robustness_table(m, results),
        "efficiency": efficiency_table(m, rep, rep2),
        "specificity": specificity_table(m, results),
        "contrasts": contrasts_table(m, results),
        "popb": popb_table(m, results),
    }
    agreement(m, rep, rep2, results)
    for name, body in tables.items():
        (paper / "tables" / f"{name}.tex").write_text(
            "% Written by python -m wete.cli.paper_numbers from results/. Do not edit.\n" + body)
    (paper / "numbers.tex").write_text(m.tex())
    specificity_figure(results, paper / "figs/specificity.pdf")
    efficiency_figure(rep, rep2, paper / "figs/efficiency.pdf")
    print(f"[paper] {len(m.items)} numbers, {len(tables)} tables, 2 figures -> {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
