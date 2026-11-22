"""wete.cli.revision on synthetic scores with known effects."""

import numpy as np
import pandas as pd
import pytest

from wete.cli import revision as R

BASES = ("transfuser_seed0", "ltf_seed0", "diffusiondrive_seed0")


def _rows(ev, pop, base, deg, lvl, scen, vals, policy=None):
    pol = policy or (base if pop == "B" else (f"{base}:none__L0" if lvl == 0 else f"{base}:{deg}__L{lvl}"))
    return [{"evaluator": ev, "population": pop, "base": base, "degradation": deg, "level": lvl,
             "policy": pol, "scenario": s, "score": v} for s, v in zip(scen, vals)]


def synthetic():
    rng = np.random.default_rng(0)
    navhard = [f"sd_n{i}" for i in range(60)]
    held = [f"sd_h{i}" for i in range(80)]
    navtest = [f"n{i}" for i in range(60)] + [f"h{i}" for i in range(80)] + [f"t{i}" for i in range(160)]
    rows = []
    # Population A. NAVSIM: tail costs 0.6 of lateral noise; projected NAVSIM: tail costs nothing;
    # head costs 0.5. L2 (lower is better): tail 0.3 of lateral. Closed loop: only harmful moves.
    effects = {
        "navsim_v1": {"lateral_noise": 0.10, "tail_lateral": 0.06, "tail_stretch": 0.04, "late_braking": 0.08,
                      "head_lateral": 0.04, "latency": 0.05},
        "navsim_v1_proj": {"lateral_noise": 0.06, "tail_lateral": 0.0, "tail_stretch": 0.0, "late_braking": 0.05},
        "navsim_v2": {"lateral_noise": 0.10, "tail_lateral": 0.06, "head_lateral": 0.04, "latency": 0.05},
        "bridgesim": {"lateral_noise": 0.10, "latency": 0.10, "tail_lateral": 0.0},
        "bridgesim_ext": {"lateral_noise": 0.10, "latency": 0.10, "tail_lateral": 0.0},
    }
    for ev, eff in effects.items():
        scen = navhard if ev == "bridgesim" else held if ev == "bridgesim_ext" else navtest
        for base in BASES:
            rows += _rows(ev, "A", base, "none", 0, scen, 0.8 + rng.normal(0, 0.02, len(scen)))
            for deg, d in eff.items():
                for lvl in (1, 2, 3):
                    rows += _rows(ev, "A", base, deg, lvl, scen, 0.8 - d * lvl + rng.normal(0, 0.02, len(scen)))
    for ev, eff in (("open_l2", {"lateral_noise": 0.5, "tail_lateral": 0.15, "head_lateral": 0.3, "latency": 0.3}),
                    ("open_l2_proj", {"lateral_noise": 0.4, "tail_lateral": 0.0})):
        for base in BASES:
            rows += _rows(ev, "A", base, "none", 0, navtest, 1.0 + rng.normal(0, 0.05, len(navtest)))
            for deg, d in eff.items():
                for lvl in (1, 2, 3):
                    rows += _rows(ev, "A", base, deg, lvl, navtest, 1.0 + d * lvl + rng.normal(0, 0.05, len(navtest)))
    # Population B: NAVSIM ranks dd > ltf > tf > mlp; closed loop ltf > mlp > tf > dd; the projected
    # NAVSIM score is in between.
    fam_mean = {"navsim_v1": {"diffusiondrive": .88, "ltf": .84, "transfuser": .84, "ego_status_mlp": .66},
                "navsim_v1_proj": {"diffusiondrive": .85, "ltf": .86, "transfuser": .83, "ego_status_mlp": .70},
                "open_l2": {"diffusiondrive": .9, "ltf": 1.0, "transfuser": 1.0, "ego_status_mlp": 1.3},
                "open_l2_proj": {"diffusiondrive": .9, "ltf": 1.0, "transfuser": 1.0, "ego_status_mlp": 1.3},
                "bridgesim": {"diffusiondrive": .43, "ltf": .52, "transfuser": .40, "ego_status_mlp": .41},
                "bridgesim_ext": {"diffusiondrive": .50, "ltf": .61, "transfuser": .52, "ego_status_mlp": .55}}
    for ev, fm in fam_mean.items():
        scen = navhard if ev == "bridgesim" else held if ev == "bridgesim_ext" else navtest
        for fam, mu in fm.items():
            for seed in ((0,) if fam == "diffusiondrive" else (0, 1, 2)):
                pol = f"{fam}_seed{seed}"
                rows += _rows(ev, "B", pol, "none", 0, scen, mu + rng.normal(0, 0.03, len(scen)), policy=pol)
    # replanning every 1.0 s: old twins hurt, re-derived twins do not
    for ev, d in (("bridgesim_rr10", 0.03), ("bridgesim_rr10_t20", 0.0)):
        for base in BASES[:2]:
            rows += _rows(ev, "A", base, "none", 0, navhard, 0.5 + rng.normal(0, 0.01, len(navhard)))
            for pert in ("tail_lateral", "tail_stretch"):
                for lvl in (1, 2, 3):
                    rows += _rows(ev, "A", base, pert, lvl, navhard, 0.5 - d * lvl + rng.normal(0, 0.01, len(navhard)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rev")
    scores = tmp / "scores.csv"
    synthetic().to_csv(scores, index=False)
    res = tmp / "results"
    rng = np.random.default_rng(1)
    for base in ("transfuser_seed0", "ltf_seed0"):
        for spec, rc, ttc in (("none__L0", 0.70, 0.90), ("late_braking__L1", 0.74, 0.84),
                              ("late_braking__L2", 0.75, 0.83), ("late_braking__L3", 0.76, 0.82)):
            d = res / "popA_bridgesim" / base / spec
            d.mkdir(parents=True)
            n = 50
            ep = 0.7 * ttc / 0.9 + rng.normal(0, 0.01, n)
            pd.DataFrame({"scenario": [f"sd_n{i}" for i in range(n)], "RC": rc + rng.normal(0, 0.01, n),
                          "TTC": ttc + rng.normal(0, 0.01, n), "EPDMS_no_ep": ep,
                          "DS": ep * rc}).to_csv(d / "per_scenario.csv", index=False)
    R.main(["--scores", str(scores), "--results", str(res), "--out", str(tmp / "o"), "--B", "200", "--R", "100"])
    return tmp / "o"


def test_projection_removes_the_benign_penalty(out):
    t = pd.read_csv(out / "projection_specificity.csv")
    full = t[(t.evaluator == "navsim_v1") & (t.perturbation == "tail_lateral")]
    proj = t[(t.evaluator == "navsim_v1_proj") & (t.perturbation == "tail_lateral")]
    assert full.ratio.min() > 0.5 and proj.ratio.abs().max() < 0.1
    assert (full.n_scenarios == 240).all()                       # NavHard scenes excluded


def test_projection_agreement_and_decomposition(out):
    a = pd.read_csv(out / "projection_agreement.csv").set_index(["pair", "scenes"])
    assert a.loc[("navsim_v1|bridgesim_ext", "all"), "n_policies"] == 10
    assert a.loc[("navsim_v1_proj|bridgesim_ext", "all"), "tau"] > a.loc[("navsim_v1|bridgesim_ext", "all"), "tau"]
    d = pd.read_csv(out / "projection_decomposition.csv")
    row = d[(d.contrast == "diffusiondrive - ltf") & (d.scenes == "shared_heldout")].iloc[0]
    assert row["full"] == pytest.approx(row["projected"] + row["tail_part"])
    assert row.bridgesim < 0 and row.n_scenes == 80


def test_head_tail_and_rr10(out):
    h = pd.read_csv(out / "head_tail.csv")
    v1 = h[h.evaluator == "navsim_v1"]
    assert set(v1.base) == set(BASES)
    assert v1.head_plus_tail_minus_full.abs().max() < 0.01      # synthetic effects add up: 0.4 + 0.6 = 1
    assert v1.tail_share.between(0.5, 0.7).all() and v1.head_share.between(0.3, 0.5).all()
    r = pd.read_csv(out / "rr10.csv")
    assert not r[r.evaluator == "bridgesim_rr10"].equivalent.all()
    assert r[r.evaluator == "bridgesim_rr10_t20"].equivalent.all()


def test_late_braking_contrasts_and_h1(out):
    lb = pd.read_csv(out / "late_braking.csv")
    assert (lb.RC_change > 0).all() and (lb.TTC_change < 0).all()
    c = pd.read_csv(out / "contrasts.csv").set_index("contrast")
    assert c.loc["diffusiondrive - ltf", "pattern"] == "reversed"
    assert c.loc["ego_status_mlp - ltf", "pattern"] == "agree"
    assert len(c) == 6
    h = pd.read_csv(out / "h1.csv")
    assert set(h.proxy) == {"open_l2", "navsim_v1", "navsim_v2"} and set(h.n) == {25, 50, 100, 200}
    assert (h.diff_lo <= h["diff"]).all() and (h["diff"] <= h.diff_hi).all()
