"""wete.shared on a synthetic score table with BridgeSim's 'sd_<token>' naming and a partial overlap."""

import numpy as np
import pandas as pd
import pytest

from wete import shared as S

META = {"open_l2": {"higher_is_better": False, "family": "open_loop"},
        "navsim_v1": {"higher_is_better": True, "family": "nonreactive"},
        "bridgesim": {"higher_is_better": True, "family": "closed_loop", "reference": True}}

# family effects: NAVSIM says tf > mlp by a lot, BridgeSim says mlp > tf; both say ltf best
NAV = {"mlp": 0.66, "ltf": 0.84, "tf": 0.84}
BS = {"mlp": 0.42, "ltf": 0.52, "tf": 0.38}


def table(seed=0):
    rng = np.random.default_rng(seed)
    nav_sc = [f"t{i:04d}" for i in range(400)]
    bs_sc = [f"sd_t{i:04d}" for i in range(300, 500)]     # 100 shared with NAVSIM
    rows = []
    for fam in ("mlp", "ltf", "tf"):
        for s in range(3):
            pol = f"{fam}_seed{s}"
            for e, scen, mu, sd in (("navsim_v1", nav_sc, NAV[fam], 0.05), ("bridgesim", bs_sc, BS[fam], 0.05)):
                vals = mu + rng.normal(0, sd, len(scen))
                rows += [{"evaluator": e, "population": "B", "base": pol, "degradation": "none", "level": 0,
                          "policy": pol, "scenario": sc, "score": v} for sc, v in zip(scen, vals)]
            ade = 1.0 - NAV[fam] + rng.normal(0, 0.02, len(nav_sc))
            rows += [{"evaluator": "open_l2", "population": "B", "base": pol, "degradation": "none", "level": 0,
                      "policy": pol, "scenario": sc, "score": v} for sc, v in zip(nav_sc, ade)]
    # Population A: lateral_noise L3 hurts both; tail_lateral hurts NAVSIM only
    for e, scen in (("navsim_v1", nav_sc), ("bridgesim", bs_sc)):
        for deg, lvl, drop in (("none", 0, 0.0), ("lateral_noise", 3, 0.3),
                               ("tail_lateral", 3, 0.2 if e == "navsim_v1" else 0.0)):
            vals = 0.8 - drop + rng.normal(0, 0.01, len(scen))
            rows += [{"evaluator": e, "population": "A", "base": "tf_seed0", "degradation": deg, "level": lvl,
                      "policy": f"tf_seed0:{deg}", "scenario": sc, "score": v} for sc, v in zip(scen, vals)]
    return pd.DataFrame(rows)


def test_normalise_and_family():
    assert S.normalise_scenario("sd_abc") == "abc" and S.normalise_scenario("abc") == "abc"
    assert S.family_of("ego_status_mlp_seed2") == "ego_status_mlp"


def test_shared_set_and_flipped_contrast(tmp_path):
    rep = S.run(table(), META, str(tmp_path), B=200)
    pb = rep["population_b"]
    assert pb["n_shared"] == 100
    assert pb["n_full"] == {"open_l2": 400, "navsim_v1": 400, "bridgesim": 200}
    c = pd.read_csv(tmp_path / "family_contrasts.csv")
    tf_mlp = c[(c.contrast == "mlp - tf") & (c.scope == "shared")].set_index("evaluator")
    assert tf_mlp.loc["navsim_v1", "hi"] < 0          # NAVSIM: tf better than mlp
    assert tf_mlp.loc["bridgesim", "lo"] > 0          # BridgeSim: mlp better than tf
    assert tf_mlp.loc["open_l2", "hi"] < 0            # L2 oriented (lower ADE = better)
    a = pd.read_csv(tmp_path / "shared_agreement.csv").set_index("pair")
    assert a.loc["open_l2|navsim_v1", "rho"] > 0.5
    assert set(a.n_scenarios) == {100}


def test_population_a_on_shared(tmp_path):
    S.run(table(), META, str(tmp_path), B=50)
    pa = pd.read_csv(tmp_path / "popA_shared.csv")
    tail = pa[(pa.degradation == "tail_lateral")].set_index("evaluator")
    assert tail.loc["navsim_v1", "mean_drop_shared"] == pytest.approx(0.2, abs=0.01)
    assert abs(tail.loc["bridgesim", "mean_drop_shared"]) < 0.01
    assert set(pa.n_shared) == {100}


def test_retest_ceiling(tmp_path):
    rng = np.random.default_rng(1)
    sc = [f"sd_{i}" for i in range(200)]
    base = rng.uniform(0, 1, 200)
    signal = {"ltf_seed0": base + 0.1 + rng.normal(0, 0.1, 200), "transfuser_seed0": base}
    for root, noise in (("orig", 0.0), ("rerun", 0.05)):
        for pol, true in signal.items():
            d = tmp_path / root / pol
            d.mkdir(parents=True)
            pd.DataFrame({"scenario": sc, "DS": true + rng.normal(0, noise, 200)}).to_csv(
                d / "per_scenario.csv", index=False)
    rows = pd.DataFrame(S.retest(tmp_path / "orig", tmp_path / "rerun")).set_index("policy")
    assert rows.loc["ltf_seed0", "spearman"] > 0.9
    assert 0.5 < rows.loc["ltf_seed0 - transfuser_seed0", "spearman"] < 1.0


def test_selection_split_is_deterministic_and_shared_across_naming():
    from wete import analysis as A
    toks = pd.Series([f"t{i:04d}" for i in range(2000)])
    sel = A.in_selection_half(toks, 0.5, "salt")
    assert 0.45 < sel.mean() < 0.55
    same = A.in_selection_half("sd_" + toks, 0.5, "salt")
    assert (sel.values == same.values).all()           # 'sd_<token>' lands in the same half


def test_analysis_split_keeps_halves_apart(tmp_path):
    from wete import analysis as A
    t = table()
    rep = A.run(t, META, str(tmp_path), cfg={"B": 50, "split": True})
    sp = rep["split"]["scenarios"]
    assert sp["bridgesim"]["selection"] + sp["bridgesim"]["evaluation"] == 200
    assert rep["monotonicity"]["on"] == "selection half"
    m = pd.read_csv(tmp_path / "discriminative_matrix.csv")
    assert (m.n_scenarios[m.evaluator == "bridgesim"] == sp["bridgesim"]["evaluation"]).all()
