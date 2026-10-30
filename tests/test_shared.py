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
