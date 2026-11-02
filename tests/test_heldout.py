"""Held-out closed-loop scenes: candidate selection, the convert step, configs and --heldout analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wete import config as config_lib
from wete.closed_loop import heldout as H

ROOT = Path(__file__).resolve().parents[1]
ENV = {"WETE_REPO": "/r", "WETE_ROOT": "/w", "WETE_DATA_ROOT": "/d", "WETE_EXP_ROOT": "/x",
       "WETE_CKPT_ROOT": "/ck", "WETE_THIRD_PARTY": "/tp", "OPENSCENE_DATA_ROOT": "/os",
       "NUPLAN_MAPS_ROOT": "/m"}


def test_rank_is_deterministic_and_salted():
    toks = [f"{i:016x}" for i in range(200)]
    assert H.rank(toks) == H.rank(list(reversed(toks))) == H.rank(["sd_" + t for t in toks])
    assert H.rank(toks) != H.rank(toks, salt="other")


def test_build_filter_excludes_navhard_and_keeps_settings():
    navtest = {"num_history_frames": 4, "num_future_frames": 10, "has_route": True,
               "log_names": ["l1"], "tokens": [f"t{i}" for i in range(50)]}
    flt = H.build_filter(navtest, ["t3", "sd_t4"], n_candidates=20)
    assert len(flt["tokens"]) == 20 and not {"t3", "t4"} & set(flt["tokens"])
    assert flt["num_future_frames"] == 10 and flt["log_names"] == ["l1"]
    assert flt["tokens"] == H.rank([t for t in navtest["tokens"] if t not in ("t3", "t4")])[:20]


def test_select_takes_converted_candidates_in_rank_order():
    cands = ["c", "a", "d", "b"]
    assert H.select(cands, ["sd_a", "sd_b", "sd_d"], count=2) == ["sd_a", "sd_d"]
    assert H.select(cands, ["sd_b"], count=5) == ["sd_b"]          # fewer converted: all of them


def test_committed_candidate_files_agree():
    lst = (ROOT / "configs/bridgesim/navtest_heldout_candidates.txt").read_text().split()
    import yaml
    flt = yaml.safe_load((ROOT / "configs/bridgesim/navtest_heldout_filter.yaml").read_text())
    assert flt["tokens"] == lst and len(lst) == H.N_CANDIDATES == len(set(lst))
    assert lst == H.rank(lst)                                       # stored in rank order


def test_convert_writes_the_heldout_list(tmp_path, monkeypatch):
    from wete.cli import bridgesim_run as BR
    (tmp_path / "cands.txt").write_text("c\na\nd\nb\n")
    scen = tmp_path / "scen"

    def fake_run(cmd, **kwargs):                  # the converter: converts a, b, d (not c)
        for t in ("a", "b", "d"):
            p = scen / f"sd_{t}" / f"sd_{t}_0"
            p.mkdir(parents=True)
            (p / f"sd_{t}.pkl").write_bytes(b"x")

        class P:
            returncode = 0
        return P()

    monkeypatch.setattr(BR.subprocess, "run", fake_run)
    cfg = {"evaluator": {"root": str(tmp_path), "scene_filter": "f.yaml", "navsim_log_path": "/l",
                         "map_root": "/m", "conversion": {"num_future_frames_extract": 40, "interpolate": True}},
           "scenarios": {"dir": str(scen), "candidates": str(tmp_path / "cands.txt"), "count": 2,
                         "expected_list": str(tmp_path / "list.txt")}}
    out = BR.action_convert(cfg)
    assert out["num_heldout"] == 2
    assert BR.expected_scenarios(cfg) == ["sd_a", "sd_d"]
    assert BR.action_convert(cfg) == {"skipped": True}             # list exists: not converted again


@pytest.mark.parametrize("name,ov,out", [
    ("stage1_bridgesim_heldout", ["policy.family=ltf"], "runs/bridgesim_ext/A/ltf_seed0/none__L0"),
    ("popA_bridgesim_heldout", ["policy.family=transfuser", "degradation.name=latency", "degradation.level=2"],
     "runs/bridgesim_ext/A/transfuser_seed0/latency__L2"),
    ("popA_bridgesim_heldout_diffusiondrive", ["degradation.name=tail_lateral", "degradation.level=3"],
     "runs/bridgesim_ext/A/diffusiondrive_seed0/tail_lateral__L3"),
    ("popB_bridgesim_heldout", ["policy.family=ego_status_mlp", "policy.seed=2"],
     "runs/bridgesim_ext/B/ego_status_mlp_seed2"),
])
def test_heldout_configs_resolve(monkeypatch, name, ov, out):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = config_lib.resolve(config_lib.load_config(ROOT / "configs" / "experiments" / f"{name}.yaml", ov))
    assert cfg["experiment"]["out_dir"] == f"/x/{out}"
    assert cfg["evaluator"]["scene_filter"] == "/r/configs/bridgesim/navtest_heldout_filter.yaml"
    assert cfg["scenarios"]["dir"] == "/x/bridgesim/scenarios/navtest_heldout"
    assert cfg["scenarios"]["count"] == 420 and cfg["compare_with_reference"] is False
    assert cfg["evaluator"]["run"]["replan_rate"] == 5                         # NavHard protocol unchanged
    if "diffusiondrive" in name:
        assert cfg["policy"]["bridgesim_args"] == ["--plan-anchor-path", "/ck/diffusiondrive/kmeans_navsim_traj_20.npy"]


def _scores():
    """bridgesim (selection) on NavHard scenes n0..n199; bridgesim_ext on h0..h199; navsim_v1 on
    both. latency is monotone everywhere. On held-out scenes navsim_v1 penalises tail_lateral."""
    rng = np.random.default_rng(0)
    rows = []
    sets = {"bridgesim": [f"sd_n{i}" for i in range(200)], "bridgesim_ext": [f"sd_h{i}" for i in range(200)],
            "navsim_v1": [f"n{i}" for i in range(200)] + [f"h{i}" for i in range(200)]}
    for e, sc in sets.items():
        for base in ("transfuser_seed0", "ltf_seed0"):
            for deg, lvl, drop in [("none", 0, 0)] + [("latency", l, 0.1 * l) for l in (1, 2, 3)] + \
                                  [("tail_lateral", l, (0.05 * l if e == "navsim_v1" else 0.0)) for l in (1, 2, 3)] + \
                                  [("lateral_noise", l, 0.08 * l) for l in (1, 2, 3)]:
                v = 0.8 - drop + rng.normal(0, 0.02, len(sc))
                rows += [{"evaluator": e, "population": "A", "base": base, "degradation": deg, "level": lvl,
                          "policy": f"{base}:{deg}", "scenario": s, "score": x} for s, x in zip(sc, v)]
    return pd.DataFrame(rows)


def test_analysis_heldout_validates_on_selection_and_measures_elsewhere(tmp_path):
    from wete import analysis as A
    meta = {"navsim_v1": {"higher_is_better": True, "family": "nonreactive"},
            "bridgesim": {"higher_is_better": True, "family": "closed_loop", "reference": True, "role": "selection"},
            "bridgesim_ext": {"higher_is_better": True, "family": "closed_loop", "reference": True}}
    rep = A.run(_scores(), meta, str(tmp_path), cfg={"B": 50, "heldout": True})
    assert set(rep["monotonicity"]["retained"]) == {"latency", "lateral_noise"}
    assert set(pd.read_csv(tmp_path / "monotonicity.csv").reference) == {"bridgesim"}
    assert rep["heldout"]["scenarios"] == {"bridgesim_ext": 200, "navsim_v1": 200}   # NavHard scenes dropped
    m = pd.read_csv(tmp_path / "discriminative_matrix.csv")
    assert "bridgesim" not in set(m.evaluator) and (m.n_scenarios == 200).all()
    spec = rep["specificity"]
    assert spec["validated"]["tail_lateral"] is True and spec["validated_on"] == "selection evaluator"
    assert spec["verdicts"]["navsim_v1"]["tail_lateral"]["verdict"] == "penalises benign"
    with pytest.raises(ValueError):
        A.run(_scores(), meta, str(tmp_path / "x"), cfg={"B": 50, "heldout": True, "split": True})


def test_excluded_scene_is_dropped_for_every_run(tmp_path, monkeypatch):
    from wete.cli import bridgesim_run as BR
    (tmp_path / "list.txt").write_text("sd_a\nsd_b\nsd_c\n")
    cfg = {"scenarios": {"expected_list": str(tmp_path / "list.txt"), "exclude": ["sd_b"]}}
    assert BR.expected_scenarios(cfg) == ["sd_a", "sd_c"]
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    real = config_lib.resolve(config_lib.load_config(ROOT / "configs/experiments/popB_bridgesim_heldout.yaml"))
    assert real["scenarios"]["exclude"] == ["sd_bb870b710b7a53e8"]
