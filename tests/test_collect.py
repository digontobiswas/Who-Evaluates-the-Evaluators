"""Collector on a synthetic runs/ tree with the native output formats, then the analysis on the
collected table (integration of collect + analysis).

NAVSIM v1 runs use the layout and files of wete.cli.navsim_v1: <out_dir>/pdm_score/<ts>.csv with
the v1.1 devkit's final "average" row, and <out_dir>/pdm_score/plans/*.plans.jsonl."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from wete import analysis
from wete import collect as C

TOKS = [f"tok{i:03d}" for i in range(40)]
SCENES = [f"scene-{i:02d}_easy" for i in range(30)]


def _write_tree(root: Path, rng):
    drops = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}
    for base in ("tf", "ltf"):
        clean = "none__L0" if base == "tf" else "clean"   # both spellings of the clean run
        for spec, lvl in [(clean, 0), ("camera_dropout__L1", 1), ("camera_dropout__L2", 2),
                          ("camera_dropout__L3", 3), ("late_braking__L1", 1), ("late_braking__L2", 2),
                          ("late_braking__L3", 3)]:
            # navsim v1.1: per-token CSV with a final "average" row and one invalid token
            d = root / "navsim_v1" / "A" / base / spec
            (d / "pdm_score" / "plans").mkdir(parents=True)
            sc = np.clip(0.85 - drops[lvl] + 0.05 * rng.normal(size=len(TOKS)), 0, 1)
            df = pd.DataFrame({"token": TOKS, "valid": True, "score": sc})
            df.loc[3, "valid"] = False
            df.loc[len(df)] = {"token": "average", "valid": False, "score": sc.mean()}
            df.to_csv(d / "pdm_score" / "2026.10.09.12.00.00.csv")
            (d / "selection.json").write_text("{}")
            with open(d / "pdm_score" / "plans" / f"{base}__{spec}.plans.jsonl", "w") as f:
                for i, t in enumerate(TOKS):
                    # plan changes only on even tokens (degradation has no effect on odd ones)
                    shift = drops[lvl] * 10 * (i % 2 == 0)
                    poses = [[2.0 * k + shift, 0.0] for k in range(1, 9)]
                    f.write(json.dumps({"token": t, "ade": 1.0 + drops[lvl] * 5 + rng.random(),
                                        "poses": poses}) + "\n")
            # navsim v2 one stage: devkit CSV with a final "average_all_frames" row
            d2 = root / "navsim_v2" / "A" / base / spec / "pdm_score"
            d2.mkdir(parents=True)
            s1 = np.clip(0.8 - drops[lvl] + 0.05 * rng.normal(size=len(TOKS)), 0, 1)
            df2 = pd.DataFrame({"token": TOKS, "valid": True, "score": s1, "lane_keeping": 1.0})
            df2.loc[len(df2)] = {"token": "average_all_frames", "valid": True, "score": s1.mean(), "lane_keeping": 1.0}
            df2.to_csv(d2 / "2026.10.31.01.00.00.csv")
            # hugsim
            for scn in SCENES:
                d3 = root / "hugsim" / "A" / base / spec / "ltf" / scn
                d3.mkdir(parents=True)
                (d3 / "eval.json").write_text(json.dumps(
                    {"hdscore": float(np.clip(0.7 - drops[lvl] + 0.05 * rng.normal(), 0, 1)),
                     "nc": 1.0}))
                with open(d3 / "wete_trace.jsonl", "w") as f:
                    for k in range(5):
                        f.write(json.dumps({"step": k, "ego_pos": [k * 2.0, 0.3 * k, 0.0]}) + "\n")


def test_collect_and_analyse(tmp_path):
    rng = np.random.default_rng(0)
    _write_tree(tmp_path / "runs", rng)
    C.LOGGED_PATHS.clear()
    C.LOGGED_PATHS.update({s.rsplit("_", 1)[0]: np.stack([np.arange(10) * 2.0, np.zeros(10)], 1)
                           for s in SCENES})
    scores, feats, report = C.collect(tmp_path / "runs")
    assert set(scores.evaluator) == {"navsim_v1", "open_l2", "navsim_v2", "hugsim"}
    v1 = scores[scores.evaluator == "navsim_v1"]
    assert "average" not in set(v1.scenario) and len(v1) == 14 * len(TOKS)
    assert v1.score.isna().sum() == 14                       # 1 invalid token x 14 policies
    assert report["navsim_v1"]["tf:none__L0"]["n_invalid"] == 1
    assert report["navsim_v1"]["ltf:clean"]["n_invalid"] == 1
    assert set(scores[scores.level == 0].degradation) == {"none"}
    assert {"deviation"} <= set(feats.columns)
    hug = feats[feats.evaluator == "hugsim"].deviation
    assert np.all(hug > 0)

    meta = {"open_l2": {"higher_is_better": False, "family": "open_loop"},
            "navsim_v1": {"higher_is_better": True, "family": "nonreactive"},
            "navsim_v2": {"higher_is_better": True, "family": "pseudo_sim"},
            "hugsim": {"higher_is_better": True, "family": "closed_loop", "reference": True}}
    beh = C.behaviour_table(tmp_path / "runs")
    assert set(beh.degradation) == {"camera_dropout", "late_braking"}
    even = beh[beh.scenario.isin(TOKS[0::2])].delta
    odd = beh[beh.scenario.isin(TOKS[1::2])].delta
    assert np.allclose(even, 3.0) and np.allclose(odd, 0.0)
    rep = analysis.run(scores, meta, str(tmp_path / "res"), feats, {"B": 100}, beh)
    assert set(rep["monotonicity"]["retained"]) == {"camera_dropout", "late_braking"}
    m = pd.read_csv(tmp_path / "res" / "discriminative_matrix.csv")
    assert (m.tau > 0.9).all()                                # all strongly ordered by construction
    assert (tmp_path / "res" / "tab_matrix.tex").exists()


def test_analysis_without_reference_is_explicitly_provisional(tmp_path):
    rng = np.random.default_rng(1)
    _write_tree(tmp_path / "runs", rng)
    scores, _, _ = C.collect(tmp_path / "runs")
    scores = scores[scores.evaluator.isin(["navsim_v1", "open_l2"])]
    meta = json.loads((Path(__file__).resolve().parents[1] / "configs" / "meta.navsim_v1.json").read_text())
    import pytest

    with pytest.raises(ValueError):
        analysis.run(scores, meta, str(tmp_path / "res0"), cfg={"B": 50})
    rep = analysis.run(scores, meta, str(tmp_path / "res"), cfg={"B": 50, "allow_no_reference": True})
    assert rep["monotonicity"]["tested"] is False and rep["monotonicity"]["retained"] == []
    assert "provisional" in rep and rep["hypotheses"] == {}
    m = pd.read_csv(tmp_path / "res" / "discriminative_matrix.csv")
    assert set(m.evaluator) == {"navsim_v1", "open_l2"} and not m.retained.any()


def test_bridgesim_score_column_is_selectable(tmp_path):
    d = tmp_path / "runs" / "bridgesim" / "A" / "ltf_seed0" / "none__L0"
    d.mkdir(parents=True)
    (d / "per_scenario.csv").write_text("scenario,DS,EPDMS_no_ep,RC\nsd_a,0.4,0.8,0.5\n")
    old = C.BRIDGESIM_SCORE
    try:
        C.BRIDGESIM_SCORE = "DS"
        assert C.read_bridgesim(d).score.tolist() == [0.4]
        C.BRIDGESIM_SCORE = "EPDMS_no_ep"
        assert C.read_bridgesim(d).score.tolist() == [0.8]
    finally:
        C.BRIDGESIM_SCORE = old


def test_navsim_v2_two_stage_group_score(tmp_path, monkeypatch):
    """Per real scene: stage-one score x mean stage-two score of its synthetic follow-ups."""
    monkeypatch.setattr(C, "NAVHARD_MAPPING", [["cur1", "prev1", [["s1a", "s1b"], ["s2a", "s2b"]]]])
    d = tmp_path / "pdm_score"
    d.mkdir()
    pd.DataFrame({"token": ["cur1", "prev1", "s1a", "s1b", "s2a", "s2b", "extended_pdm_score_combined"],
                  "valid": [True] * 6 + [True],
                  "score": [0.8, 0.5, 1.0, 0.4, 0.6, 0.8, 0.3]}).to_csv(d / "2026.10.31.csv")
    sc, ft = C.read_navsim_v2_ph(tmp_path)
    got = dict(zip(sc.scenario, sc.score))
    assert got["cur1"] == pytest.approx(0.8 * (1.0 + 0.6) / 2)
    assert got["prev1"] == pytest.approx(0.5 * (0.4 + 0.8) / 2)
    assert set(ft.columns) == {"scenario", "stage_one", "stage_two_mean"}
