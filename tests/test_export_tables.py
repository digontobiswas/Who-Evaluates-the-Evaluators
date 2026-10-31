"""wete.cli.export_tables on summary.json files shaped like the cluster's."""

import json

import pandas as pd

from wete.cli import export_tables as X


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_tables_copy_values_from_summaries(tmp_path):
    res = tmp_path / "results"
    _write(res / "stage1_pdms_transfuser_seed0_navtest" / "summary.json", {
        "experiment": "stage1_pdms_transfuser_seed0_navtest", "policy": "transfuser_seed0", "split": "navtest",
        "result": {"pdms": 83.39, "score_sem": 0.00265, "num_valid": 12146, "num_expected": 12146,
                   "complete": True, "mean": {"score": 0.8339}},
        "comparison": {"reference_pdms": 84.0, "delta_pdms": -0.61, "reference_source": "paper"},
        "checkpoint": {"sha256": "abc"}})
    _write(res / "stage1_pdms_transfuser_seed0_smoke" / "summary.json", {"experiment": "smoke"})  # ignored
    _write(res / "popA" / "ltf_seed0" / "none__L0" / "summary.json", {
        "policy": "ltf_seed0", "degradation": {"name": "none", "level": 0, "value": None},
        "result": {"pdms": 84.14, "score_sem": 0.0026, "num_valid": 12146, "num_expected": 12146,
                   "complete": True, "mean": {}},
        "open_loop": {"mean_ade": 0.669, "effective_k": None}})
    _write(res / "popA" / "ltf_seed0" / "camera_dropout__L1" / "summary.json", {
        "policy": "ltf_seed0", "degradation": {"name": "camera_dropout", "level": 1, "value": None},
        "result": {"pdms": 47.23, "score_sem": 0.0041, "num_valid": 12146, "num_expected": 12146,
                   "complete": True, "mean": {}},
        "open_loop": {"mean_ade": 1.425, "effective_k": [1]}})
    _write(res / "stage1_bridgesim_navhard_ltf_seed0" / "summary.json", {
        "experiment": "stage1_bridgesim_navhard_ltf_seed0", "policy": "ltf_seed0", "extra_evaluator_args": [],
        "result": {"epdms": 67.72, "ds": 51.94, "rc": 76.14, "num_scored": 421, "num_expected": 421,
                   "complete": True, "mean": {"NC": 0.97}},
        "comparison": {"reference": {"epdms": 69.29, "ds": 50.1, "rc": 70.64}, "delta_epdms": -1.57}})
    X.main(["--results", str(res), "--out", str(tmp_path / "tables")])
    s1 = pd.read_csv(tmp_path / "tables" / "stage1_navsim_navtest.csv")
    assert list(s1.policy) == ["transfuser_seed0"] and s1.pdms[0] == 83.39 and round(s1.pdms_sem[0], 3) == 0.265
    pa = pd.read_csv(tmp_path / "tables" / "popA_navsim_navtest.csv")
    assert list(pa.degradation) == ["none", "camera_dropout"] and pa.cameras_blanked[1] == 1
    bs = pd.read_csv(tmp_path / "tables" / "bridgesim_navhard.csv")
    assert bs.epdms[0] == 67.72 and bs.published_epdms[0] == 69.29 and bs.extra_evaluator_args[0] == "none"
    assert (tmp_path / "tables" / "README.txt").is_file()


def test_population_a_bridgesim_table(tmp_path):
    res = tmp_path / "results"
    for spec, deg, level, epdms in (("none__L0", None, None, 63.3), ("latency__L2", "latency", 2, 41.0)):
        _write(res / "popA_bridgesim" / "transfuser_seed0" / spec / "summary.json", {
            "policy": "transfuser_seed0",
            "degradation": {"name": deg, "level": level, "value": None, "wrap": False},
            "result": {"epdms": epdms, "epdms_sem": 1.0, "ds": 40.0, "ds_sem": 1.1, "rc": 63.8,
                       "num_scored": 421, "num_expected": 421, "complete": True, "mean": {}}})
    X.main(["--results", str(res), "--out", str(tmp_path / "tables")])
    t = pd.read_csv(tmp_path / "tables" / "popA_bridgesim_navhard.csv")
    assert list(t.degradation) == ["none", "latency"] and list(t.level) == [0, 2]


def test_population_b_tables_include_reused_seed0(tmp_path):
    res = tmp_path / "results"
    nav = {"result": {"pdms": 83.0, "score_sem": 0.003, "num_valid": 12146, "num_expected": 12146,
                      "complete": True, "mean": {}}, "open_loop": {"mean_ade": 0.7, "mean_l2_1s": 0.12}}
    _write(res / "popA" / "ltf_seed0" / "none__L0" / "summary.json", {"policy": "ltf_seed0", "degradation": {"name": "none", "level": 0}, **nav})
    _write(res / "popB" / "ego_status_mlp_seed1" / "summary.json", {"policy": "ego_status_mlp_seed1", **nav})
    bs = {"result": {"epdms": 60.0, "ds": 40.0, "rc": 66.0, "num_scored": 421, "num_expected": 421,
                     "complete": True, "mean": {}}}
    _write(res / "popA_bridgesim" / "transfuser_seed0" / "none__L0" / "summary.json", {"policy": "transfuser_seed0", "degradation": {"name": "none", "level": 0}, **bs})
    _write(res / "popB_bridgesim" / "ltf_seed2" / "summary.json", {"policy": "ltf_seed2", **bs})
    X.main(["--results", str(res), "--out", str(tmp_path / "tables")])
    nb = pd.read_csv(tmp_path / "tables" / "popB_navsim_navtest.csv")
    bb = pd.read_csv(tmp_path / "tables" / "popB_bridgesim_navhard.csv")
    assert nb.policy.tolist() == ["ego_status_mlp_seed1", "ltf_seed0"]
    assert bb.policy.tolist() == ["ltf_seed2", "transfuser_seed0"]


def test_navsim_v2_tables(tmp_path):
    res = tmp_path / "results"
    r1 = {"pdms": 70.0, "epdms": 70.0, "score_sem": 0.003, "num_valid": 100, "num_expected": 100, "complete": True,
          "mean": {}}
    r2 = {**r1, "two_stage": True, "epdms_stage_one": 80.0, "epdms_stage_two": 60.0, "num_expected": None}
    deg = {"name": "lateral_noise", "level": 1, "value": None}
    _write(res / "popA_navtest_v2" / "ltf_seed0" / "lateral_noise__L1" / "summary.json",
           {"policy": "ltf_seed0", "degradation": deg, "result": r1, "open_loop": {}})
    _write(res / "popA_navhard_v2" / "ltf_seed0" / "lateral_noise__L1" / "summary.json",
           {"policy": "ltf_seed0", "degradation": deg, "result": r2, "open_loop": {}})
    _write(res / "popB_navhard_v2" / "ego_status_mlp_seed1" / "summary.json",
           {"policy": "ego_status_mlp_seed1", "result": r2, "open_loop": {}})
    X.main(["--results", str(res), "--out", str(tmp_path / "tables")])
    a = pd.read_csv(tmp_path / "tables" / "popA_navsim_v2_navhard.csv")
    assert a.epdms_stage_two.tolist() == [60.0] and a.pdms.tolist() == [70.0]
    assert pd.read_csv(tmp_path / "tables" / "popA_navsim_v2_navtest.csv").policy.tolist() == ["ltf_seed0"]
    assert pd.read_csv(tmp_path / "tables" / "popB_navsim_v2_navhard.csv").policy.tolist() == ["ego_status_mlp_seed1"]
