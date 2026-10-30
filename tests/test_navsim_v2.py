"""NAVSIM v2 pieces: devkit command lines, the EPDMS report, and the experiment configs."""

import os
from pathlib import Path

import pandas as pd
import pytest

from wete import config as config_lib
from wete.navsim import devkit_v1 as K
from wete.navsim import report as R

WORKER = {"name": "single_machine_thread_pool", "use_process_pool": False}
ROOT = Path(__file__).resolve().parents[1]


def _score_cmd(two_stage):
    return K.pdm_score_command_v2(
        devkit_root=Path("/dk"), split="navhard_two_stage" if two_stage else "navtest",
        navsim_log_path=Path("/logs"), sensor_blobs_path=Path("/sensors"), cache_dir=Path("/cache"),
        output_dir=Path("/out"), experiment_name="x", agent="degraded_transfuser_agent",
        agent_overrides=["agent.base_agent.config.latent=true"], checkpoint=Path("/ck.ckpt"),
        worker=WORKER, num_workers=8, two_stage=two_stage,
        synthetic_sensor_path=Path("/syn/sensor_blobs") if two_stage else None,
        synthetic_scenes_path=Path("/syn/synthetic_scene_pickles") if two_stage else None,
        checkpoint_key=K.WRAPPER_CHECKPOINT_KEY, config_searchpath=Path("/repo/configs/navsim_hydra/v2"))


def test_one_stage_command():
    cmd = _score_cmd(False)
    assert cmd[1].endswith("run_pdm_score_one_stage.py")
    assert 'original_sensor_path="/sensors"' in cmd and "traffic_agents=non_reactive" in cmd
    assert not any(c.startswith("synthetic_") for c in cmd)
    assert not any(c.startswith("sensor_blobs_path") for c in cmd)


def test_two_stage_command():
    cmd = _score_cmd(True)
    assert cmd[1].endswith("navsim/planning/script/run_pdm_score.py")
    assert 'synthetic_scenes_path="/syn/synthetic_scene_pickles"' in cmd
    assert not any(c.startswith("traffic_agents") for c in cmd)


def test_two_stage_needs_synthetic_paths():
    with pytest.raises(K.DevkitError):
        K.pdm_score_command_v2(Path("/dk"), "navhard_two_stage", Path("/l"), Path("/s"), Path("/c"), Path("/o"),
                               "x", "a", [], Path("/ck"), WORKER, 4, two_stage=True)


def test_cache_command_v2():
    cmd = K.metric_cache_command_v2(Path("/dk"), "navhard_two_stage", Path("/logs"), Path("/cache"),
                                    {"name": "ray_distributed_no_torch"}, 40, synthetic_scenes_path=Path("/syn"))
    assert 'metric_cache_path="/cache"' in cmd and 'synthetic_scenes_path="/syn"' in cmd
    assert not any(c.startswith("cache.cache_path") for c in cmd)


def test_summarize_v2_one_and_two_stage(tmp_path):
    one = pd.DataFrame({"token": ["a", "b", "average_all_frames"], "valid": [True, True, True],
                        "score": [0.8, 0.6, 0.7], "lane_keeping": [1.0, 0.0, 0.5]})
    one.to_csv(tmp_path / "one.csv")
    r1 = R.summarize_v2(R.load_scores_v2(tmp_path / "one.csv"), two_stage=False, expected_tokens=["a", "b"])
    assert r1["epdms"] == pytest.approx(70.0) and r1["matches_devkit_average"] and r1["complete"]
    two = pd.DataFrame({"token": ["a", "s1", "extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                                  "extended_pdm_score_combined"],
                        "valid": [True, True, True, True, True], "score": [0.9, 0.5, 0.9, 0.5, 0.45]})
    two.to_csv(tmp_path / "two.csv")
    r2 = R.summarize_v2(R.load_scores_v2(tmp_path / "two.csv"), two_stage=True)
    assert r2["epdms"] == pytest.approx(45.0) and r2["epdms_stage_two"] == pytest.approx(50.0)
    assert r2["num_scored"] == 2 and r2["complete"]


@pytest.mark.parametrize("name,ov,out", [
    ("popA_ltf_navtest_v2", ["degradation.name=latency", "degradation.level=3", "policy.seed=1"],
     "runs/navsim_v2/A/ltf_seed1/latency__L3"),
    ("popB_ego_status_mlp_navhard_v2", ["policy.seed=2"], "runs/navsim_v2_ph/B/ego_status_mlp_seed2"),
    ("popA_transfuser_navhard_v2", ["degradation.name=tail_lateral", "degradation.level=1"],
     "runs/navsim_v2_ph/A/transfuser_seed0/tail_lateral__L1"),
])
def test_v2_experiment_configs_resolve(monkeypatch, name, ov, out):
    for k, v in {"WETE_REPO": "/r", "WETE_ROOT": "/w", "WETE_DATA_ROOT": "/d", "WETE_EXP_ROOT": "/x",
                 "WETE_CKPT_ROOT": "/ck", "WETE_THIRD_PARTY": "/tp", "OPENSCENE_DATA_ROOT": "/os",
                 "NUPLAN_MAPS_ROOT": "/m"}.items():
        monkeypatch.setenv(k, v)
    cfg = config_lib.resolve(config_lib.load_config(ROOT / "configs" / "experiments" / f"{name}.yaml", ov))
    assert cfg["experiment"]["out_dir"] == f"/x/{out}"
    assert cfg["evaluator"]["devkit_version"] == "v2" and cfg["evaluator"]["devkit_root"] == "/tp/navsim_v2"
    assert cfg["degradation"]["wrap"] is True
    if "navhard" in name:
        assert cfg["evaluator"]["two_stage"] and cfg["evaluator"]["synthetic_scenes_path"] == "/os/navhard_two_stage/synthetic_scene_pickles"
