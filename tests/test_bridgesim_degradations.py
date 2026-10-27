"""wete.closed_loop.degraded_adapter with a stand-in for a BridgeSim model adapter."""

import json
import sys

import numpy as np
import pytest

from wete import degradations as D
from wete.closed_loop import bridgesim, degraded_adapter as DA
from wete.cli import bridgesim_degraded_eval as E

CAMS = ["CAM_L0", "CAM_F0", "CAM_R0"]


class TransfuserAdapter:
    """Mimics BridgeSim's adapter API: 3 cameras, 8 waypoints at 0.5 s, output [left, forward]."""

    def __init__(self):
        self.seen = None
        self.loaded = False

    def load_model(self):
        self.loaded = True

    def get_camera_configs(self):
        return {c: {} for c in CAMS}

    def get_waypoint_dt(self):
        return 0.5

    def prepare_input(self, images, ego_state, scenario_data, frame_id):
        self.seen = images
        return {"frame": frame_id}

    def run_inference(self, model_input):
        return model_input

    def parse_output(self, model_output, ego_state):
        forward = 5.0 * np.arange(1, 9)            # 10 m/s straight ahead
        return {"trajectory": np.stack([np.zeros(8), forward], axis=1)}


def frame(value):
    return {c: np.full((4, 4, 3), value, np.uint8) for c in CAMS}


def make(name, level, tmp_path=None):
    spec = D.DegradationSpec(name, level)
    log = None if tmp_path is None else tmp_path / "plans.jsonl"
    return DA.wrap(TransfuserAdapter(), spec, "sd_test", sim_dt=0.1, replan_interval_s=0.5, plan_log=log)


def ego(x=0.0, y=0.0, heading=0.0):
    return {"position": np.array([x, y, 0.0]), "heading": heading}


def test_wrapper_keeps_class_name_and_delegates():
    a = make("none", 0)
    assert type(a).__name__ == "TransfuserAdapter"
    a.load_model()
    assert a._base.loaded and a.get_waypoint_dt() == 0.5


def test_identity_passes_images_and_plan_through():
    a = make("none", 0)
    a.prepare_input(images=frame(7), ego_state=ego(), scenario_data={}, frame_id=0)
    assert all(int(v[0, 0, 0]) == 7 for v in a._base.seen.values())
    out = a.parse_output({}, ego())["trajectory"]
    np.testing.assert_allclose(out, TransfuserAdapter().parse_output({}, ego())["trajectory"])


def test_camera_dropout_front_first():
    a = make("camera_dropout", 1)
    a.prepare_input(images=frame(9), ego_state=ego(), scenario_data={}, frame_id=0)
    assert a._base.seen["CAM_F0"].sum() == 0 and a._base.seen["CAM_L0"].sum() > 0


def test_stale_obs_serves_older_frame():
    a = make("stale_obs", 1)  # 0.5 s = 5 frames at 10 Hz
    for f in range(8):
        a.prepare_input(images=frame(f), ego_state=ego(), scenario_data={}, frame_id=f)
    assert int(a._base.seen["CAM_F0"][0, 0, 0]) == 7 - 5


def test_output_degradations_change_the_plan(tmp_path):
    a = make("lateral_noise", 3, tmp_path)
    left = a.parse_output({}, ego())["trajectory"][:, 0]
    assert abs(left[-1]) == pytest.approx(1.5, rel=1e-6)
    a = make("late_braking", 3)
    forward = a.parse_output({}, ego())["trajectory"][:, 1]
    assert forward[-1] == pytest.approx(1.7 * 40.0, rel=1e-6)
    record = json.loads((tmp_path / "plans.jsonl").read_text().splitlines()[0])
    assert record["degradation"] == "lateral_noise" and len(record["executed_fl"]) == 8


def test_latency_reprojects_the_old_plan_into_the_current_frame():
    a = make("latency", 2)  # plan from 2 replans (1.0 s) ago
    a.parse_output({}, ego(0.0))            # produced at x=0
    a.parse_output({}, ego(5.0))            # produced at x=5
    out = a.parse_output({}, ego(10.0))["trajectory"]  # now at x=10, executing the plan from x=0
    # The old plan's waypoints are at world x = 5, 10, ..., 40; the first 1.0 s (2 waypoints) are
    # in the past, so the executed plan starts at world x=15, i.e. 5 m ahead of the current ego.
    np.testing.assert_allclose(out[:, 1], [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-12)


def test_latency_handles_rotation():
    a = make("latency", 1)
    a.parse_output({}, ego(0.0, 0.0, 0.0))                     # plan along +x
    out = a.parse_output({}, ego(0.0, 0.0, np.pi / 2))["trajectory"]  # ego turned to face +y
    # The old plan (along world +x) now lies to the ego's right: forward 0, left negative.
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-9)
    assert (out[:, 0] < 0).all()


def test_unsupported_degradations_are_refused():
    with pytest.raises(ValueError):
        make("history_removal", 1)
    with pytest.raises(SystemExit):
        E.parse_ours(["--degradation", "history_removal", "--level", "1"])
    spec, rest = E.parse_ours(["--degradation", "latency", "--level", "2", "--model-type", "ltf"])
    assert spec.param == 2 and rest == ["--model-type", "ltf"]


def test_command_uses_the_wrapper_only_when_needed(tmp_path):
    run = {"traffic_mode": "log_replay", "eval_mode": "closed_loop", "controller": "pure_pursuit",
           "sim_dt": 0.1, "replan_rate": 5, "ego_replay_frames": 20, "eval_frames": 80}
    plain = bridgesim.evaluator_command(tmp_path, "ltf", tmp_path, tmp_path, tmp_path, run,
                                        degradation={"name": "none", "level": 0})
    assert plain[1].endswith("unified_evaluator.py")
    wrapped = bridgesim.evaluator_command(tmp_path, "ltf", tmp_path, tmp_path, tmp_path, run,
                                          degradation={"name": "latency", "level": 2, "value": None})
    assert wrapped[1:7] == ["-m", "wete.cli.bridgesim_degraded_eval", "--degradation", "latency", "--level", "2"]
    rerun = bridgesim.evaluator_command(tmp_path, "ltf", tmp_path, tmp_path, tmp_path, run,
                                        degradation={"name": "none", "level": 0, "wrap": True})
    assert "wete.cli.bridgesim_degraded_eval" in rerun
