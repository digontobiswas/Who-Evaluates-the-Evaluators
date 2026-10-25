"""End-to-end test of the HUGSIM pipe relay with a fake env and a fake policy
speaking HUGSIM's real protocol over real FIFOs (no HUGSIM install needed)."""

import pickle
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters" / "hugsim"))
from wete_hugsim import relay as RL  # noqa: E402
from wete import degradations as D  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
PLAN = np.stack([np.zeros(6), np.arange(1, 7) * 2.0], 1)   # straight ahead (x right = 0, y fwd)


def _run(tmp_path, spec, n_steps=3):
    outer, inner = tmp_path / "out", tmp_path / "out" / "wete_inner"
    inner.mkdir(parents=True)
    for p in (outer / "obs_pipe", outer / "plan_pipe", inner / "obs_pipe", inner / "plan_pipe"):
        RL.ensure_fifo(p)
    seen_by_policy, plans_to_env = [], []

    def policy():
        while True:
            msg = RL.read_msg(inner / "obs_pipe")
            if isinstance(msg, str):
                return
            seen_by_policy.append(msg[0]["rgb"])
            RL.write_msg(inner / "plan_pipe", PLAN.copy())

    def env():
        for t in range(n_steps):
            obs = {"rgb": {c: np.full((4, 4, 3), 100 + t, np.uint8) for c in CAMS}}
            RL.write_msg(outer / "obs_pipe", (obs, {"ego_pos": [float(t), 0.0, 0.0], "timestamp": t}))
            plans_to_env.append(RL.read_msg(outer / "plan_pipe"))
        RL.write_msg(outer / "obs_pipe", "Done")

    th = [threading.Thread(target=f, daemon=True) for f in
          (policy, env, lambda: RL.relay(outer, inner, spec, "scene0", 0.25))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in th), "relay deadlocked"
    return seen_by_policy, plans_to_env, outer


def test_relay_clean_is_transparent(tmp_path):
    seen, plans, outer = _run(tmp_path, D.DegradationSpec())
    assert len(seen) == 3 and all(s["CAM_FRONT"].mean() > 0 for s in seen)
    for p in plans:
        np.testing.assert_allclose(p, PLAN)
    lines = (outer / "wete_trace.jsonl").read_text().splitlines()
    assert len(lines) == 3
    import json
    p0 = np.asarray(json.loads(lines[0])["plan0"])            # (forward, left) frame
    np.testing.assert_allclose(p0[:, 0], PLAN[:, 1])
    assert "plan0" not in json.loads(lines[1])


def test_relay_camera_dropout(tmp_path):
    seen, _, _ = _run(tmp_path, D.DegradationSpec("camera_dropout", 1))
    for s in seen:
        assert s["CAM_FRONT"].sum() == 0
        assert s["CAM_FRONT_LEFT"].sum() > 0


def test_relay_lateral_noise_in_lidar_frame(tmp_path):
    _, plans, _ = _run(tmp_path, D.DegradationSpec("lateral_noise", 3))
    p = plans[0]
    # forward (y) progress preserved, lateral (x, right) offset of 1.5 m at the end
    np.testing.assert_allclose(p[:, 1], PLAN[:, 1], atol=1e-6)
    assert abs(p[-1, 0]) == pytest.approx(1.5, rel=1e-6)


def test_relay_stale_obs(tmp_path):
    seen, _, _ = _run(tmp_path, D.DegradationSpec("stale_obs", 1), n_steps=5)  # 0.5 s = 2 steps
    vals = [int(s["CAM_FRONT"][0, 0, 0]) for s in seen]
    assert vals == [100, 100, 100, 101, 102]
