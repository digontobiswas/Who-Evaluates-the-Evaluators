"""Specificity arm: benign tail perturbations (degradations.tail_lateral / tail_stretch), their use
in the BridgeSim adapter, and the specificity section of the analysis."""

import numpy as np
import pandas as pd
import pytest

from wete import analysis
from wete import degradations as D
from wete.closed_loop import degraded_adapter as DA

DT = 0.5  # NAVSIM policies: 8 waypoints at 0.5 s


def plan(speed, curvature=0.0, n=8, dt=DT):
    """(n, 3) constant-speed arc in the ego frame [forward, left, heading]."""
    s = speed * dt * np.arange(1, n + 1)
    if abs(curvature) < 1e-9:
        return np.stack([s, np.zeros(n), np.zeros(n)], axis=1)
    h = curvature * s
    return np.stack([np.sin(h) / curvature, (1 - np.cos(h)) / curvature, h], axis=1)


# ---------------------------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------------------------

def test_anchor_needs_both_time_and_distance():
    assert D.tail_anchor(plan(10.0), DT) == 2          # t = 1.5 s, s = 15 m
    assert D.tail_anchor(plan(4.0), DT) == 4           # s reaches 10 m at t = 2.5 s
    assert D.tail_anchor(plan(1.0), DT) is None        # never 10 m within 4 s: identity
    assert D.tail_anchor(plan(2.5), DT) is None        # 10 m only at the last waypoint


@pytest.mark.parametrize("name,param", [("tail_lateral", 1.5), ("tail_stretch", 0.7)])
@pytest.mark.parametrize("speed,curv", [(10.0, 0.0), (6.0, 0.02), (15.0, -0.01)])
def test_head_is_unchanged_and_tail_moves(name, param, speed, curv):
    p = plan(speed, curv)
    a = D.tail_anchor(p, DT)
    spec = D.DegradationSpec(name, 3)
    out = D.apply_output(p, spec, "sd_x", DT)
    assert out.shape == p.shape
    np.testing.assert_array_equal(out[: a + 1], p[: a + 1])
    assert np.linalg.norm(out[-1, :2] - p[-1, :2]) > 0.3


def test_tail_lateral_matches_harmful_direction_and_peak():
    p = plan(10.0)
    harm = D.apply_output(p, D.DegradationSpec("lateral_noise", 3), "sd_dir", DT)
    benign = D.apply_output(p, D.DegradationSpec("tail_lateral", 3), "sd_dir", DT)
    assert np.sign(harm[-1, 1]) == np.sign(benign[-1, 1])
    assert abs(benign[-1, 1]) == pytest.approx(1.5)


def test_tail_stretch_lengthens_only_the_tail():
    p = plan(10.0)
    out = D.tail_stretch(p, 0.4, DT)
    a = D.tail_anchor(p, DT)
    s_a, s_end = p[a, 0], p[-1, 0]
    assert out[-1, 0] == pytest.approx(s_a + 1.4 * (s_end - s_a))


def test_levels_are_nested_and_clean_is_identity():
    p = plan(8.0, 0.01)
    errs = [np.abs(D.apply_output(p, D.DegradationSpec("tail_lateral", l), "sd", DT) - p).max() for l in range(4)]
    assert errs[0] == 0.0 and errs[1] < errs[2] < errs[3]


def test_population_a_default_excludes_benign():
    names = {s.degradation for s in D.population_a_specs()}
    assert not names & set(D.BENIGN)


# ---------------------------------------------------------------------------------------------
# The claim behind the arm: under BridgeSim's protocol (ea5d45f) the executed controls do not
# change. Re-implements the documented rules of base_evaluator (linear interpolation to 0.1 s,
# replan every 5 frames, waypoints consumed from the replan offset) and PurePursuitController
# (target speed from the first 10 waypoints; aim = waypoint whose distance is closest to
# Lf = 0.1 v + 4).
# ---------------------------------------------------------------------------------------------

def _interp(p_fl, model_dt=DT, sim_dt=0.1):
    pts = np.vstack([[0.0, 0.0], p_fl[:, :2]])
    t = np.arange(len(pts)) * model_dt
    ti = np.arange(sim_dt, t[-1] + sim_dt / 2, sim_dt)
    return np.stack([np.interp(ti, t, pts[:, 0]), np.interp(ti, t, pts[:, 1])], axis=1)


def _controls(p_fl, v):
    wp = _interp(p_fl)
    out = []
    for offset in range(5):                      # frames until the next replan
        # ego assumed to follow the plan; waypoints re-expressed relative to the consumed point
        here = wp[offset - 1] if offset > 0 else np.zeros(2)
        sub = wp[offset:] - here
        k = min(len(sub) - 1, 10)
        speed = np.linalg.norm(np.diff(np.vstack([sub[: k + 1]]), axis=0), axis=1).sum() / (k * 0.1)
        lf = 0.1 * v + 4.0
        aim = int(np.argmin(np.abs(np.linalg.norm(sub, axis=1) - lf)))
        out.append((round(speed, 9), aim, tuple(np.round(sub[aim], 9))))
    return out


@pytest.mark.parametrize("name", ["tail_lateral", "tail_stretch"])
@pytest.mark.parametrize("v", [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
@pytest.mark.parametrize("curv", [0.0, 0.03, -0.05])
def test_executed_controls_are_unchanged_under_the_bridgesim_protocol(name, v, curv):
    p = plan(v, curv)
    q = D.apply_output(p, D.DegradationSpec(name, 3), "sd_ctrl", DT)
    assert _controls(p, v) == _controls(q, v)


# ---------------------------------------------------------------------------------------------
# BridgeSim adapter
# ---------------------------------------------------------------------------------------------

class Adapter:
    def get_camera_configs(self):
        return {"CAM_F0": {}}

    def get_waypoint_dt(self):
        return 0.5

    def parse_output(self, model_output, ego_state):
        forward = 5.0 * np.arange(1, 9)
        return {"trajectory": np.stack([np.zeros(8), forward], axis=1)}   # [left, forward]


def test_adapter_applies_tail_perturbation_in_its_own_axes():
    a = DA.wrap(Adapter(), D.DegradationSpec("tail_lateral", 3), "sd_a", sim_dt=0.1, replan_interval_s=0.5)
    traj = a.parse_output(None, {"position": np.zeros(3), "heading": 0.0})["trajectory"]
    clean = Adapter().parse_output(None, None)["trajectory"]
    np.testing.assert_array_equal(traj[:3], clean[:3])
    assert abs(traj[-1, 0]) == pytest.approx(1.5)            # lateral column moved
    assert traj[-1, 1] == pytest.approx(clean[-1, 1], abs=1e-6)


# ---------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------

def _table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    sc = [f"s{i}" for i in range(n)]
    base_score = rng.uniform(0.3, 0.9, n)
    for base in ("tf", "ltf"):
        for deg, lvl, ref_drop, l2_rise in [("none", 0, 0.0, 0.0)] + \
                [("lateral_noise", l, 0.05 * l, 0.4 * l) for l in (1, 2, 3)] + \
                [("tail_lateral", l, 0.0, 0.3 * l) for l in (1, 2, 3)]:
            ref = base_score - ref_drop + rng.normal(0, 0.002, n)
            l2 = 0.5 + l2_rise + rng.normal(0, 0.01, n)
            for e, vals in (("bridgesim", ref), ("open_l2", l2)):
                rows += [{"evaluator": e, "population": "A", "base": base, "degradation": deg, "level": lvl,
                          "policy": f"{base}_{deg}_{lvl}", "scenario": s, "score": v} for s, v in zip(sc, vals)]
    return pd.DataFrame(rows)


META = {"bridgesim": {"higher_is_better": True, "family": "closed_loop", "reference": True},
        "open_l2": {"higher_is_better": False, "family": "open_loop", "reference": False}}


def test_specificity_validates_benign_and_flags_plan_distance(tmp_path):
    rep = analysis.run(_table(), META, str(tmp_path), cfg={"B": 300})
    spec = rep["specificity"]
    assert spec["validated"]["tail_lateral"] is True
    assert spec["verdicts"]["bridgesim"]["tail_lateral"]["verdict"] == "specific"
    assert spec["verdicts"]["open_l2"]["tail_lateral"]["verdict"] == "penalises benign"
    assert spec["verdicts"]["open_l2"]["tail_lateral"]["max_ratio"] == pytest.approx(0.75, abs=0.05)
    # benign perturbations are not part of Population A's monotonicity test or matrix
    assert "tail_lateral" not in rep["monotonicity"]["retained"] + rep["monotonicity"]["discarded"]
    df = pd.read_csv(tmp_path / "specificity.csv")
    assert set(df.perturbation) == {"tail_lateral"} and len(df) == 2 * 2 * 3


def test_specificity_fails_validation_when_reference_moves(tmp_path):
    t = _table()
    m = (t.evaluator == "bridgesim") & (t.degradation == "tail_lateral")
    t.loc[m, "score"] -= 0.03
    rep = analysis.run(t, META, str(tmp_path), cfg={"B": 300})
    assert rep["specificity"]["validated"]["tail_lateral"] is False
