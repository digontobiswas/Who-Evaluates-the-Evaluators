"""Revision (Nov 2026): head-only twin, projected plan, tail cut for a 1.0 s replanning interval,
and the configs/wiring that carry them to NAVSIM and BridgeSim."""

from pathlib import Path

import numpy as np
import pytest

from wete import config as config_lib
from wete import degradations as D
from wete.navsim import devkit_v1

from test_specificity import DT, _interp, plan

ROOT = Path(__file__).resolve().parents[1]
ENV = {"WETE_REPO": "/r", "WETE_ROOT": "/w", "WETE_DATA_ROOT": "/d", "WETE_EXP_ROOT": "/x",
       "WETE_CKPT_ROOT": "/ck", "WETE_THIRD_PARTY": "/tp", "OPENSCENE_DATA_ROOT": "/os",
       "NUPLAN_MAPS_ROOT": "/m"}


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("speed,curv", [(10.0, 0.0), (6.0, 0.02), (15.0, -0.01)])
def test_head_lateral_equals_lateral_noise_up_to_the_anchor_and_holds_after(speed, curv):
    p = plan(speed, curv)
    a = D.tail_anchor(p, DT)
    full = D.apply_output(p, D.DegradationSpec("lateral_noise", 3), "sd_h", DT)
    head = D.apply_output(p, D.DegradationSpec("head_lateral", 3), "sd_h", DT)
    np.testing.assert_allclose(head[: a + 1, :2], full[: a + 1, :2], atol=1e-9)
    # beyond the anchor the tail is carried with the offset reached at the anchor
    off_a = np.linalg.norm(head[a, :2] - p[a, :2])
    for i in range(a + 1, len(p)):
        assert np.linalg.norm(head[i, :2] - p[i, :2]) == pytest.approx(off_a, rel=1e-6, abs=1e-9)
    assert np.linalg.norm(full[-1, :2] - p[-1, :2]) > off_a


def test_head_lateral_without_anchor_is_lateral_noise():
    p = plan(1.0)                                  # never 10 m: the whole plan is executed
    np.testing.assert_array_equal(D.apply_output(p, D.DegradationSpec("head_lateral", 2), "s", DT),
                                  D.apply_output(p, D.DegradationSpec("lateral_noise", 2), "s", DT))


def test_head_lateral_is_diagnostic_only():
    assert "head_lateral" not in {s.degradation for s in D.population_a_specs()}


@pytest.mark.parametrize("speed,curv", [(10.0, 0.0), (6.0, 0.02), (15.0, -0.01)])
def test_projection_keeps_the_head_and_extrapolates_at_constant_velocity(speed, curv):
    p = plan(speed, curv)
    a = D.tail_anchor(p, DT)
    q = D.project_tail(p, DT)
    np.testing.assert_array_equal(q[: a + 1], p[: a + 1])
    step = p[a, :2] - p[a - 1, :2]
    for k, i in enumerate(range(a + 1, len(p)), start=1):
        np.testing.assert_allclose(q[i, :2], p[a, :2] + k * step, atol=1e-9)


@pytest.mark.parametrize("name", ["tail_lateral", "tail_stretch"])
def test_projection_removes_every_tail_perturbation(name):
    """The projected score of a benign twin equals the projected clean score by construction."""
    for speed, curv in [(10.0, 0.0), (6.0, 0.02), (15.0, -0.01), (4.0, 0.0)]:
        p = plan(speed, curv)
        twin = D.apply_output(p, D.DegradationSpec(name, 3), "sd_p", DT)
        np.testing.assert_allclose(D.project_tail(twin, DT), D.project_tail(p, DT), atol=1e-9)


def test_projection_without_anchor_is_identity():
    p = plan(1.0)
    np.testing.assert_array_equal(D.project_tail(p, DT), p)


# --------------------------------------------------------------------------------------------
# Tail cut for a 1.0 s replanning interval (BridgeSim PurePursuitController at ea5d45f)
# --------------------------------------------------------------------------------------------

def test_tail_cut_rule_reproduces_the_original_cut_and_gives_two_seconds_at_one_second():
    assert D.tail_t_cut_for(5) == 1.5
    assert D.tail_t_cut_for(10) == 2.0


def _controls(p_fl, v, replan_frames):
    """Controller inputs at every frame between two replans (see test_specificity._controls)."""
    wp = _interp(p_fl)
    out = []
    for offset in range(replan_frames):
        here = wp[offset - 1] if offset > 0 else np.zeros(2)
        sub = wp[offset:] - here
        k = min(len(sub) - 1, 10)
        speed = np.linalg.norm(np.diff(sub[: k + 1], axis=0), axis=1).sum() / (k * 0.1)
        lf = 0.1 * v + 4.0
        aim = int(np.argmin(np.abs(np.linalg.norm(sub, axis=1) - lf)))
        out.append((round(speed, 9), aim, tuple(np.round(sub[aim], 9))))
    return out


@pytest.mark.parametrize("name", ["tail_lateral", "tail_stretch"])
@pytest.mark.parametrize("v", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
@pytest.mark.parametrize("curv", [0.0, 0.03, -0.05])
def test_rederived_twins_leave_controls_unchanged_at_one_second_replanning(name, v, curv):
    p = plan(v, curv)
    spec = D.DegradationSpec(name, 3, tail_t_cut=D.tail_t_cut_for(10))
    assert _controls(p, v, 10) == _controls(D.apply_output(p, spec, "sd_c", DT), v, 10)


def test_original_twins_do_change_controls_at_one_second_replanning():
    """Expected outcome (a) of item 3: with the 1.5 s cut the controller reads part of the tail."""
    changed = 0
    for v in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        p = plan(v, 0.0)
        q = D.apply_output(p, D.DegradationSpec("tail_stretch", 3), "sd_c", DT)
        changed += _controls(p, v, 10) != _controls(q, v, 10)
    assert changed > 0


# --------------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------------

def test_wrapper_override_only_when_projected():
    assert not any("projection" in o for o in devkit_v1.wrapper_overrides({"name": "none"}, None))
    assert "agent.projection=tail_cv" in devkit_v1.wrapper_overrides({"name": "none", "projection": "tail_cv"}, None)


def test_every_wrapper_yaml_declares_projection():
    for f in list((ROOT / "configs/navsim_hydra").glob("v*/agent/degraded_*.yaml")):
        assert "\nprojection: none" in f.read_text(), f


def test_bridgesim_command_passes_the_tail_cut():
    from wete.closed_loop import bridgesim
    run = {"traffic_mode": "log_replay", "eval_mode": "closed_loop", "controller": "pure_pursuit",
           "sim_dt": 0.1, "replan_rate": 10, "ego_replay_frames": 20, "eval_frames": 80}
    cmd = bridgesim.evaluator_command(Path("/b"), "ltf", Path("/c"), Path("/s"), Path("/o"), run,
                                      degradation={"name": "tail_lateral", "level": 2, "tail_t_cut": 2.0})
    assert cmd[cmd.index("--tail-t-cut") + 1] == "2.0" and cmd[cmd.index("--replan-rate") + 1] == "10"
    from wete.cli import bridgesim_degraded_eval as E
    spec, rest = E.parse_ours(["--degradation", "tail_lateral", "--level", "2", "--tail-t-cut", "2.0", "--x", "1"])
    assert spec.tail_t_cut == 2.0 and rest == ["--x", "1"]


@pytest.mark.parametrize("name,ov,out,extra", [
    ("popA_ltf_navtest_proj", ["degradation.name=tail_lateral", "degradation.level=2"],
     "runs/navsim_v1_proj/A/ltf_seed0/tail_lateral__L2", {"projection": "tail_cv"}),
    ("popA_diffusiondrive_navtest_proj", [], "runs/navsim_v1_proj/A/diffusiondrive_seed0/none__L0",
     {"projection": "tail_cv", "wrap": True}),
    ("popB_ego_status_mlp_navtest_proj", ["policy.seed=2"], "runs/navsim_v1_proj/B/ego_status_mlp_seed2",
     {"projection": "tail_cv", "wrap": True}),
    ("popA_bridgesim_rr10", ["policy.family=ltf", "degradation.name=tail_lateral", "degradation.level=1"],
     "runs/bridgesim_rr10/A/ltf_seed0/tail_lateral__L1", {"tail_t_cut": None}),
    ("popA_bridgesim_rr10_t20", ["policy.family=transfuser", "degradation.name=tail_stretch", "degradation.level=3"],
     "runs/bridgesim_rr10_t20/A/transfuser_seed0/tail_stretch__L3", {"tail_t_cut": 2.0}),
])
def test_revision_configs_resolve(monkeypatch, name, ov, out, extra):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = config_lib.resolve(config_lib.load_config(ROOT / "configs/experiments" / f"{name}.yaml", ov))
    assert cfg["experiment"]["out_dir"] == f"/x/{out}"
    for k, v in extra.items():
        assert cfg["degradation"][k] == v, k
    if "rr10" in name:
        assert cfg["evaluator"]["run"]["replan_rate"] == 10 and cfg["compare_with_reference"] is False
