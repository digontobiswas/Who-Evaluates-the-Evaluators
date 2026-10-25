import numpy as np
import pytest

from wete import degradations as D
from wete.wrappers import ClosedLoopDegrader


def straight(n=8, v=10.0):
    t = np.arange(1, n + 1) * 0.5
    return np.stack([v * t, np.zeros(n), np.zeros(n)], 1)


def test_level_zero_is_identity_everywhere():
    tr = straight()
    rng = D.scenario_rng("s", "lateral_noise")
    np.testing.assert_allclose(D.lateral_noise(tr, 0.0, rng), tr)
    np.testing.assert_allclose(D.late_braking(tr, 0.0), tr)
    img = np.full((4, 4, 3), 100, np.uint8)
    np.testing.assert_array_equal(D.image_noise(img, 0.0, rng), img)


def test_lateral_noise_nested_across_levels():
    tr = straight()
    offs = []
    for lvl in range(4):
        p = D.LEVELS["lateral_noise"][lvl]
        out = D.lateral_noise(tr, p, D.scenario_rng("tokA", "lateral_noise"))
        offs.append(out[-1, 1])
    signs = {np.sign(o) for o in offs[1:]}
    assert len(signs) == 1                         # same direction at every level
    assert np.all(np.diff(np.abs(offs)) > 0)       # magnitude grows with level
    assert abs(offs[3]) == pytest.approx(1.5, rel=1e-6)


def test_late_braking_scales_arc_length_on_straight_path():
    tr = straight()
    out = D.late_braking(tr, 0.4)
    np.testing.assert_allclose(out[:, 0], 1.4 * tr[:, 0])
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-12)


def test_late_braking_on_stopping_plan_stops_later():
    x = np.array([4.0, 7.0, 9.0, 10.0, 10.5, 10.6, 10.6, 10.6])
    tr = np.stack([x, np.zeros(8)], 1)
    out = D.late_braking(tr, 0.7)
    assert out[-1, 0] == pytest.approx(1.7 * 10.6)


def test_camera_dropout_order_and_effective_count():
    imgs = {c: np.ones((2, 2, 3)) for c in ["cam_l0", "cam_f0", "cam_r0"]}
    order = D.dropout_order_for(list(imgs))
    assert order[:3] == ["cam_f0", "cam_l0", "cam_r0"]
    out = D.camera_dropout(imgs, 1, order)
    assert out["cam_f0"].sum() == 0 and out["cam_l0"].sum() > 0
    assert D.effective_dropout(4, 3) == 3


def test_image_noise_common_random_numbers():
    img = np.full((8, 8, 3), 128.0)
    a = D.image_noise(img, 10, D.scenario_rng("s", "image_noise", "cam|0")) - img
    b = D.image_noise(img, 20, D.scenario_rng("s", "image_noise", "cam|0")) - img
    np.testing.assert_allclose(b, 2 * a, atol=1e-4)  # same field, scaled (float32 rounding)


def test_history_mask_and_removal():
    np.testing.assert_array_equal(D.history_keep_mask(4, 1.0), [1, 1, 1, 1])
    np.testing.assert_array_equal(D.history_keep_mask(4, 0.33), [0, 0, 1, 1])
    np.testing.assert_array_equal(D.history_keep_mask(4, 0.0), [0, 0, 0, 1])
    assert D.remove_history(["a", "b", "c", "d"], 0.33) == ["c", "c", "c", "d"]


def test_stale_history_shift():
    assert D.stale_history([0, 1, 2, 3], 1.0, 0.5) == [0, 0, 0, 1]
    assert D.stale_frame_index(4, 1.5, 0.5) == 0
    assert D.stale_frame_index(4, 5.0, 0.5) == 0   # clamped


def test_latency_queue():
    q = D.LatencyQueue(2)
    outs = [q(np.array([i])) for i in range(5)]
    assert [int(o[0]) for o in outs] == [0, 0, 0, 1, 2]


def test_spec_parse_and_population():
    s = D.DegradationSpec.parse("camera_dropout@2")
    assert (s.degradation, s.level, s.param) == ("camera_dropout", 2, 2)
    assert D.DegradationSpec.parse("none").is_identity
    specs = D.population_a_specs(closed_loop=False)
    assert all(sp.degradation != "latency" for sp in specs)
    assert len(specs) == 1 + 6 * 3
    over = D.population_a_specs(["late_braking"], levels={"late_braking": [0, 0.02, 0.05, 0.15]})
    assert [sp.param for sp in over] == [0.0, 0.02, 0.05, 0.15]


def test_closed_loop_degrader_stale_and_latency():
    spec = D.DegradationSpec("stale_obs", 2)            # 1.0 s
    deg = ClosedLoopDegrader(spec, "s", step_dt=0.25)
    seen = [deg.observe({"cam_f0": np.full((1, 1, 3), t, float)})["cam_f0"][0, 0, 0] for t in range(8)]
    assert seen == [0, 0, 0, 0, 0, 1, 2, 3]             # 4 steps behind, clamped at start
    lat = ClosedLoopDegrader(D.DegradationSpec("latency", 1), "s", step_dt=0.25)
    plans = [lat.act(np.full((8, 2), i, float))[0, 0] for i in range(4)]
    assert plans == [0, 0, 1, 2]
