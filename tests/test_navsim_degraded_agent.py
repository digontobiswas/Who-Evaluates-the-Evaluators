"""DegradedAgent against stubs of the NAVSIM API, in both shapes this project pins.

v1.1 (3e8291b): AbstractAgent.__init__(requires_scene); Trajectory(poses) has a default sampling.
v2   (0a380a9): AbstractAgent.__init__(trajectory_sampling, requires_scene).

The stubs copy the field layout of navsim/common/dataclasses.py and abstract_agent.py at those
commits. They test our logic only; the smoke job on the cluster tests it against the real devkit.
"""

import importlib
import json
import sys
import types
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pytest

CAMS = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
USED = ["cam_l0", "cam_f0", "cam_r0"]
STUBBED = [
    "nuplan",
    "nuplan.planning",
    "nuplan.planning.simulation",
    "nuplan.planning.simulation.trajectory",
    "nuplan.planning.simulation.trajectory.trajectory_sampling",
    "navsim",
    "navsim.agents",
    "navsim.agents.abstract_agent",
    "navsim.common",
    "navsim.common.dataclasses",
    "wete.navsim.degraded_agent",
]


def _build_stubs(version: str):
    mods = {name: types.ModuleType(name) for name in STUBBED if name != "wete.navsim.degraded_agent"}

    @dataclass
    class TrajectorySampling:
        time_horizon: float = 4
        interval_length: float = 0.5

        @property
        def num_poses(self):
            return int(self.time_horizon / self.interval_length)

    @dataclass
    class Camera:
        image: Optional[np.ndarray] = None

    @dataclass
    class Cameras:
        cam_f0: Camera
        cam_l0: Camera
        cam_l1: Camera
        cam_l2: Camera
        cam_r0: Camera
        cam_r1: Camera
        cam_r2: Camera
        cam_b0: Camera

    @dataclass
    class Lidar:
        lidar_pc: Optional[np.ndarray] = None

    @dataclass
    class EgoStatus:
        ego_pose: np.ndarray
        ego_velocity: np.ndarray
        ego_acceleration: np.ndarray
        driving_command: np.ndarray

    @dataclass
    class AgentInput:
        ego_statuses: List[EgoStatus]
        cameras: List[Cameras]
        lidars: List[Lidar]

    @dataclass
    class SensorConfig:
        cam_f0: Union[bool, List[int]]
        cam_l0: Union[bool, List[int]]
        cam_l1: Union[bool, List[int]]
        cam_l2: Union[bool, List[int]]
        cam_r0: Union[bool, List[int]]
        cam_r1: Union[bool, List[int]]
        cam_r2: Union[bool, List[int]]
        cam_b0: Union[bool, List[int]]
        lidar_pc: Union[bool, List[int]]

    @dataclass
    class Trajectory:
        poses: np.ndarray
        trajectory_sampling: TrajectorySampling = field(default_factory=TrajectorySampling)

    @dataclass
    class SceneMetadata:
        initial_token: str

    class Scene:
        def __init__(self, token, future):
            self.scene_metadata = SceneMetadata(token)
            self._future = future

        def get_future_trajectory(self, n):
            return Trajectory(self._future[:n])

    if version == "v1":

        class AbstractAgent:
            def __init__(self, requires_scene: bool = False):
                self.requires_scene = requires_scene

    else:

        class AbstractAgent:
            def __init__(self, trajectory_sampling, requires_scene: bool = False):
                self.requires_scene = requires_scene
                self._trajectory_sampling = trajectory_sampling

    mods["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling = TrajectorySampling
    dataclasses_module = mods["navsim.common.dataclasses"]
    for name, obj in dict(
        Camera=Camera,
        Cameras=Cameras,
        Lidar=Lidar,
        EgoStatus=EgoStatus,
        AgentInput=AgentInput,
        SensorConfig=SensorConfig,
        Trajectory=Trajectory,
        Scene=Scene,
    ).items():
        setattr(dataclasses_module, name, obj)
    mods["navsim.agents.abstract_agent"].AbstractAgent = AbstractAgent
    return mods


@pytest.fixture(params=["v1", "v2"])
def nv(request):
    """Fresh import of the wrapper under one stubbed devkit; sys.modules restored afterwards."""
    saved = {name: sys.modules.get(name) for name in STUBBED}
    for name in STUBBED:
        sys.modules.pop(name, None)
    sys.modules.update(_build_stubs(request.param))
    module = importlib.import_module("wete.navsim.degraded_agent")
    ns = types.SimpleNamespace(version=request.param, module=module, dc=sys.modules["navsim.common.dataclasses"])
    try:
        yield ns
    finally:
        for name in STUBBED:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class FakeTransFuser:
    """Like TransFuser in v1.1: loads all eight cameras and the LiDAR of frame 3, reads three cameras."""

    requires_scene = False

    def __init__(self, dc, version):
        self.dc = dc
        if version == "v2":
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            self._trajectory_sampling = TrajectorySampling()
        self.seen = None
        self.seen_lidar = None

    def name(self):
        return "fake_tf"

    def initialize(self):
        pass

    def get_sensor_config(self):
        return self.dc.SensorConfig(**{c: [3] for c in CAMS}, lidar_pc=[3])

    def compute_trajectory(self, ai):
        cur = ai.cameras[-1]
        self.seen = {c: getattr(cur, c).image for c in CAMS}
        self.seen_lidar = ai.lidars[-1].lidar_pc
        t = np.arange(1, 9) * 0.5
        return self.dc.Trajectory(np.stack([10 * t, 0 * t, 0 * t], 1).astype(np.float32))


def make_input(dc, frames=(0, 1, 2, 3)):
    """Agent input whose frame f has pixel value 10 * (f + 1); frames not listed are empty."""
    cams, lids = [], []
    for f in range(4):
        if f in frames:
            cams.append(dc.Cameras(**{c: dc.Camera(np.full((4, 4, 3), 10 * (f + 1), np.uint8)) for c in CAMS}))
            lids.append(dc.Lidar(np.full((6, 5), f, np.float32)))
        else:
            cams.append(dc.Cameras(**{c: dc.Camera() for c in CAMS}))
            lids.append(dc.Lidar())
    egos = [dc.EgoStatus(np.zeros(3), np.array([f, 0.0]), np.zeros(2), np.zeros(4)) for f in range(4)]
    return dc.AgentInput(egos, cams, lids)


def make_scene(dc, token="tok123"):
    t = np.arange(1, 9) * 0.5
    return dc.Scene(token, np.stack([10 * t, 0 * t, 0 * t], 1))


def agent(nv, *args, **kwargs):
    base = FakeTransFuser(nv.dc, nv.version)
    kwargs.setdefault("used_cameras", USED)
    return base, nv.module.DegradedAgent(base, *args, **kwargs)


def test_constructs_under_both_devkit_apis(nv):
    _, ag = agent(nv)
    assert ag.requires_scene is True
    assert nv.module._ABSTRACT_TAKES_SAMPLING is (nv.version == "v2")


def test_clean_passthrough_logs_plan_and_l2(nv, tmp_path):
    base, ag = agent(nv, log_dir=str(tmp_path))
    tr = ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    np.testing.assert_allclose(tr.poses[:, 0], np.arange(1, 9) * 5.0)
    assert ag.get_sensor_config() == base.get_sensor_config()
    record = json.loads((tmp_path / "fake_tf__clean.plans.jsonl").read_text().strip())
    assert record["token"] == "tok123" and record["ade"] == 0.0 and record["l2_3s"] == 0.0
    assert len(record["poses"]) == 8


def test_camera_dropout_blanks_used_cameras_front_first(nv, tmp_path):
    base, ag = agent(nv, "camera_dropout", 1)
    ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    assert base.seen["cam_f0"].sum() == 0 and base.seen["cam_l0"].sum() > 0

    base, ag = agent(nv, "camera_dropout", 3, log_dir=str(tmp_path))  # 4 requested, 3 used
    ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    assert all(base.seen[c].sum() == 0 for c in USED)
    assert base.seen["cam_b0"].sum() > 0  # an unused camera is never spent on
    assert base.seen_lidar.sum() > 0
    record = json.loads(next(tmp_path.glob("*.plans.jsonl")).read_text().strip())
    assert record["effective_k"] == 3


def test_used_cameras_must_be_loaded_by_base(nv):
    base = FakeTransFuser(nv.dc, nv.version)
    with pytest.raises(ValueError):
        nv.module.DegradedAgent(base, used_cameras=["cam_x9"])


def test_stale_obs_loads_only_the_older_frame(nv):
    base, ag = agent(nv, "stale_obs", 2)  # 1.0 s -> history index 1
    cfg = ag.get_sensor_config()
    assert cfg.cam_f0 == [1] and cfg.lidar_pc == [1]
    # The devkit loads only frame 1, so the current frame arrives empty; the wrapper moves frame 1 in.
    ag.compute_trajectory(make_input(nv.dc, frames=(1,)), make_scene(nv.dc))
    assert int(base.seen["cam_f0"][0, 0, 0]) == 20
    assert float(base.seen_lidar[0, 0]) == 1.0


def test_history_removal_is_noop_for_single_frame_agent(nv):
    base, ag = agent(nv, "history_removal", 3)
    assert ag.get_sensor_config().cam_f0 == [0, 1, 2, 3]
    ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    assert int(base.seen["cam_f0"][0, 0, 0]) == 40


def test_image_noise_same_direction_across_levels(nv):
    seen = {}
    for level in (1, 2, 3):
        base, ag = agent(nv, "image_noise", level)
        ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
        seen[level] = base.seen["cam_f0"].astype(float) - 40.0
        assert base.seen["cam_b0"].sum() == 4 * 4 * 3 * 40  # unused camera untouched
    # Common random numbers: the level-2 field is the level-1 field scaled (up to rounding).
    corr = np.corrcoef(seen[1].ravel(), seen[2].ravel())[0, 1]
    assert corr > 0.9


def test_output_degradations(nv):
    _, ag = agent(nv, "late_braking", 3)
    tr = ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    assert tr.poses[-1, 0] == pytest.approx(1.7 * 40.0, rel=1e-5)
    _, ag = agent(nv, "lateral_noise", 3)
    tr = ag.compute_trajectory(make_input(nv.dc), make_scene(nv.dc))
    assert abs(tr.poses[-1, 1]) == pytest.approx(1.5, rel=1e-5)


def test_rejects_latency_and_bad_levels(nv):
    base = FakeTransFuser(nv.dc, nv.version)
    with pytest.raises(ValueError):
        nv.module.DegradedAgent(base, "latency", 1)
    with pytest.raises(ValueError):
        nv.module.DegradedAgent(base, "camera_dropout", 7)
    with pytest.raises(KeyError):
        nv.module.DegradedAgent(base, "rain", 1)


def test_requires_scene(nv):
    _, ag = agent(nv)
    with pytest.raises(RuntimeError):
        ag.compute_trajectory(make_input(nv.dc), None)
