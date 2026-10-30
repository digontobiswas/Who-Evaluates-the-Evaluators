"""NAVSIM agent wrapper that applies one graded degradation (Population A).

Works with both devkits this project pins (configs/third_party.env):

* NAVSIM v1.1 (``3e8291b``): ``AbstractAgent.__init__(requires_scene)``; the trajectory sampling
  lives inside the agent's own config.
* NAVSIM v2 (``0a380a9``): ``AbstractAgent.__init__(trajectory_sampling, requires_scene)``.

The version is detected from the installed ``AbstractAgent`` signature, not assumed. Both
devkits call ``agent.compute_trajectory(agent_input, scene)`` when ``requires_scene`` is true,
which is how the wrapper learns the scenario token (for common random numbers and for logging).

What each degradation does here
-------------------------------
camera_dropout   zero the first k cameras the base agent USES (``used_cameras``), in the
                 priority order of ``wete.degradations.DEFAULT_DROPOUT_PRIORITY`` (f0, l0, r0).
                 TransFuser and LTF use three cameras, so level 3 (4 cameras) blanks all three;
                 the logged ``effective_k`` says so. LiDAR is untouched unless
                 ``also_drop_lidar`` is true.
image_noise      Gaussian noise on the used cameras of every loaded frame. The noise field is
                 keyed by (token, camera, frame), not by level, so levels share one direction.
stale_obs        the current sensor frame is replaced by the one ``s`` seconds older (history is
                 4 frames at 2 Hz, so 0.5/1.0/1.5 s are history indices 2/1/0). Only that older
                 frame is loaded from disk. Ego status stays current unless
                 ``stale_ego_status`` is true.
history_removal  removed past frames are replaced by the oldest kept one. A no-op for
                 single-frame agents (TransFuser and LTF read only the current frame).
lateral_noise    smooth lateral drift on the output trajectory, direction fixed per token.
late_braking     arc-length stretch of the output trajectory.
tail_lateral,    specificity arm: the two operators above applied only beyond the first 1.5 s and
tail_stretch     10 m of the plan (degradations.tail_anchor), the part a replanning closed-loop
                 controller never executes.
latency          single-shot analogue of BridgeSim's latency: the policy plans from the whole
                 input of k frames ago (sensors and ego status, k = level parameter in replans of
                 0.5 s, at most 3 because the history holds 1.5 s), and that plan is re-expressed in
                 the current ego frame (through the logged ego pose of the older frame), its first k
                 waypoints, which now lie in the past, are dropped, and k waypoints are appended at
                 the last waypoint's velocity so the plan keeps its length. Level 3 is therefore
                 1.5 s here and 2.0 s in BridgeSim.

Every executed plan is appended to ``<log_dir>/<agent name>.plans.jsonl`` with open-loop L2
against the logged future (ADE, FDE, L2 at 1/2/3 s) where the scene has one. These lines give
the open-loop L2 evaluator and the behaviour-conditioned reliability labels.

Cost note: with ``requires_scene`` the devkit loads the scene in addition to the agent input,
so the current frame's sensors are read twice. The smoke job measures what this costs.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

from wete import degradations as D

CAMERA_FIELDS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
NUM_HISTORY = 4  # NAVSIM agent input: 4 frames at 2 Hz, the current frame is index 3
FRAME_DT = 0.5

# NAVSIM v2 added `trajectory_sampling` to AbstractAgent.__init__; v1.1 does not have it.
_ABSTRACT_TAKES_SAMPLING = "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters


def _requested(value) -> bool:
    """A SensorConfig field asks for data if it is True or a non-empty list of frames."""
    return (isinstance(value, bool) and value) or (isinstance(value, list) and len(value) > 0)


def stale_source_index(staleness_s: float, num_frames: int = NUM_HISTORY, frame_dt: float = FRAME_DT) -> int:
    """History index whose frame is served as the current one."""
    return D.stale_frame_index(num_frames, staleness_s, frame_dt)


def latency_frames(replans: float, num_frames: int = NUM_HISTORY) -> int:
    """Latency in history frames (one replan = one 0.5 s frame), capped by the stored history."""
    return int(min(max(int(round(replans)), 0), num_frames - 1))


def delayed_plan(poses: np.ndarray, old_pose: np.ndarray, k: int) -> np.ndarray:
    """Plan made k frames ago, as executed now.

    poses: (N, 3) [x, y, heading] in the ego frame of k frames ago; waypoint i at (i + 1) * 0.5 s.
    old_pose: [x, y, heading] of that ego frame in the current ego frame (NAVSIM's EgoStatus).
    Returns (N, 3) in the current ego frame: waypoints k.. of the old plan, then k waypoints
    continuing at the last waypoint's velocity and heading."""
    poses = np.asarray(poses, dtype=np.float64)
    if k <= 0:
        return poses
    x0, y0, h0 = (float(v) for v in np.asarray(old_pose, dtype=np.float64)[:3])
    c, s = np.cos(h0), np.sin(h0)
    cur = np.empty_like(poses)
    cur[:, 0] = x0 + c * poses[:, 0] - s * poses[:, 1]
    cur[:, 1] = y0 + s * poses[:, 0] + c * poses[:, 1]
    cur[:, 2] = np.arctan2(np.sin(poses[:, 2] + h0), np.cos(poses[:, 2] + h0))
    kept = cur[k:]
    if len(kept) >= 2:
        step = kept[-1, :2] - kept[-2, :2]
    elif len(kept) == 1:
        step = kept[-1, :2] - (cur[-2, :2] if len(cur) > 1 else np.zeros(2))
    else:
        raise ValueError(f"latency of {k} frames leaves no waypoint of a {len(poses)}-waypoint plan")
    extra = np.array([[*(kept[-1, :2] + (j + 1) * step), kept[-1, 2]] for j in range(k)])
    return np.vstack([kept, extra])


class DegradedAgent(AbstractAgent):
    def __init__(
        self,
        base_agent: AbstractAgent,
        degradation: str = "none",
        level: int = 0,
        value: Optional[float] = None,
        used_cameras: Optional[Sequence[str]] = None,
        log_dir: Optional[str] = None,
        also_drop_lidar: bool = False,
        stale_ego_status: bool = False,
    ):
        if _ABSTRACT_TAKES_SAMPLING:
            super().__init__(trajectory_sampling=base_agent._trajectory_sampling, requires_scene=True)
        else:
            super().__init__(requires_scene=True)
        self.base = base_agent
        self.spec = D.DegradationSpec(degradation, int(level), None if value is None else float(value))
        if self.spec.degradation != "none":
            D.level_param(self.spec.degradation, self.spec.level)  # validates name and level
        if self.spec.degradation in D.CLOSED_LOOP_ONLY - {"latency"}:
            raise ValueError(f"{self.spec.degradation} is closed-loop only; NAVSIM is single-shot")
        self._latency_k = latency_frames(self.spec.param) if self.spec.degradation == "latency" else 0
        self.also_drop_lidar = bool(also_drop_lidar)
        self.stale_ego_status = bool(stale_ego_status)
        self.log_dir = Path(log_dir) if log_dir else None

        base_cfg = base_agent.get_sensor_config()
        loaded = [c for c in CAMERA_FIELDS if _requested(getattr(base_cfg, c))]
        if used_cameras:
            unknown = sorted(set(used_cameras) - set(CAMERA_FIELDS))
            if unknown:
                raise ValueError(f"unknown cameras in used_cameras: {unknown}")
            not_loaded = sorted(set(used_cameras) - set(loaded))
            if not_loaded:
                raise ValueError(f"used_cameras {not_loaded} are not loaded by the base agent's sensor config")
            self._used_cams: List[str] = list(used_cameras)
        else:
            self._used_cams = loaded
        self._dropout_order = D.dropout_order_for(self._used_cams)

    # ---- AbstractAgent API ---------------------------------------------------------------
    def name(self) -> str:
        return f"{self.base.name()}__{self.spec.policy_suffix}"

    def initialize(self) -> None:
        self.base.initialize()

    def get_sensor_config(self) -> SensorConfig:
        """Load only the frames the degradation needs.

        stale_obs: just the older frame that will be served as current.
        history_removal: every history frame.
        Anything else: the base agent's own config, unchanged.
        """
        cfg = self.base.get_sensor_config()
        if self.spec.is_identity:
            return cfg
        if self.spec.degradation == "stale_obs":
            frames = [stale_source_index(self.spec.param)]
        elif self.spec.degradation == "latency":
            frames = [NUM_HISTORY - 1 - self._latency_k]
        elif self.spec.degradation == "history_removal":
            frames = list(range(NUM_HISTORY))
        else:
            return cfg
        values = {f.name: (list(frames) if _requested(getattr(cfg, f.name)) else False) for f in fields(cfg)}
        return SensorConfig(**values)

    def forward(self, features):
        return self.base.forward(features)

    def get_feature_builders(self):
        return self.base.get_feature_builders()

    def get_target_builders(self):
        return self.base.get_target_builders()

    def compute_loss(self, features, targets, predictions):
        return self.base.compute_loss(features, targets, predictions)

    def get_optimizers(self):
        return self.base.get_optimizers()

    # ---- degradations ----------------------------------------------------------------------
    def _degrade_input(self, ai: AgentInput, token: str) -> AgentInput:
        d, p = self.spec.degradation, self.spec.param
        if d == "latency" and not self.spec.is_identity:
            k = self._latency_k
            seconds = k * FRAME_DT
            return AgentInput(
                D.stale_history(list(ai.ego_statuses), seconds, FRAME_DT),
                D.stale_history(list(ai.cameras), seconds, FRAME_DT),
                D.stale_history(list(ai.lidars), seconds, FRAME_DT),
            )
        if self.spec.is_identity or D.ACTS_ON[d] != "input":
            return ai
        cams = list(ai.cameras)
        lids = list(ai.lidars)
        egos = list(ai.ego_statuses)
        if d == "stale_obs":
            src = stale_source_index(p, num_frames=len(cams))
            cams[-1] = cams[src]
            lids[-1] = lids[src]
            if self.stale_ego_status:
                egos[-1] = egos[src]
        elif d == "history_removal":
            cams = D.remove_history(cams, p)
            lids = D.remove_history(lids, p)
        elif d == "camera_dropout":
            k = int(p)
            cams = [self._drop_cams(c, k) for c in cams]
            if self.also_drop_lidar and k >= len(self._used_cams):
                lids = [
                    replace(lidar, lidar_pc=np.zeros_like(lidar.lidar_pc)) if lidar.lidar_pc is not None else lidar
                    for lidar in lids
                ]
        elif d == "image_noise":
            cams = [self._noise_cams(c, p, token, i) for i, c in enumerate(cams)]
        return AgentInput(egos, cams, lids)

    def _drop_cams(self, cameras, k: int):
        out = copy.copy(cameras)
        blanked = 0
        for name in self._dropout_order:
            if blanked >= k:
                break
            cam = getattr(out, name)
            if cam.image is not None:
                setattr(out, name, replace(cam, image=np.zeros_like(cam.image)))
                blanked += 1
        return out

    def _noise_cams(self, cameras, sigma: float, token: str, frame: int):
        out = copy.copy(cameras)
        for name in self._used_cams:
            cam = getattr(out, name)
            if cam.image is not None:
                rng = D.scenario_rng(token, "image_noise", f"{name}|{frame}")
                setattr(out, name, replace(cam, image=D.image_noise(cam.image, sigma, rng)))
        return out

    def _degrade_output(self, traj: Trajectory, token: str, original: Optional[AgentInput] = None) -> Trajectory:
        d, p = self.spec.degradation, self.spec.param
        if d == "latency" and not self.spec.is_identity:
            k = self._latency_k
            old_pose = np.asarray(original.ego_statuses[-1 - k].ego_pose, dtype=np.float64)
            poses = delayed_plan(np.asarray(traj.poses), old_pose, k)
            return Trajectory(np.asarray(poses, dtype=np.float32), traj.trajectory_sampling)
        if self.spec.is_identity or D.ACTS_ON[d] != "output":
            return traj
        poses = D.apply_output(traj.poses, self.spec, token, traj.trajectory_sampling.interval_length)
        return Trajectory(np.asarray(poses, dtype=np.float32), traj.trajectory_sampling)

    def compute_trajectory(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Trajectory:
        if scene is None:
            raise RuntimeError("DegradedAgent needs the scene (requires_scene=True) to key its randomness")
        token = scene.scene_metadata.initial_token
        ai = self._degrade_input(agent_input, token)
        if self.base.requires_scene:
            traj = self.base.compute_trajectory(ai, scene)
        else:
            traj = self.base.compute_trajectory(ai)
        traj = self._degrade_output(traj, token, agent_input)
        self._log_plan(traj, scene, token)
        return traj

    # ---- logging ---------------------------------------------------------------------------
    def _log_plan(self, traj: Trajectory, scene: Scene, token: str) -> None:
        """One JSON line per token: the executed plan, and open-loop L2 against the log."""
        if self.log_dir is None:
            return
        record = {
            "token": token,
            "degradation": self.spec.degradation,
            "level": self.spec.level,
            "param": self.spec.param,
            "poses": np.round(np.asarray(traj.poses)[:, :2], 4).tolist(),
            "effective_k": D.effective_dropout(self.spec.param, len(self._used_cams))
            if self.spec.degradation == "camera_dropout"
            else None,
        }
        try:
            expert = scene.get_future_trajectory(traj.trajectory_sampling.num_poses).poses
            dist = np.linalg.norm(np.asarray(traj.poses)[:, :2] - np.asarray(expert)[:, :2], axis=1)
            dt = traj.trajectory_sampling.interval_length
            record.update({"ade": float(dist.mean()), "fde": float(dist[-1])})
            for seconds in (1, 2, 3):
                index = int(round(seconds / dt)) - 1
                if 0 <= index < len(dist):
                    record[f"l2_{seconds}s"] = float(dist[index])
        except Exception as error:  # synthetic (v2 stage-two) scenes have no logged future
            record["l2_error"] = type(error).__name__
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # One short line per write in append mode; workers of one job share the file.
        with open(self.log_dir / f"{self.name()}.plans.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
