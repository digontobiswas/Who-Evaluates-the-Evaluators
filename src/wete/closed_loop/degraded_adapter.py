"""Population A degradations for BridgeSim, applied around an unmodified BridgeSim model adapter.

BridgeSim's evaluator calls, every simulation frame (10 Hz)::

    model_input = adapter.prepare_input(images=..., ego_state=..., scenario_data=..., frame_id=...)

and, at replanning frames only (every ``replan_rate`` frames, 2 Hz in our protocol)::

    parsed = adapter.parse_output(adapter.run_inference(model_input), ego_state)
    parsed["trajectory"]          # (N, 2) waypoints in the current ego frame, columns [left, forward]

``DegradedAdapter`` forwards everything to the real adapter and changes only those two calls:

input side (prepare_input, every frame, via wete.wrappers.ClosedLoopDegrader)
    camera_dropout  blank the first k cameras the policy reads, front camera first
    image_noise     Gaussian pixel noise; the noise field is keyed by (scenario, camera, frame),
                    not by level, so levels share one direction (common random numbers)
    stale_obs       the policy sees the camera frame from s seconds ago (ego state stays current)

output side (parse_output, every replan)
    lateral_noise   smooth lateral drift of the plan, direction fixed per scenario
    late_braking    arc-length stretch of the plan
    latency         the plan executed now is the one produced k replans ago. It is re-expressed
                    from the ego frame it was produced in to the current ego frame (through world
                    coordinates), and its first k*replan_interval seconds, which lie in the past,
                    are dropped. Until k plans exist, the oldest one is used.

history_removal does not apply: the BridgeSim adapters of TransFuser and LTF are single-frame.
Every executed plan is appended to ``plan_log`` (JSON lines) for behaviour-conditioned labels.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np

from wete import degradations as D
from wete.wrappers import ClosedLoopDegrader

# BridgeSim camera names for NAVSIM policies (bridgesim/utils/camera_utils.NAVSIM_CAM_CONFIGS).
DROPOUT_PRIORITY = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_B0", "CAM_L2", "CAM_R2"]
SUPPORTED = {"none", "camera_dropout", "image_noise", "stale_obs", "lateral_noise", "late_braking", "latency"}


def ego_to_world(plan_fl: np.ndarray, position: Sequence[float], heading: float) -> np.ndarray:
    """(N, 2) [forward, left] in the ego frame at (position, heading) -> world (x, y).
    Same convention as BridgeSim's base_evaluator.process_frame."""
    c, s = np.cos(heading), np.sin(heading)
    fwd, left = plan_fl[:, 0], plan_fl[:, 1]
    return np.stack([position[0] + c * fwd - s * left, position[1] + s * fwd + c * left], axis=1)


def world_to_ego(points: np.ndarray, position: Sequence[float], heading: float) -> np.ndarray:
    """Inverse of ego_to_world: world (x, y) -> (N, 2) [forward, left]."""
    c, s = np.cos(heading), np.sin(heading)
    dx, dy = points[:, 0] - position[0], points[:, 1] - position[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


class DegradedAdapter:
    """Proxy around a BridgeSim model adapter. Use :func:`wrap` to build one."""

    def __init__(self, base: Any, spec: D.DegradationSpec, scenario_id: str, sim_dt: float,
                 replan_interval_s: float, plan_log: Optional[Path] = None):
        if spec.degradation not in SUPPORTED:
            raise ValueError(f"{spec.degradation} is not available in BridgeSim (supported: {sorted(SUPPORTED)})")
        self._base = base
        self.spec = spec
        self.scenario_id = scenario_id
        self.replan_interval_s = float(replan_interval_s)
        cams = list(base.get_camera_configs().keys())
        order = [c for c in DROPOUT_PRIORITY if c in cams] + [c for c in cams if c not in DROPOUT_PRIORITY]
        self.used_cameras = cams
        self._degrader = ClosedLoopDegrader(spec, scenario_id, step_dt=sim_dt, camera_order=order)
        self._world_plans: Deque[np.ndarray] = deque(maxlen=int(spec.param) + 1 if spec.degradation == "latency" else 1)
        self._replans = 0
        self._plan_log = Path(plan_log) if plan_log else None

    # Everything not overridden goes to the real adapter.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        if images:
            images = self._degrader.observe(dict(images))
        return self._base.prepare_input(images=images, ego_state=ego_state,
                                        scenario_data=scenario_data, frame_id=frame_id)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._base.parse_output(model_output, ego_state)
        traj = np.asarray(parsed["trajectory"], dtype=np.float64)        # [left, forward]
        fl = traj[:, [1, 0]]                                               # [forward, left]
        d, p = self.spec.degradation, self.spec.param
        if not self.spec.is_identity:
            if d == "lateral_noise":
                fl = D.lateral_noise(fl, p, D.scenario_rng(self.scenario_id, d))
            elif d == "late_braking":
                fl = D.late_braking(fl, p)
            elif d == "latency":
                fl = self._delayed(fl, ego_state)
        out = dict(parsed)
        out["trajectory"] = fl[:, [1, 0]]
        self._log(fl)
        self._replans += 1
        return out

    def _delayed(self, fl: np.ndarray, ego_state: Dict[str, Any]) -> np.ndarray:
        position = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        heading = float(ego_state["heading"])
        self._world_plans.append(ego_to_world(fl, position, heading))
        age = len(self._world_plans) - 1                                   # replans since production
        source = self._world_plans[0]
        model_dt = float(self._base.get_waypoint_dt())
        drop = int(round(age * self.replan_interval_s / model_dt))
        drop = min(drop, len(source) - 2)                                  # keep at least two waypoints
        return world_to_ego(source[drop:], position, heading)

    def _log(self, fl: np.ndarray) -> None:
        if self._plan_log is None:
            return
        self._plan_log.parent.mkdir(parents=True, exist_ok=True)
        record = {"scenario": self.scenario_id, "replan": self._replans, "degradation": self.spec.degradation,
                  "level": self.spec.level, "param": self.spec.param,
                  "executed_fl": np.round(fl, 4).tolist()}
        if self.spec.degradation == "camera_dropout":
            record["effective_k"] = D.effective_dropout(self.spec.param, len(self.used_cameras))
        with open(self._plan_log, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def wrap(base: Any, spec: D.DegradationSpec, scenario_id: str, sim_dt: float, replan_interval_s: float,
         plan_log: Optional[Path] = None) -> DegradedAdapter:
    """Build the proxy under the base adapter's class name: BridgeSim checks
    ``type(adapter).__name__`` in a few places (visualisation, DiffusionDrive handling)."""
    cls = type(type(base).__name__, (DegradedAdapter,), {})
    return cls(base, spec, scenario_id, sim_dt, replan_interval_s, plan_log)
