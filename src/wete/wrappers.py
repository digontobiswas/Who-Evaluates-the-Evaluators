"""Stateful, framework-agnostic degradation wrapper for CLOSED-LOOP evaluators
(HUGSIM pipe relay, world-model rollouts). One instance per (policy, scenario)
rollout; call `reset()` between scenarios.

    deg = ClosedLoopDegrader(spec, scenario_id, step_dt=0.25, camera_order=[...])
    obs_for_policy = deg.observe(obs_images)      # every step, before the policy
    plan_to_execute = deg.act(plan)               # every step, after the policy

The open-loop / single-shot NAVSIM path does not use this class: NAVSIM hands
the agent its whole 1.5 s history at once (see adapters/navsim).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from . import degradations as D


class ClosedLoopDegrader:
    def __init__(self, spec: D.DegradationSpec, scenario_id: str, step_dt: float,
                 camera_order: Sequence[str] = D.DEFAULT_DROPOUT_PRIORITY,
                 history_len: int = 4, history_dt: float = 0.5):
        """
        step_dt     : simulator step (HUGSIM: 0.25 s, configs/sim/kinematic.yaml)
        history_len : number of frames the policy consumes as history (incl. current)
        history_dt  : spacing of those frames (NAVSIM-trained policies: 0.5 s)
        """
        self.spec = spec
        self.sid = scenario_id
        self.dt = step_dt
        self.camera_order = list(camera_order)
        self.history_len = history_len
        self.history_dt = history_dt
        # buffer long enough for max staleness + full history window
        max_stale = max(D.LEVELS["stale_obs"]) if spec.degradation == "stale_obs" else 0.0
        stale = spec.param if spec.degradation == "stale_obs" else 0.0
        self._buf_len = int(np.ceil((max(max_stale, stale) + history_len * history_dt) / step_dt)) + 2
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._obs: Deque[Dict[str, np.ndarray]] = deque(maxlen=self._buf_len)
        self._lat = D.LatencyQueue(int(self.spec.param) if self.spec.degradation == "latency" else 0)
        self._rng_traj = D.scenario_rng(self.sid, self.spec.degradation, "traj")
        self._lat_sign_state = None
        self._step = 0

    # ------------------------------------------------------------------
    def _image_op(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d, p = self.spec.degradation, self.spec.param
        if self.spec.is_identity:
            return images
        if d == "camera_dropout":
            return D.camera_dropout(images, int(p), self.camera_order)
        if d == "image_noise":
            out = {}
            for cam, img in images.items():
                # common random numbers per (scenario, camera, step)
                rng = D.scenario_rng(self.sid, "image_noise", f"{cam}|{self._step}")
                out[cam] = D.image_noise(img, p, rng)
            return out
        return images

    def observe(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Returns the CURRENT frame the policy should see (stale if requested)."""
        self._obs.append(images)
        self._step += 1
        if self.spec.degradation == "stale_obs":
            shift = int(round(self.spec.param / self.dt))
            idx = max(0, len(self._obs) - 1 - shift)
            frame = self._obs[idx]
        else:
            frame = images
        return self._image_op(frame)

    def history(self) -> List[Dict[str, np.ndarray]]:
        """History window (oldest -> newest) at `history_dt` spacing, with
        stale shift and history removal applied. For policies that consume
        history; single-frame policies only need `observe`."""
        stride = max(1, int(round(self.history_dt / self.dt)))
        shift = int(round(self.spec.param / self.dt)) if self.spec.degradation == "stale_obs" else 0
        n = len(self._obs)
        idxs = [max(0, n - 1 - shift - stride * j) for j in reversed(range(self.history_len))]
        frames = [self._obs[i] for i in idxs]
        if self.spec.degradation == "history_removal":
            frames = D.remove_history(frames, self.spec.param)
        return [self._image_op(f) for f in frames]

    def act(self, plan: np.ndarray) -> np.ndarray:
        """Apply output-side degradations to the plan the policy just produced."""
        d, p = self.spec.degradation, self.spec.param
        out = np.asarray(plan, dtype=np.float64)
        if d == "lateral_noise" and p > 0:
            # direction fixed per scenario (common random numbers)
            out = D.lateral_noise(out, p, D.scenario_rng(self.sid, d))
        elif d == "late_braking" and p > 0:
            out = D.late_braking(out, p)
        if d == "latency":
            out = self._lat(out)
        return out
