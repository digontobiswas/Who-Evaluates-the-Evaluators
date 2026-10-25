"""Closed-loop policy evaluation inside an action-conditioned driving world model.

The proposal leaves the world model unspecified (main.tex: \\TODO{name}). This
module fixes the INTERFACE and the parts that do not depend on the model:

  * WorldModel   : what a model adapter must implement (reset / step / layout)
  * rollout()    : closed loop, with ClosedLoopDegrader applied exactly as in
                   HUGSIM, so Population A means the same thing in both
  * temporal_consistency(): cheap per-scenario reliability feature
  * score_rollout(): PDMS-style scoring from the model's predicted layout

Scoring generated video needs agent boxes. Two honest options:
  (a) the model predicts layout (boxes / occupancy) jointly with pixels; use it;
  (b) run a fixed 3D detector on generated frames. Then the score inherits
      detector errors on generated data -- part of what is being measured,
      but it must be reported, and the same detector must be used for all
      policies.
`score_rollout` accepts whichever layout source is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from wete import degradations as D
from wete.wrappers import ClosedLoopDegrader


@dataclass
class Layout:
    """Predicted scene state at one step, ego frame (x forward, y left)."""
    agent_boxes: np.ndarray            # (K, 5): x, y, length, width, heading
    agent_vel: Optional[np.ndarray]    # (K, 2) or None
    drivable: Optional[Callable[[np.ndarray], np.ndarray]] = None  # xy -> bool


@dataclass
class StepRecord:
    ego_xyh: np.ndarray                # world frame pose after the step
    ego_v: float
    ego_a: float
    layout: Layout
    frames: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)


class WorldModel(ABC):
    """Adapter contract for an action-conditioned driving world model."""

    step_dt: float = 0.5
    camera_names: Sequence[str] = ()

    @abstractmethod
    def reset(self, scenario_id: str, seed: int) -> Dict[str, np.ndarray]:
        """Condition on the logged context of the scenario; return current frames."""

    @abstractmethod
    def step(self, plan: np.ndarray) -> StepRecord:
        """Advance one step, executing the plan (N x 2|3, ego frame, x fwd y left)
        with the model's ego dynamics / tracker; return the generated step."""


def rollout(wm: WorldModel, policy: Callable[[Dict[str, np.ndarray], List[Dict]], np.ndarray],
            spec: D.DegradationSpec, scenario_id: str, horizon_s: float, seed: int,
            keep_frames: bool = False) -> List[StepRecord]:
    """policy(current_frames, history_frames) -> plan. `seed` must differ only by
    scenario, not by policy, if you want common random numbers in generation
    (recommended: it reduces variance of paired level comparisons; report it)."""
    deg = ClosedLoopDegrader(spec, scenario_id, step_dt=wm.step_dt,
                             camera_order=D.dropout_order_for(wm.camera_names) or D.DEFAULT_DROPOUT_PRIORITY)
    frames = wm.reset(scenario_id, seed)
    recs: List[StepRecord] = []
    for _ in range(int(round(horizon_s / wm.step_dt))):
        cur = deg.observe(frames)
        plan = deg.act(policy(cur, deg.history()))
        rec = wm.step(plan)
        frames = rec.frames
        if not keep_frames:
            rec.frames = {}
        recs.append(rec)
    return recs


def score_rollout(recs: List[StepRecord], expert_progress_m: float,
                  ego_len: float = 4.5, ego_wid: float = 1.9) -> Dict[str, float]:
    """PDMS-style score (NC x DAC x weighted(EP, TTC, C)); simplified geometry:
    axis-aligned-in-ego-frame overlap test and constant-velocity TTC."""
    collided, off_road, ttc_viol = False, False, False
    acc = []
    for r in recs:
        b = r.layout.agent_boxes
        if len(b):
            dx = np.abs(b[:, 0]) - (b[:, 2] + ego_len) / 2
            dy = np.abs(b[:, 1]) - (b[:, 3] + ego_wid) / 2
            if np.any((dx < 0) & (dy < 0)):
                collided = True
            ahead = (b[:, 0] > 0) & (np.abs(b[:, 1]) < (b[:, 3] + ego_wid) / 2)
            if ahead.any() and r.layout.agent_vel is not None:
                closing = r.ego_v - r.layout.agent_vel[ahead, 0]
                gap = b[ahead, 0] - (b[ahead, 2] + ego_len) / 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(closing > 0.1, gap / closing, np.inf)
                ttc_viol |= bool(np.any((t < 1.0) & (gap > 0)))
        if r.layout.drivable is not None and not bool(np.all(r.layout.drivable(np.zeros((1, 2))))):
            off_road = True
        acc.append(r.ego_a)
    prog = 0.0
    if len(recs) >= 2:
        prog = float(np.linalg.norm(recs[-1].ego_xyh[:2] - recs[0].ego_xyh[:2]))
    ep = min(1.0, prog / max(expert_progress_m, 1.0))
    comf = float(np.mean(np.abs(acc) <= 4.0)) if acc else 1.0
    score = (not collided) * (not off_road) * (5 * (not ttc_viol) + 5 * ep + 2 * comf) / 12.0
    return {"score": float(score), "nc": float(not collided), "dac": float(not off_road),
            "ttc": float(not ttc_viol), "ep": ep, "comfort": comf}


def temporal_consistency(frames: Sequence[np.ndarray], max_pairs: int = 16) -> float:
    """Cheap reliability feature: 1 - mean photometric error between frame t+1
    and frame t warped by dense optical flow (Farneback), in [0, 1].
    High = temporally consistent generation. Requires opencv-python."""
    import cv2
    if len(frames) < 2:
        return float("nan")
    idx = np.linspace(0, len(frames) - 2, min(max_pairs, len(frames) - 1)).astype(int)
    errs = []
    for i in idx:
        a = cv2.cvtColor(np.asarray(frames[i], np.uint8), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.asarray(frames[i + 1], np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        warped = cv2.remap(a, (gx + flow[..., 0]).astype(np.float32),
                           (gy + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        errs.append(np.abs(warped.astype(float) - b.astype(float)).mean() / 255.0)
    return float(1.0 - np.mean(errs))
