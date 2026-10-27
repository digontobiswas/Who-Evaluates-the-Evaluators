"""Graded degradation operators (Population A).

All operators are pure functions on numpy data so the same code is used by the
NAVSIM wrapper, the HUGSIM pipe relay and the toy simulator.

Conventions
-----------
* Trajectory: array (N, 3) of (x, y, heading) in the ego frame at t=0,
  x forward, y left (NAVSIM convention). (N, 2) is also accepted.
* Images: dict camera_name -> HxWxC array (uint8 or float in [0, 255]).
* Level 0 is always the identity.

Common random numbers
---------------------
Stochastic operators draw their randomness from `scenario_rng(scenario_id,
degradation)` which does NOT depend on the level. Level l and level l+1 therefore
share the same noise direction and differ only in magnitude, so the variants are
nested and paired differences between levels have far lower variance than with
independent noise. This matters for the monotonicity test: with independent
noise a single unlucky draw can make level 2 look better than level 1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Level tables (proposal, "Degradations used in Population A")
# --------------------------------------------------------------------------

LEVELS: Dict[str, List[float]] = {
    # number of cameras blanked
    "camera_dropout": [0, 1, 2, 4],
    # Gaussian sigma in pixel values (0-255 scale)
    "image_noise": [0.0, 10.0, 25.0, 50.0],
    # age of the observation the policy acts on, seconds
    "stale_obs": [0.0, 0.5, 1.0, 1.5],
    # action applied this many control steps late (closed-loop only)
    "latency": [0, 1, 2, 4],
    # fraction of past frames kept
    "history_removal": [1.0, 0.66, 0.33, 0.0],
    # peak lateral drift of the planned trajectory, metres
    "lateral_noise": [0.0, 0.3, 0.8, 1.5],
    # longitudinal stretch of the planned trajectory (late / weak braking)
    "late_braking": [0.0, 0.15, 0.40, 0.70],
    # Specificity arm (benign): the same two output operators, applied only to the tail of the
    # plan that a replanning closed-loop controller never executes (see `tail_anchor`). Same
    # parameters as their harmful counterparts so the two can be compared level by level.
    "tail_lateral": [0.0, 0.3, 0.8, 1.5],
    "tail_stretch": [0.0, 0.15, 0.40, 0.70],
}

# Benign perturbation -> the harmful degradation it mirrors.
BENIGN: Dict[str, str] = {"tail_lateral": "lateral_noise", "tail_stretch": "late_braking"}

# Where the unexecuted tail starts: the first waypoint that is both at least TAIL_T_CUT seconds
# and at least TAIL_S_CUT metres (arc length) ahead. Chosen from BridgeSim's protocol (commit
# ea5d45f), before any specificity run:
#   * the plan is replaced every 0.5 s (replan_rate 5 x sim_dt 0.1), so at most the waypoints up
#     to 0.4 s after the replan are consumed;
#   * PurePursuitController sets the target speed from the first 10 waypoints at 0.1 s, i.e. up
#     to 0.4 + 1.0 = 1.4 s of the plan  ->  TAIL_T_CUT = 1.5 s;
#   * it steers towards the waypoint whose distance is closest to Lf = 4 + 0.1 v metres. With up
#     to 0.4 s consumed, the tail starts at least TAIL_S_CUT - 0.4 v metres from the car, which is
#     > Lf + 1 m for v <= 10 m/s; above that the 1.5 s cut gives >= 1.1 v > Lf + 1 m.
# Under that protocol the executed controls are unchanged, so a closed-loop score should not
# move. Whether it does not is TESTED on the reference evaluator (equivalence test), not assumed.
TAIL_T_CUT = 1.5
TAIL_S_CUT = 10.0

FAILURE_CLASS: Dict[str, str] = {
    "camera_dropout": "perception",
    "image_noise": "perception",
    "stale_obs": "timing",
    "latency": "timing",
    "history_removal": "memory",
    "lateral_noise": "control",
    "late_braking": "control",
    "tail_lateral": "benign",
    "tail_stretch": "benign",
}

ACTS_ON: Dict[str, str] = {
    "camera_dropout": "input",
    "image_noise": "input",
    "stale_obs": "input",
    "latency": "output",
    "history_removal": "input",
    "lateral_noise": "output",
    "late_braking": "output",
    "tail_lateral": "output",
    "tail_stretch": "output",
}

CLOSED_LOOP_ONLY = {"latency"}

# Order in which cameras are blanked. The front camera goes first because a
# blanked rear camera is invisible to most planners (TransFuser only reads
# l0/f0/r0); blanking unused cameras would make the "degradation" a no-op and
# fail the monotonicity test for a trivial reason. Policies whose sensor config
# excludes a camera simply skip it: see `dropout_order_for`.
DEFAULT_DROPOUT_PRIORITY: List[str] = [
    "cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_b0", "cam_l2", "cam_r2",
]


def level_param(degradation: str, level: int) -> float:
    if degradation not in LEVELS:
        raise KeyError(f"unknown degradation {degradation!r}; known: {sorted(LEVELS)}")
    if not 0 <= level < len(LEVELS[degradation]):
        raise ValueError(f"level must be in [0, {len(LEVELS[degradation]) - 1}], got {level}")
    return LEVELS[degradation][level]


def scenario_rng(scenario_id: str, degradation: str, salt: str = "") -> np.random.Generator:
    """Deterministic RNG keyed by scenario and degradation type, NOT by level."""
    h = hashlib.sha256(f"{scenario_id}|{degradation}|{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# --------------------------------------------------------------------------
# Input-side operators
# --------------------------------------------------------------------------

def dropout_order_for(used_cameras: Sequence[str],
                      priority: Sequence[str] = DEFAULT_DROPOUT_PRIORITY) -> List[str]:
    """Blanking order restricted to the cameras the policy actually reads."""
    used = set(used_cameras)
    order = [c for c in priority if c in used]
    order += sorted(used - set(order))
    return order


def camera_dropout(images: Mapping[str, np.ndarray], n_blank: int,
                   order: Sequence[str]) -> Dict[str, np.ndarray]:
    """Blank (zero) the first `n_blank` cameras of `order` that are present.

    If the policy reads fewer cameras than n_blank, all of them are blanked;
    callers should record the effective count (see `effective_dropout`).
    """
    out = dict(images)
    blanked = 0
    for cam in order:
        if blanked >= n_blank:
            break
        if cam in out and out[cam] is not None:
            out[cam] = np.zeros_like(out[cam])
            blanked += 1
    return out


def effective_dropout(n_blank: int, n_used: int) -> int:
    return min(int(n_blank), int(n_used))


def image_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise in pixel units, clipped to [0, 255].

    The standard-normal field is drawn once per (scenario, camera) by the caller
    via `rng`, so different levels scale the same field (common random numbers).
    """
    if sigma <= 0:
        return image
    z = rng.standard_normal(image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + sigma * z
    noisy = np.clip(noisy, 0.0, 255.0)
    return noisy.astype(image.dtype) if np.issubdtype(image.dtype, np.integer) else noisy


def stale_frame_index(n_frames: int, staleness_s: float, frame_dt_s: float) -> int:
    """Index (into a history ordered oldest -> newest) of the frame to use as
    'current' when observations are `staleness_s` old. Clamped to the oldest
    available frame; callers should log when clamping occurs because then
    levels may collapse (e.g. NAVSIM keeps 1.5 s of history at 2 Hz)."""
    shift = int(round(staleness_s / frame_dt_s))
    return max(0, n_frames - 1 - shift)


def stale_history(frames: Sequence, staleness_s: float, frame_dt_s: float) -> List:
    """Shift a history (oldest -> newest) back in time by `staleness_s`,
    padding the front with the oldest frame so the length is unchanged."""
    n = len(frames)
    shift = int(round(staleness_s / frame_dt_s))
    if shift <= 0:
        return list(frames)
    kept = list(frames[: max(1, n - shift)])
    pad = [kept[0]] * (n - len(kept))
    return pad + kept


def history_keep_mask(n_frames: int, frac_kept: float) -> np.ndarray:
    """Boolean mask over past frames (oldest -> newest, current frame last).
    The current frame is always kept; of the n-1 past frames we keep the
    most recent round(frac * (n-1)). Removing the most distant ones first keeps
    levels nested."""
    mask = np.zeros(n_frames, dtype=bool)
    mask[-1] = True
    n_past = n_frames - 1
    k = int(round(frac_kept * n_past))
    if k > 0:
        mask[-1 - k:-1] = True
    return mask


def remove_history(frames: Sequence, frac_kept: float) -> List:
    """Replace removed past frames by a copy of the oldest kept frame.
    (Zero-filling would inject an out-of-distribution signal and conflate
    'no memory' with 'garbage input'.)"""
    mask = history_keep_mask(len(frames), frac_kept)
    first_kept = int(np.argmax(mask))
    return [frames[i] if mask[i] else frames[first_kept] for i in range(len(frames))]


# --------------------------------------------------------------------------
# Output-side operators
# --------------------------------------------------------------------------

def _as_xyh(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[-1] == 2:
        traj = np.concatenate([traj, _headings(traj)[:, None]], axis=-1)
    return traj.copy()


def _headings(xy: np.ndarray) -> np.ndarray:
    d = np.diff(np.vstack([[0.0, 0.0], xy[:, :2]]), axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    small = np.hypot(d[:, 0], d[:, 1]) < 1e-3
    # keep previous heading for near-stationary steps
    for i in np.where(small)[0]:
        h[i] = h[i - 1] if i > 0 else 0.0
    return h


def lateral_noise(traj: np.ndarray, peak_m: float, rng: np.random.Generator) -> np.ndarray:
    """Add a smooth lateral drift that grows linearly to `peak_m` at the last
    waypoint, applied perpendicular to the local heading.

    Direction (left/right) and a small shape jitter come from `rng` (common
    across levels). Drift rather than i.i.d. jitter: i.i.d. jitter of 0.3 m is
    largely filtered by any tracking controller and would fail monotonicity."""
    out = _as_xyh(traj)
    if peak_m <= 0:
        return out[:, : np.asarray(traj).shape[-1]]
    n = len(out)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    ramp = np.linspace(1.0 / n, 1.0, n)
    off = sign * peak_m * ramp
    h = out[:, 2]
    out[:, 0] += -np.sin(h) * off
    out[:, 1] += np.cos(h) * off
    out[:, 2] = _headings(out[:, :2])
    return out[:, : np.asarray(traj).shape[-1]]


def late_braking(traj: np.ndarray, stretch: float) -> np.ndarray:
    """Stretch the trajectory along its own path: the arc length reached at
    time t becomes (1 + stretch) * s(t). For a decelerating plan this means
    stopping later and further (late / weak braking); beyond the end of the
    planned path we extrapolate along the final heading.

    Note: this also speeds up a constant-speed plan by the same factor. That is
    intended (the policy 'overshoots' its own plan) but means the operator is
    not purely a braking failure; report it as 'longitudinal overshoot' if the
    braking-only interpretation is challenged."""
    out = _as_xyh(traj)
    if stretch <= 0:
        return out[:, : np.asarray(traj).shape[-1]]
    pts = np.vstack([[0.0, 0.0], out[:, :2]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_new = (1.0 + stretch) * s[1:]
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    beyond = s_new > s[-1]
    if beyond.any():
        hend = out[-1, 2]
        extra = s_new[beyond] - s[-1]
        x[beyond] = pts[-1, 0] + np.cos(hend) * extra
        y[beyond] = pts[-1, 1] + np.sin(hend) * extra
    out[:, 0], out[:, 1] = x, y
    out[:, 2] = _headings(out[:, :2])
    return out[:, : np.asarray(traj).shape[-1]]


def tail_anchor(traj: np.ndarray, dt: float, t_cut: float = TAIL_T_CUT,
                s_cut: float = TAIL_S_CUT) -> Optional[int]:
    """Index of the last waypoint left unchanged by the tail operators: the first waypoint at
    least `t_cut` seconds and `s_cut` metres (arc length from the origin) ahead. Waypoint i is at
    time (i + 1) * dt. None if no waypoint qualifies or it is the last one (nothing to change)."""
    xy = np.asarray(traj, dtype=np.float64)[:, :2]
    s = np.cumsum(np.hypot(*np.diff(np.vstack([[0.0, 0.0], xy]), axis=0).T))
    t = (np.arange(len(xy)) + 1) * float(dt)
    ok = np.where((t >= t_cut - 1e-9) & (s >= s_cut))[0]
    if len(ok) == 0 or ok[0] >= len(xy) - 1:
        return None
    return int(ok[0])


def _keep_head(original: np.ndarray, changed: np.ndarray, a: int) -> np.ndarray:
    """Waypoints 0..a (positions and headings) stay exactly as the policy produced them."""
    changed[: a + 1] = original[: a + 1]
    return changed


def tail_lateral(traj: np.ndarray, peak_m: float, rng: np.random.Generator, dt: float,
                 t_cut: float = TAIL_T_CUT, s_cut: float = TAIL_S_CUT) -> np.ndarray:
    """Benign twin of `lateral_noise`: a lateral drift that is zero up to the anchor waypoint and
    grows linearly to `peak_m` at the last waypoint. Pass the `lateral_noise` rng so the direction
    matches the harmful operator on the same scenario."""
    shape = np.asarray(traj).shape[-1]
    out = _as_xyh(traj)
    sign = 1.0 if rng.random() < 0.5 else -1.0   # drawn first, as in lateral_noise
    a = tail_anchor(out, dt, t_cut, s_cut)
    if peak_m <= 0 or a is None:
        return out[:, :shape]
    orig = out.copy()
    n = len(out)
    ramp = np.zeros(n)
    ramp[a + 1:] = (np.arange(a + 1, n) - a) / (n - 1 - a)
    off = sign * peak_m * ramp
    h = orig[:, 2]
    out[:, 0] += -np.sin(h) * off
    out[:, 1] += np.cos(h) * off
    out[:, 2] = _headings(out[:, :2])
    return _keep_head(orig, out, a)[:, :shape]


def tail_stretch(traj: np.ndarray, stretch: float, dt: float,
                 t_cut: float = TAIL_T_CUT, s_cut: float = TAIL_S_CUT) -> np.ndarray:
    """Benign twin of `late_braking`: beyond the anchor waypoint (arc length s_a), arc length
    s becomes s_a + (1 + stretch) (s - s_a); waypoints up to the anchor are unchanged."""
    shape = np.asarray(traj).shape[-1]
    out = _as_xyh(traj)
    a = tail_anchor(out, dt, t_cut, s_cut)
    if stretch <= 0 or a is None:
        return out[:, :shape]
    orig = out.copy()
    pts = np.vstack([[0.0, 0.0], out[:, :2]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    s_a = s[a + 1]
    s_new = np.where(s[1:] > s_a, s_a + (1.0 + stretch) * (s[1:] - s_a), s[1:])
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    beyond = s_new > s[-1]
    if beyond.any():
        hend = orig[-1, 2]
        extra = s_new[beyond] - s[-1]
        x[beyond] = pts[-1, 0] + np.cos(hend) * extra
        y[beyond] = pts[-1, 1] + np.sin(hend) * extra
    out[:, 0], out[:, 1] = x, y
    out[:, 2] = _headings(out[:, :2])
    return _keep_head(orig, out, a)[:, :shape]


def apply_output(traj: np.ndarray, spec: "DegradationSpec", scenario_id: str, dt: float) -> np.ndarray:
    """Every stateless output-side operator, in one place for the NAVSIM agent and the BridgeSim
    adapter. `dt`: time between waypoints (s), waypoint i at (i + 1) * dt. Latency is stateful
    and handled by the closed-loop adapters."""
    d, p = spec.degradation, spec.param
    if spec.is_identity:
        return np.asarray(traj)
    if d == "lateral_noise":
        return lateral_noise(traj, p, scenario_rng(scenario_id, d))
    if d == "late_braking":
        return late_braking(traj, p)
    if d == "tail_lateral":
        return tail_lateral(traj, p, scenario_rng(scenario_id, "lateral_noise"), dt)
    if d == "tail_stretch":
        return tail_stretch(traj, p, dt)
    return np.asarray(traj)


class LatencyQueue:
    """FIFO that returns the plan produced `delay` steps ago (closed loop only).
    Until the queue is full it returns the oldest plan available, i.e. the
    vehicle executes its very first plan for the first `delay` steps."""

    def __init__(self, delay: int):
        self.delay = int(delay)
        self._buf: List[np.ndarray] = []

    def __call__(self, plan: np.ndarray) -> np.ndarray:
        self._buf.append(plan)
        if len(self._buf) > self.delay + 1:
            self._buf.pop(0)
        return self._buf[0]

    def reset(self) -> None:
        self._buf.clear()


# --------------------------------------------------------------------------
# Spec object used by all adapters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DegradationSpec:
    degradation: str = "none"
    level: int = 0
    value: Optional[float] = None     # explicit parameter (e.g. from calibrate_levels)

    @property
    def param(self) -> float:
        if self.value is not None:
            return float(self.value)
        if self.degradation == "none" or self.level == 0:
            return 0.0 if self.degradation != "history_removal" else 1.0
        return level_param(self.degradation, self.level)

    @property
    def is_identity(self) -> bool:
        return self.degradation == "none" or self.level == 0

    @property
    def policy_suffix(self) -> str:
        return "clean" if self.is_identity else f"{self.degradation}@L{self.level}"

    @classmethod
    def parse(cls, s: str) -> "DegradationSpec":
        """'camera_dropout@2' or 'none'."""
        if s in ("", "none", "clean"):
            return cls()
        name, _, lvl = s.partition("@")
        lvl = lvl.lstrip("Ll")
        spec = cls(name, int(lvl))
        level_param(spec.degradation, spec.level)  # validate
        return spec


def population_a_specs(degradations: Optional[Sequence[str]] = None,
                       closed_loop: bool = True,
                       levels: Optional[Mapping[str, Sequence[float]]] = None) -> List[DegradationSpec]:
    """All (degradation, level) specs, with a single shared level-0 entry.
    `levels` overrides the default parameter table (index 0 = clean value)."""
    table = {**LEVELS, **(levels or {})}
    degs = list(degradations or [d for d in LEVELS if d not in BENIGN])
    if not closed_loop:
        degs = [d for d in degs if d not in CLOSED_LOOP_ONLY]
    specs = [DegradationSpec()]
    for d in degs:
        for lvl in range(1, len(table[d])):
            specs.append(DegradationSpec(d, lvl, value=float(table[d][lvl])))
    return specs


# --------------------------------------------------------------------------
# Level calibration on a DISJOINT pilot scenario set
# --------------------------------------------------------------------------

def calibrate_levels(pilot: Mapping[float, float], clean_value: float, n_levels: int = 3,
                     min_step: float = 0.0) -> Optional[List[float]]:
    """Choose `n_levels` parameter values from a pilot sweep.

    pilot: candidate parameter -> mean reference-evaluator score (higher =
    better), measured on PILOT scenarios that are NOT reused in the main study
    (otherwise the monotonicity test is applied to levels selected on the same
    data, which inflates its pass rate).

    Targets: equally spaced score drops between the clean score and the worst
    score reached before saturation. Returns None if the sweep never drops by
    more than `min_step` per level (the degradation is invisible to the
    reference and should be reported as such, not forced).
    """
    s0 = pilot[clean_value]
    cand = sorted((p for p in pilot if p != clean_value), key=lambda p: abs(p - clean_value))
    # keep the running-minimum frontier (monotone part of the sweep)
    frontier, best = [], s0
    for p in cand:
        if pilot[p] < best - 1e-12:
            frontier.append(p)
            best = pilot[p]
    if not frontier:
        return None
    worst = pilot[frontier[-1]]
    total = s0 - worst
    if total < n_levels * min_step or total <= 0:
        return None
    chosen: List[float] = []
    for i in range(1, n_levels + 1):
        target = s0 - total * i / n_levels
        opts = [p for p in frontier if not chosen or abs(p - clean_value) > abs(chosen[-1] - clean_value)]
        if not opts:
            return None
        chosen.append(min(opts, key=lambda p: abs(pilot[p] - target)))
    return chosen
