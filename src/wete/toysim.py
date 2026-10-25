"""Toy driving world with PLANTED evaluator blind spots.

PURPOSE: validate the analysis pipeline, NOT to produce findings. Every
evaluator here has properties chosen by us, so any 'result' on this data is a
property of this file. Never report toy numbers as evidence about NAVSIM,
HUGSIM or world models.

World
-----
Straight road, ego follows a lead vehicle that brakes at a random time
(30 % of scenarios have no lead). Lane centre y = 0. dt = 0.1 s. Vehicles are
4.5 m long; `gap` is bumper-to-bumper.

Policies
--------
IDM longitudinal planner + exponential return-to-centre lateral planner,
replanning every 0.5 s with a 4 s / 8-waypoint plan (NAVSIM-like). Perception:
lead detected with probability p_det; gap measured with Gaussian noise; lead
speed estimated from up to 3 past frames at 2 Hz (NAVSIM keeps 1.5 s of
history). Policies with uses_history=False (like single-frame TransFuser)
assume zero relative speed.

Evaluators (all scores in [0,1] except L2) and their PLANTED properties
-----------------------------------------------------------------------
closed_ref   reactive-free closed loop, 15 s, faithful perception     (E*, truth)
gs3d         closed loop, perception noise grows with the ego's deviation from
             the logged path x per-scene reconstruction quality kappa_s
             -> per-scenario reliability varies; cheap proxy = deviation
world_model  closed loop with faithful rendering BUT (i) generated lead flees
             when gap < 6 m and closing (never-seen-collisions bias) and
             (ii) 'road follows the car': lateral position is not evaluated,
             (iii) fresh generation noise per rollout scaled by (1 - c_s)
             -> blind to lateral_noise BY CONSTRUCTION (tau must be ~0)
nonreactive  one 4 s rollout from a logged state, lead follows the log (NAVSIM v1)
pseudo_sim2  stage 1 = nonreactive; stage 2 re-plans from the stage-1 endpoint
             with noisier 'synthesised views'; score = s1 * s2 (NAVSIM v2-like)
open_l2      ADE of the plan vs the logged expert plan (lower is better)
null_random  i.i.d. U(0,1) scores: NULL CONTROL, tau CI must cover 0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .degradations import CLOSED_LOOP_ONLY, LEVELS, DegradationSpec, population_a_specs

DT = 0.1
REPLAN = 5                # steps (0.5 s)
H_PLAN = 8                # waypoints at 0.5 s -> 4 s
T_LOG = 300               # 30 s of log
H_CL = 150                # closed-loop horizon, 15 s
VLEN = 4.5
NO_LEAD = 1e6
HIST_K = 40               # frames of logged history available to single-shot evaluators (>= 1.5 s stale + 1.5 s history)
CAM_FACTOR = {0: 1.0, 1: 0.65, 2: 0.45, 3: 0.3, 4: 0.2}   # detection prob. multiplier


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


# --------------------------------------------------------------------------
# Scenarios and expert log
# --------------------------------------------------------------------------

@dataclass
class Scenarios:
    n: int
    ids: List[str]
    has_lead: np.ndarray
    v_des: np.ndarray
    x_lead: np.ndarray          # (n, T_LOG) lead rear-bumper position minus VLEN offset handled in gap
    v_lead: np.ndarray          # (n, T_LOG)
    x_log: np.ndarray           # (n, T_LOG) expert ego position
    v_log: np.ndarray
    k_anchor: np.ndarray        # (n,) anchor index for single-shot evaluators
    kappa: np.ndarray           # gs3d reconstruction-quality factor
    wm_consist: np.ndarray      # world-model temporal consistency c_s in (0,1)
    z_gap: np.ndarray           # (n, T_LOG) common random numbers for gap noise
    u_det: np.ndarray           # (n, T_LOG) common random numbers for detection
    lat_sign: np.ndarray        # (n,) common lateral drift direction


def make_scenarios(n: int = 400, seed: int = 0) -> Scenarios:
    rng = np.random.default_rng(seed)
    has_lead = rng.random(n) > 0.3
    g0 = rng.uniform(15, 40, n)
    vl0 = rng.uniform(8, 15, n)
    t_b = rng.uniform(4, 12, n)
    a_b = rng.uniform(2, 7, n)
    v_end = rng.uniform(0, 0.5, n) * vl0
    v_des = rng.uniform(12, 16, n)
    v0 = np.clip(vl0 + rng.uniform(-2, 2, n), 4, None)

    t = np.arange(T_LOG) * DT
    vl = np.empty((n, T_LOG))
    vl[:, 0] = vl0
    for k in range(1, T_LOG):
        braking = (t[k] >= t_b) & (vl[:, k - 1] > v_end)
        vl[:, k] = np.where(braking, np.maximum(v_end, vl[:, k - 1] - a_b * DT), vl[:, k - 1])
    xl = g0[:, None] + VLEN + np.concatenate([np.zeros((n, 1)), np.cumsum(vl[:, :-1] * DT, 1)], 1)
    xl[~has_lead] = NO_LEAD
    vl[~has_lead] = v_des[~has_lead, None]

    # expert: IDM, perfect perception, cautious-normal parameters
    xe = np.zeros((n, T_LOG))
    ve = np.zeros((n, T_LOG))
    ve[:, 0] = v0
    for k in range(1, T_LOG):
        gap = xl[:, k - 1] - xe[:, k - 1] - VLEN
        a = idm(ve[:, k - 1], v_des, gap, vl[:, k - 1], T=1.5, a_max=1.5, b=2.0)
        ve[:, k] = np.maximum(0, ve[:, k - 1] + a * DT)
        xe[:, k] = xe[:, k - 1] + ve[:, k - 1] * DT
    k_anchor = np.clip(((t_b + rng.uniform(-2.0, 1.0, n)) / DT).astype(int), 20, T_LOG - 85)
    k_anchor[~has_lead] = rng.integers(20, T_LOG - 85, (~has_lead).sum())
    return Scenarios(
        n=n, ids=[f"s{i:04d}" for i in range(n)], has_lead=has_lead, v_des=v_des,
        x_lead=xl, v_lead=vl, x_log=xe, v_log=ve, k_anchor=k_anchor,
        kappa=0.6 * np.exp(rng.normal(0, 0.7, n)),
        wm_consist=rng.beta(4, 2, n),
        z_gap=rng.standard_normal((n, T_LOG)), u_det=rng.random((n, T_LOG)),
        lat_sign=np.where(rng.random(n) < 0.5, -1.0, 1.0))


def idm(v, v_des, gap, v_lead, T, a_max, b, s0=2.0):
    s_star = s0 + np.maximum(0, v * T + v * (v - v_lead) / (2 * np.sqrt(a_max * b)))
    inter = np.where(gap < NO_LEAD / 2, (s_star / np.maximum(gap, 0.1)) ** 2, 0.0)
    return np.clip(a_max * (1 - (v / v_des) ** 4 - inter), -8.0, a_max)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    name: str
    T: float = 1.5
    a_max: float = 1.5
    b: float = 2.0
    perc_sigma: float = 0.3
    p_det: float = 0.99
    uses_history: bool = True
    deg: DegradationSpec = field(default_factory=DegradationSpec)

    @property
    def pid(self) -> str:
        return f"{self.name}:{self.deg.policy_suffix}"

    def p(self, d: str) -> float:
        if self.deg.degradation == d:
            return self.deg.param
        return LEVELS[d][0]


BASES = [
    Policy("cautious", T=1.9, a_max=1.2, b=1.8),
    Policy("normal", T=1.4, a_max=1.6, b=2.2),
    Policy("single_frame", T=1.5, a_max=1.5, b=2.0, uses_history=False),   # TransFuser analogue
]


def natural_policies() -> List[Policy]:
    rng = np.random.default_rng(123)
    out = []
    for i in range(8):
        out.append(Policy(f"nat{i}", T=rng.uniform(1.0, 2.2), a_max=rng.uniform(1.0, 2.2),
                          b=rng.uniform(1.5, 3.0), perc_sigma=rng.uniform(0.2, 2.0),
                          p_det=rng.uniform(0.6, 0.99), uses_history=bool(rng.random() > 0.3)))
    return out


def checkpoint_policies(n_ckpt: int = 6) -> List[Policy]:
    out = []
    for e in range(n_ckpt):
        f = e / (n_ckpt - 1)
        out.append(Policy(f"ckpt_e{e}", T=2.6 - 1.1 * f, a_max=1.0 + 0.5 * f, b=1.5 + 0.5 * f,
                          perc_sigma=2.5 - 2.2 * f, p_det=0.55 + 0.43 * f, uses_history=e >= 2))
    return out


# --------------------------------------------------------------------------
# Perception and planning (vectorised over scenarios)
# --------------------------------------------------------------------------

def sigma_and_pdet(pol: Policy, n: int, extra_sigma=0.0, pdet_mult=1.0):
    s_img = pol.p("image_noise")
    sigma = np.sqrt((pol.perc_sigma + 0.04 * s_img) ** 2 + np.asarray(extra_sigma) ** 2) * np.ones(n)
    n_drop = int(pol.p("camera_dropout"))
    pd_ = pol.p_det * CAM_FACTOR[n_drop] * np.exp(-s_img / 120.0) * pdet_mult * np.ones(n)
    return sigma, pd_


def perceive(pol: Policy, gm: np.ndarray, det: np.ndarray, k: int, v_ego: np.ndarray):
    """gm, det: (n, K) measured gap / detection history (index = time step).
    Returns detected (n,), gap estimate, lead-speed estimate."""
    ks = max(0, k - int(round(pol.p("stale_obs") / DT)))
    detected = det[:, ks]
    g = gm[:, ks]
    kept = int(round(pol.p("history_removal") * 3)) if pol.uses_history else 0
    if kept > 0:
        ko = max(0, ks - REPLAN * kept)
        span = (ks - ko) * DT
        vrel = (g - gm[:, ko]) / span if span > 0 else 0.0
        vrel = np.where(det[:, ko], vrel, 0.0)
        vl = np.maximum(0.0, v_ego + vrel)
    else:
        vl = v_ego.copy()
    return detected, g, vl


def plan(pol: Policy, v0, v_des, detected, g_est, vl_est, y0, lat_sign):
    """Returns (x_plan (n,8), y_plan (n,8)) relative longitudinal / absolute lateral."""
    n = len(v0)
    x = np.zeros(n)
    v = v0.copy()
    xl = np.where(detected, g_est + VLEN, NO_LEAD)
    vl = np.where(detected, vl_est, v_des)
    xs = []
    for i in range(1, 41):
        gap = xl - x - VLEN
        a = idm(v, v_des, gap, vl, pol.T, pol.a_max, pol.b)
        x = x + v * DT + 0.5 * a * DT ** 2
        v = np.maximum(0, v + a * DT)
        xl = np.where(xl < NO_LEAD / 2, xl + vl * DT, xl)
        if i % 5 == 0:
            xs.append(x.copy())
    xp = np.stack(xs, 1)
    tw = np.arange(1, H_PLAN + 1) * 0.5
    yp = y0[:, None] * np.exp(-tw / 1.5)
    yp = yp + lat_sign[:, None] * pol.p("lateral_noise") * (tw / tw[-1])
    xp = xp * (1.0 + pol.p("late_braking"))      # arc-length stretch on a straight road
    return xp, yp


# --------------------------------------------------------------------------
# Scoring (PDMS-like) -- shared by all simulated evaluators
# --------------------------------------------------------------------------

def pdm_like(gap, ve, vl, y, a, dist, exp_dist, has_lead, lateral=True):
    closing = ve - vl > 0.1
    ttc = np.where(closing, gap / np.maximum(ve - vl, 1e-3), np.inf)
    col = (has_lead[:, None] & (gap <= 0)).any(1)
    ttc_ok = ~(has_lead[:, None] & closing & (ttc < 1.0) & (gap > 0)).any(1)
    prog = np.clip(dist / np.maximum(exp_dist, 1.0), 0, 1)
    comf = (np.abs(a) <= 4.0).mean(1)
    if lateral:
        drv = (np.abs(y) < 2.5).all(1)
        # continuous lane-centering term; a binary |y|<0.9 test is blind to the
        # ~0.1-0.7 m steady offsets that closed-loop replanning leaves behind
        lane = np.clip(1 - np.abs(y) / 1.75, 0, 1).mean(1)
    else:
        drv = np.ones(len(gap), bool)
        lane = np.ones(len(gap))
    return (~col) * drv * (5 * ttc_ok + 5 * prog + 2 * comf + 2 * lane) / 14.0


# --------------------------------------------------------------------------
# Closed loop (closed_ref, gs3d, world_model)
# --------------------------------------------------------------------------

def closed_loop(sc: Scenarios, pol: Policy, mode: str = "ref"):
    n = sc.n
    x = np.zeros(n)
    v = sc.v_log[:, 0].copy()
    y = np.zeros(n)
    xl = sc.x_lead[:, 0].copy()
    vl_t = sc.v_lead.copy()
    gm = np.zeros((n, H_CL + 1))
    det = np.zeros((n, H_CL + 1), bool)
    G, VE, VL, Y, A = (np.zeros((n, H_CL)) for _ in range(5))
    collided = np.zeros(n, bool)
    delay = int(pol.p("latency"))
    queue: List[Tuple[np.ndarray, np.ndarray]] = []
    wm_rng = np.random.default_rng(_seed("wm", pol.pid))
    render_rng = np.random.default_rng(_seed("render", pol.pid))
    dev_acc = np.zeros(n)
    xp = yp = None
    for k in range(H_CL):
        gap = xl - x - VLEN
        # rendering / generation
        sigma, pdet = sigma_and_pdet(pol, n)
        gm[:, k] = gap + sigma * sc.z_gap[:, k]
        det[:, k] = sc.has_lead & (gap < 60) & (sc.u_det[:, k] < pdet)
        if mode == "gs3d":
            # Rendering artefacts grow with distance from the recorded path.
            # They are drawn from a rollout-specific RNG (NOT the common random
            # numbers): two policies that drive differently see different
            # artefacts, so artefacts do not cancel in paired level comparisons.
            dev = np.abs(y) + np.abs(x - sc.x_log[:, k]) / 8.0
            dev_acc += dev
            gm[:, k] += 2.0 * sc.kappa * dev * render_rng.standard_normal(n)
            det[:, k] &= render_rng.random(n) < np.exp(-0.3 * sc.kappa * dev)
        if k % REPLAN == 0:
            d_, g_, v_ = perceive(pol, gm, det, k, v)
            xp, yp = plan(pol, v, sc.v_des, d_, g_, v_, y, sc.lat_sign)
            queue.append((xp, yp))
            if len(queue) > delay + 1:
                queue.pop(0)
            age = len(queue) - 1        # periods since the executed plan was made
        exp_xp, exp_yp = queue[0]
        # executing during [t_p + 0.5*age, t_p + 0.5*(age+1)): waypoint `age`
        idx = min(age, H_PLAN - 1)
        prev = exp_xp[:, idx - 1] if idx > 0 else np.zeros(n)
        v_tgt = (exp_xp[:, idx] - prev) / 0.5
        a = np.clip(2.0 * (v_tgt - v), -8.0, 4.0)
        a = np.where(collided, 0.0, a)
        y_tgt = exp_yp[:, idx]
        # lead motion
        vlk = vl_t[:, k].copy()
        if mode == "world_model":
            flee = sc.has_lead & (gap < 6.0) & (v > vlk)
            vlk = np.where(flee, v + 1.0, vlk)
            vl_t[:, k:] = np.where(flee[:, None], np.maximum(vl_t[:, k:], vlk[:, None]), vl_t[:, k:])
            if k % REPLAN == 0:
                xl = xl + np.where(sc.has_lead, (1 - sc.wm_consist) * 1.5 * wm_rng.standard_normal(n), 0.0)
        G[:, k], VE[:, k], VL[:, k], Y[:, k], A[:, k] = gap, v, vlk, y, a
        collided |= sc.has_lead & (gap <= 0)
        x = x + np.where(collided, 0.0, v * DT)
        v = np.where(collided, 0.0, np.maximum(0, v + a * DT))
        y = y + (y_tgt - y) * DT / 0.5
        xl = np.where(sc.has_lead, xl + vlk * DT, xl)
    exp_dist = sc.x_log[:, H_CL] - sc.x_log[:, 0]
    score = pdm_like(G, VE, VL, Y, A, x, exp_dist, sc.has_lead, lateral=(mode != "world_model"))
    info = {"deviation": dev_acc / H_CL}
    return score, info


# --------------------------------------------------------------------------
# Single-shot evaluators (open_l2, nonreactive, pseudo_sim2)
# --------------------------------------------------------------------------

def _logged_history(sc: Scenarios, pol: Policy, k_a: np.ndarray, extra_sigma=0.0, pdet_mult=1.0,
                    gap_override: Optional[np.ndarray] = None):
    """Measured gap/detection history on the log up to each scenario's anchor,
    re-indexed so the anchor is the last column (K = 16 frames)."""
    n = sc.n
    K = HIST_K
    cols = k_a[:, None] + np.arange(-K + 1, 1)[None, :]
    cols = np.clip(cols, 0, T_LOG - 1)
    rows = np.arange(n)[:, None]
    gap = sc.x_lead[rows, cols] - sc.x_log[rows, cols] - VLEN if gap_override is None else gap_override
    sigma, pdet = sigma_and_pdet(pol, n, extra_sigma, pdet_mult)
    gm = gap + sigma[:, None] * sc.z_gap[rows, cols]
    det = sc.has_lead[:, None] & (gap < 60) & (sc.u_det[rows, cols] < pdet[:, None])
    return gm, det, K - 1


def _plan_at(sc, pol, k_a, v0, y0, **kw):
    gm, det, k = _logged_history(sc, pol, k_a, **kw)
    d_, g_, v_ = perceive(pol, gm, det, k, v0)
    return plan(pol, v0, sc.v_des, d_, g_, v_, y0, sc.lat_sign)


def _score_plan(sc, xp, yp, k0, x0, has_lead):
    """Non-reactive scoring of an 8-waypoint plan starting at log index k0
    from ego position x0 (lead follows the log)."""
    n = sc.n
    rows = np.arange(n)
    ks = np.clip(k0[:, None] + REPLAN * np.arange(1, H_PLAN + 1)[None, :], 0, T_LOG - 1)
    xe = x0[:, None] + xp
    gap = sc.x_lead[rows[:, None], ks] - xe - VLEN
    vel = np.diff(np.concatenate([np.zeros((n, 1)), xp], 1), axis=1) / 0.5
    acc = np.diff(np.concatenate([vel[:, :1], vel], 1), axis=1) / 0.5
    vl = sc.v_lead[rows[:, None], ks]
    exp_dist = sc.x_log[rows, ks[:, -1]] - sc.x_log[rows, k0]
    return pdm_like(gap, vel, vl, yp, acc, xp[:, -1], exp_dist, has_lead)


def single_shot(sc: Scenarios, pol: Policy):
    n = sc.n
    rows = np.arange(n)
    k_a = sc.k_anchor
    v0 = sc.v_log[rows, k_a]
    y0 = np.zeros(n)
    xp, yp = _plan_at(sc, pol, k_a, v0, y0)
    # open-loop L2 vs expert 4 s future
    ks = k_a[:, None] + REPLAN * np.arange(1, H_PLAN + 1)[None, :]
    x_exp = sc.x_log[rows[:, None], ks] - sc.x_log[rows, k_a][:, None]
    l2 = np.sqrt((xp - x_exp) ** 2 + yp ** 2).mean(1)
    s1 = _score_plan(sc, xp, yp, k_a, sc.x_log[rows, k_a], sc.has_lead)
    # stage 2: from stage-1 endpoint, synthesised-view perception
    k2 = k_a + 40
    x2 = sc.x_log[rows, k_a] + xp[:, -1]
    v2 = np.maximum(0, (xp[:, -1] - xp[:, -2]) / 0.5)
    # history consistent with the stage-1 rollout (ego along plan, lead on log)
    K = HIST_K
    tcols = np.arange(-K + 1, 1)
    xe_hist = np.empty((n, K))
    tt = (tcols + K - 1) * DT                      # (K-1)*DT s before stage-2 start .. 0, mapped onto the plan
    t_plan = np.arange(0, H_PLAN + 1) * 0.5
    xp0 = np.concatenate([np.zeros((n, 1)), xp], 1)
    for i in range(n):
        xe_hist[i] = sc.x_log[i, k_a[i]] + np.interp(4.0 - (K - 1) * DT + tt, t_plan, xp0[i])
    lcols = np.clip(k2[:, None] + tcols[None, :], 0, T_LOG - 1)
    gap_hist = sc.x_lead[rows[:, None], lcols] - xe_hist - VLEN
    xp2, yp2 = _plan_at(sc, pol, k2, v2, yp[:, -1], extra_sigma=1.0, pdet_mult=0.9, gap_override=gap_hist)
    s2 = _score_plan(sc, xp2, yp2, k2, x2, sc.has_lead)
    return {"open_l2": l2, "nonreactive": s1, "pseudo_sim2": s1 * s2, "_stage_gap": np.abs(s1 - s2),
            "_plan": np.concatenate([xp, yp], 1)}


# --------------------------------------------------------------------------
# Driver: build the long-format table + features
# --------------------------------------------------------------------------

META = {
    "open_l2": {"higher_is_better": False, "family": "open_loop"},
    "nonreactive": {"higher_is_better": True, "family": "nonreactive"},
    "pseudo_sim2": {"higher_is_better": True, "family": "pseudo_sim"},
    "gs3d": {"higher_is_better": True, "family": "closed_loop"},
    "world_model": {"higher_is_better": True, "family": "world_model"},
    "closed_ref": {"higher_is_better": True, "family": "closed_loop", "reference": True},
    "null_random": {"higher_is_better": True, "family": "null"},
    # POSITIVE CONTROL for reliability prediction (not an evaluator analogue):
    # equals closed_ref, except that with probability p_s (a known function of
    # a per-scenario 'artifact index') the score is replaced by noise.
    "positive_control": {"higher_is_better": True, "family": "control"},
}


def evaluate_policy(sc: Scenarios, pol: Policy) -> Tuple[Dict[str, np.ndarray], Dict]:
    out: Dict[str, np.ndarray] = {}
    aux: Dict = {}
    if pol.deg.degradation not in CLOSED_LOOP_ONLY:
        ss = single_shot(sc, pol)
        aux["stage_gap"] = ss.pop("_stage_gap")
        aux["plan"] = ss.pop("_plan")             # executed (post-degradation) plan at the anchor
        out.update(ss)
        rng = np.random.default_rng(_seed("null", pol.pid))
        out["null_random"] = rng.random(sc.n)
    out["closed_ref"], _ = closed_loop(sc, pol, "ref")
    out["gs3d"], info = closed_loop(sc, pol, "gs3d")
    aux["deviation"] = info["deviation"]
    out["world_model"], _ = closed_loop(sc, pol, "world_model")
    return out, aux


PILOT_GRIDS = {
    "camera_dropout": [0, 1, 2, 3, 4],
    "image_noise": [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80],
    "stale_obs": [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
    "latency": [0, 1, 2, 3, 4, 5, 6],
    "history_removal": [1.0, 0.66, 0.33, 0.0],
    "lateral_noise": [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "late_braking": [0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.7],
}


def pilot_calibration(n_pilot: int = 150, seed: int = 10_000) -> Dict[str, object]:
    """Sweep each degradation on PILOT scenarios (different seed, never reused)
    with the reference evaluator, averaged over base policies, and choose levels
    with `calibrate_levels`. Returns {'levels': {deg: [clean, l1, l2, l3]},
    'uncalibratable': [...], 'sweeps': {...}}."""
    from .degradations import calibrate_levels
    sc = make_scenarios(n_pilot, seed)
    levels, bad, sweeps = {}, [], {}
    for d, grid in PILOT_GRIDS.items():
        clean = float(LEVELS[d][0])
        per_base, sweeps[d] = [], {}
        for b in BASES:
            means = {}
            for p in grid:
                spec = DegradationSpec(d, 1, value=float(p)) if p != clean else DegradationSpec()
                s, _ = closed_loop(sc, replace(b, deg=spec), "ref")
                means[float(p)] = float(s.mean())
            sweeps[d][b.name] = means
            per_base.append(calibrate_levels(means, clean, 3, min_step=0.01))
        if any(c is None for c in per_base):
            bad.append(d)          # some base never shows a drop: report, do not force
            continue
        # most severe choice per level, so that every base shows a drop
        chosen = [max(vals, key=lambda v: abs(v - clean)) for vals in zip(*per_base)]
        if len(set(chosen)) < 3:
            bad.append(d)
            continue
        levels[d] = [clean] + chosen
    return {"levels": levels, "uncalibratable": bad, "sweeps": sweeps}


def generate(n_scen: int = 400, seed: int = 0,
             levels: Optional[Dict[str, List[float]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """levels: calibrated level table (from pilot_calibration). Degradations
    missing from it fall back to the proposal's default table, so that the
    monotonicity test can still report them as discarded."""
    sc = make_scenarios(n_scen, seed)
    rows = []
    feats_src = {}

    def add(pop, base, deg, level, pid, res):
        for ev, s in res.items():
            rows.append(pd.DataFrame({"evaluator": ev, "population": pop, "base": base,
                                      "degradation": deg, "level": level, "policy": pid,
                                      "scenario": sc.ids, "score": s}))

    plans: Dict[Tuple[str, str, int], np.ndarray] = {}
    for b in BASES:
        for spec in population_a_specs(levels=levels):
            pol = replace(b, deg=spec)
            res, aux = evaluate_policy(sc, pol)
            add("A", b.name, spec.degradation, spec.level, pol.pid, res)
            if "plan" in aux:
                plans[(b.name, spec.degradation, spec.level)] = aux["plan"]
            if spec.is_identity and b.name == BASES[0].name:
                feats_src = aux
    # behaviour deltas: mean waypoint displacement, level-3 plan vs clean plan,
    # same logged input (evaluator-independent effect indicator)
    beh_rows = []
    for (bn, d, lvl), P in plans.items():
        if d == "none" or lvl != 3:
            continue
        C = plans[(bn, "none", 0)]
        dx = P[:, :H_PLAN] - C[:, :H_PLAN]
        dy = P[:, H_PLAN:] - C[:, H_PLAN:]
        beh_rows.append(pd.DataFrame({"base": bn, "degradation": d, "scenario": sc.ids,
                                      "delta": np.sqrt(dx ** 2 + dy ** 2).mean(1)}))
    for p in natural_policies():
        res, _ = evaluate_policy(sc, p)
        add("B", p.name, "none", 0, p.pid, res)
    for e, p in enumerate(checkpoint_policies()):
        res, _ = evaluate_policy(sc, p)
        add("C", "ckpt", "none", e, p.pid, res)
    scores = pd.concat(rows, ignore_index=True)

    # positive control: corrupt closed_ref scores per (policy, scenario) with a
    # scenario-dependent probability p_s = sigmoid(6 (a_s - 0.5)), a_s ~ U(0,1)
    arng = np.random.default_rng(_seed("artifact", seed))
    art = arng.random(sc.n)
    p_s = 1 / (1 + np.exp(-6 * (art - 0.5)))
    pc = scores[scores.evaluator == "closed_ref"].copy()
    pos = pc.scenario.map({s_: i for i, s_ in enumerate(sc.ids)}).values
    u = np.array([np.random.default_rng(_seed("pc", pid)).random(sc.n) for pid in pc.policy.unique()])
    pol_idx = pc.policy.map({p_: i for i, p_ in enumerate(pc.policy.unique())}).values
    corrupt = u[pol_idx, pos] < p_s[pos]
    noise = np.random.default_rng(_seed("pc_noise", seed)).random(len(pc))
    pc["score"] = np.where(corrupt, noise, pc.score.values)
    pc["evaluator"] = "positive_control"
    scores = pd.concat([scores, pc], ignore_index=True)

    frng = np.random.default_rng(_seed("features", seed))
    rows0 = np.arange(sc.n)
    speed = sc.v_log[rows0, sc.k_anchor]
    common = {"speed": speed, "has_lead": sc.has_lead.astype(float),
              "noise_feature": frng.standard_normal(sc.n)}          # negative control
    fr = []
    spec_feats = {
        "gs3d": {"deviation": feats_src["deviation"]},
        "world_model": {"consistency": np.clip(sc.wm_consist + 0.1 * frng.standard_normal(sc.n), 0, 1)},
        "pseudo_sim2": {"stage_gap": feats_src["stage_gap"]},
        "nonreactive": {}, "open_l2": {}, "closed_ref": {},
        "positive_control": {"artifact_index": np.clip(art + 0.1 * frng.standard_normal(sc.n), 0, 1)},
    }
    for ev, extra in spec_feats.items():
        fr.append(pd.DataFrame({"evaluator": ev, "scenario": sc.ids, **extra, **common}))
    features = pd.concat(fr, ignore_index=True)
    behavior = pd.concat(beh_rows, ignore_index=True) if beh_rows else pd.DataFrame()
    return scores, features, {"scenarios": sc, "behavior": behavior}
