"""Degradation relay between the HUGSIM environment and a driving policy.

Written against HUGSIM main @ 62c690d (2025-11-08).

HUGSIM protocol (closed_loop.py): every step the env pickles (obs, info) into
<output>/obs_pipe and then reads a plan (N x 2 numpy, LiDAR frame: x right,
y forward -- docstring of sim/utils/sim_utils.traj2control) from
<output>/plan_pipe; the string 'Done' ends the episode. The policy process is
launched as `zsh <ad_shell> <cuda_id> <output>`.

The relay takes the policy's place on the outer pipes and starts the real
policy on inner pipes:

  env --obs--> [outer/obs_pipe] relay --degraded obs--> [inner/obs_pipe] policy
  env <-plan-- [outer/plan_pipe] relay <--raw plan---- [inner/plan_pipe] policy

Neither HUGSIM nor the policy is modified. Launch via relay.sh (see README).

Limitations, stated rather than hidden
--------------------------------------
* Degradations are applied to obs['rgb'] only. 'semantic' and 'depth' are
  passed through unchanged (privileged channels; a policy that reads them is
  not a camera-only policy, and that must be reported).
* latency: the executed plan is the one produced `k` steps earlier, expressed
  in the ego frame of the time it was produced (no re-projection into the
  current ego frame, because info['ego_rot'] conventions are dataset
  dependent). With dt = 0.25 s and k <= 4 the ego moves <= ~10 m, so this is a
  plan-lag approximation of actuation latency; verify on 2-3 scenes by
  comparing with a re-projected variant before relying on it.
* Per-step ego poses are appended to <output>/wete_trace.jsonl so that the
  'deviation from logged path' reliability feature can be computed later.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import stat
import sys
from pathlib import Path

import numpy as np

from wete import degradations as D
from wete.wrappers import ClosedLoopDegrader

# nuScenes-style camera names used by HUGSIM, in dropout priority order
HUGSIM_DROPOUT_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def lidar_to_fl(plan: np.ndarray) -> np.ndarray:
    """HUGSIM plan (x right, y forward) -> (x forward, y left)."""
    return np.stack([plan[:, 1], -plan[:, 0]], axis=1)


def fl_to_lidar(traj: np.ndarray) -> np.ndarray:
    return np.stack([-traj[:, 1], traj[:, 0]], axis=1)


def ensure_fifo(path: Path) -> None:
    if path.exists():
        if not stat.S_ISFIFO(path.stat().st_mode):
            raise RuntimeError(f"{path} exists and is not a FIFO")
    else:
        os.mkfifo(path)


def read_msg(path: Path):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def write_msg(path: Path, obj) -> None:
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))


def relay(outer: Path, inner: Path, spec: D.DegradationSpec, scenario: str, step_dt: float) -> None:
    for p in (outer / "obs_pipe", outer / "plan_pipe", inner / "obs_pipe", inner / "plan_pipe"):
        ensure_fifo(p)
    deg = ClosedLoopDegrader(spec, scenario, step_dt=step_dt, camera_order=HUGSIM_DROPOUT_ORDER)
    trace = open(outer / "wete_trace.jsonl", "a")
    step = 0
    while True:
        msg = read_msg(outer / "obs_pipe")
        if isinstance(msg, str):            # 'Done'
            write_msg(inner / "obs_pipe", msg)
            break
        obs, info = msg
        obs = dict(obs)
        obs["rgb"] = deg.observe(obs["rgb"])
        write_msg(inner / "obs_pipe", (obs, info))
        plan = read_msg(inner / "plan_pipe")
        executed_fl = None
        if plan is not None:
            fl = lidar_to_fl(np.asarray(plan, dtype=np.float64))
            executed_fl = deg.act(fl)
            plan = fl_to_lidar(executed_fl).astype(np.asarray(plan).dtype)
        write_msg(outer / "plan_pipe", plan)
        rec = {"step": step, "ego_pos": info.get("ego_pos"), "timestamp": info.get("timestamp"),
               "collision": bool(info.get("collision", False))}
        if step == 0 and executed_fl is not None:
            # first executed plan (forward, left): every policy sees the same
            # initial state, so plan displacement vs. the clean run measures the
            # degradation's behavioural effect (undefined for stale_obs and
            # latency, which are identity at step 0)
            rec["plan0"] = np.round(executed_fl, 4).tolist()
        trace.write(json.dumps(rec) + "\n")
        trace.flush()
        step += 1
        if plan is None:
            break
    trace.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", required=True, help="HUGSIM output dir (env side)")
    ap.add_argument("--inner", required=True, help="dir handed to the real policy")
    ap.add_argument("--spec", default="none", help="e.g. camera_dropout@2 or none")
    ap.add_argument("--value", type=float, default=None, help="calibrated parameter (overrides level table)")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--step-dt", type=float, default=0.25, help="HUGSIM configs/sim/kinematic.yaml dt")
    a = ap.parse_args(argv)
    spec = D.DegradationSpec.parse(a.spec)
    if a.value is not None:
        spec = D.DegradationSpec(spec.degradation, spec.level, a.value)
    relay(Path(a.outer), Path(a.inner), spec, a.scenario, a.step_dt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
