"""Interface for a driving policy as an evaluator sees it.

This file is an interface only. No policy is implemented here yet; concrete wrappers
(a pretrained NAVSIM agent, and the graded degradations around it) arrive in stage 2.

Convention the harness is built around
--------------------------------------
Every policy in the study is a NAVSIM agent, so the common currency between evaluators and
policies is NAVSIM's own pair of types: an ``AgentInput`` goes in (ego status, cameras and
LiDAR for the last few frames) and a ``Trajectory`` comes out (BEV poses x, y, heading in the
ego frame). Each evaluator adapter is responsible for converting its native observation into
an ``AgentInput``. The types are referred to as ``Any`` here so that importing this module
does not require the NAVSIM devkit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

Observation = Any  # navsim.common.dataclasses.AgentInput
Plan = Any  # navsim.common.dataclasses.Trajectory


@dataclass(frozen=True)
class SensorRequirements:
    """What a policy needs from an evaluator at every planning step.

    An evaluator uses this to decide whether it can host the policy at all. For example a
    simulator that renders no LiDAR cannot host a policy with ``lidar=True``.
    """

    cameras: Tuple[str, ...]  # NAVSIM camera names, e.g. ("cam_l0", "cam_f0", "cam_r0")
    lidar: bool
    history_frames: int  # number of frames including the current one (1 = single frame)
    frame_interval_s: float  # spacing of history frames in seconds (NAVSIM logs are 2 Hz)


class PolicyWrapper(ABC):
    """A policy under test, optionally with a degradation applied around it."""

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Stable identifier used in result files, e.g. ``transfuser_seed0/camera_dropout/2``."""

    @abstractmethod
    def sensor_requirements(self) -> SensorRequirements:
        """Sensors and history the evaluator must provide."""

    @abstractmethod
    def initialize(self) -> None:
        """Load weights. Called once per worker process, before the first :meth:`plan`."""

    @abstractmethod
    def reset(self, scenario_id: str) -> None:
        """Clear per-episode state (history buffers, latency queues) at the start of a scenario."""

    @abstractmethod
    def plan(self, observation: Observation) -> Plan:
        """Return the planned trajectory for one observation."""
