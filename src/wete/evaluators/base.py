"""Interface for an evaluator: anything that turns (policy, scenario) into a score.

This file is an interface only. No evaluator is implemented against it yet. The stage 1
reproduction job (``wete.cli.navsim_v1``) calls the unmodified NAVSIM devkit on purpose, so
that the installation is validated before any of our own code sits between a checkpoint and
its published score.

The unit of everything downstream (ordering recovery, bootstrap intervals, the reliability
model) is the per-scenario score, so an evaluator must return one record per scenario rather
than only an average.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from wete.policies.base import PolicyWrapper, SensorRequirements

# Evaluator families compared in the study.
FAMILIES = ("open_loop", "pseudo_sim_v1", "pseudo_sim_v2", "gaussian_splat", "world_model", "graphics_sim")


@dataclass(frozen=True)
class ScenarioScore:
    """Outcome of one policy on one scenario."""

    scenario_id: str
    score: float  # the evaluator's headline score for this scenario, higher is better
    valid: bool  # False if the evaluator could not score the scenario
    submetrics: Mapping[str, float] = field(default_factory=dict)  # e.g. NC, DAC, EP, TTC, comfort
    signals: Mapping[str, float] = field(default_factory=dict)  # cheap reliability signals, stage 5


@dataclass(frozen=True)
class EvaluationResult:
    """All scenario scores of one policy under one evaluator, with provenance."""

    evaluator_id: str
    policy_id: str
    scores: List[ScenarioScore]
    metadata: Dict[str, Any] = field(default_factory=dict)  # code versions, config, timings


class Evaluator(ABC):
    """Scores a policy on a fixed set of scenarios."""

    @property
    @abstractmethod
    def evaluator_id(self) -> str:
        """Stable identifier used in result files, e.g. ``navsim_v1_pdms``."""

    @property
    @abstractmethod
    def family(self) -> str:
        """One of :data:`FAMILIES`."""

    @abstractmethod
    def scenario_ids(self) -> Sequence[str]:
        """Identifiers of every scenario this evaluator can score."""

    @abstractmethod
    def supports(self, requirements: SensorRequirements) -> bool:
        """Whether a policy with these sensor requirements can be hosted."""

    @abstractmethod
    def evaluate(
        self,
        policy: PolicyWrapper,
        scenario_ids: Optional[Sequence[str]] = None,
    ) -> EvaluationResult:
        """Run ``policy`` on the given scenarios (all scenarios if ``None``)."""
