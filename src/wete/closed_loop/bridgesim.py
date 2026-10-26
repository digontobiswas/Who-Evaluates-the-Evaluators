"""Shared pieces for running BridgeSim's own scripts unmodified.

BridgeSim writes one ``driving_score_summary.csv`` per evaluated scenario; its last row,
``frame_id == AVERAGE``, holds the scenario's scores (fractions in [0, 1]). The leaderboard is the
plain mean of those rows over scenarios (scripts/sequential_batch_eval.sh at ea5d45f).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

# navsim_baselines family -> BridgeSim --model-type
MODEL_TYPES = {"transfuser": "transfuser", "ltf": "ltf", "ego_status_mlp": "egomlp"}

# Columns of the AVERAGE row, in BridgeSim's order. DS = EPDMS_no_ep x RC per scenario.
METRICS = ("NC", "DAC", "DDC", "TL", "TTC", "LK", "HC", "EC", "DS", "EPDMS_no_ep", "RC")


def evaluator_command(
    root: Path,
    model_type: str,
    checkpoint: Path,
    scenario_path: Path,
    output_dir: Path,
    run: Mapping[str, Any],
    extra: Iterable[str] = (),
) -> List[str]:
    """Command line for bridgesim/evaluation/unified_evaluator.py with the configured protocol."""
    return [
        sys.executable, str(Path(root) / "bridgesim/evaluation/unified_evaluator.py"),
        "--model-type", model_type,
        "--checkpoint", str(checkpoint),
        "--scenario-path", str(scenario_path),
        "--output-dir", str(output_dir),
        "--traffic-mode", str(run["traffic_mode"]),
        "--eval-mode", str(run["eval_mode"]),
        "--controller", str(run["controller"]),
        "--sim-dt", str(run["sim_dt"]),
        "--replan-rate", str(int(run["replan_rate"])),
        "--ego-replay-frames", str(int(run["ego_replay_frames"])),
        "--eval-frames", str(int(run["eval_frames"])),
        "--no-save-perframe",
        *extra,
    ]


def read_average_row(run_dir: Path) -> Optional[Dict[str, str]]:
    """The AVERAGE row of the newest driving_score_summary.csv below ``run_dir``, or None."""
    files = sorted(Path(run_dir).rglob("driving_score_summary.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    with open(files[-1], newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("frame_id") == "AVERAGE"]
    return rows[-1] if rows else None


def converted_scenarios(scenario_dir: Path) -> List[str]:
    """Scenario directories the converter finished: sd_<token>/sd_<token>_0/*.pkl."""
    return sorted(p.name for p in Path(scenario_dir).glob("sd_*") if list((p / f"{p.name}_0").glob("*.pkl")))


def read_scenario_list(path: Path) -> List[str]:
    """BridgeSim's converted_list files hold scenario names separated by any whitespace."""
    return sorted(set(Path(path).read_text().split()))
