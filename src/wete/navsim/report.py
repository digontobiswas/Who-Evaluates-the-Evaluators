"""Turn the devkit's PDM score CSV into a summary we can check against a published number.

The devkit writes one row per scenario plus a final row whose token is ``average``. Scenario
scores are fractions in [0, 1]; published PDMS values are the mean score times 100.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

AVERAGE_TOKEN = "average"

# Columns of navsim.common.dataclasses.PDMResults in NAVSIM v1.1.
SUBMETRIC_COLUMNS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
SCORE_COLUMN = "score"


class ReportError(RuntimeError):
    """Raised when the score file is missing or malformed."""


def find_score_csv(output_dir: Path) -> Path:
    """The devkit names the file after the wall-clock time; take the newest one."""
    candidates = sorted(output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise ReportError(f"No score CSV found in {output_dir}")
    return candidates[-1]


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Read the devkit CSV. The ``valid`` column becomes a proper boolean."""
    frame = pd.read_csv(csv_path, index_col=0, dtype={"token": str})
    required = {"token", "valid", SCORE_COLUMN, *SUBMETRIC_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ReportError(f"{csv_path} lacks columns {sorted(missing)}")
    frame["valid"] = frame["valid"].astype(str).str.strip().str.lower().eq("true")
    return frame


def summarize(scores: pd.DataFrame, expected_tokens: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Aggregate per-scenario scores and report every way the run could be incomplete."""
    per_scenario = scores[scores["token"] != AVERAGE_TOKEN]
    average_rows = scores[scores["token"] == AVERAGE_TOKEN]

    duplicated = int(per_scenario["token"].duplicated().sum())
    valid = per_scenario[per_scenario["valid"]]
    num_scored = int(len(per_scenario))
    num_valid = int(len(valid))

    means = {column: float(valid[column].mean()) for column in (*SUBMETRIC_COLUMNS, SCORE_COLUMN)} if num_valid else {}
    score_std = float(valid[SCORE_COLUMN].std(ddof=1)) if num_valid > 1 else float("nan")

    summary: Dict[str, Any] = {
        "num_scored": num_scored,
        "num_valid": num_valid,
        "num_invalid": num_scored - num_valid,
        "num_duplicated_tokens": duplicated,
        "mean": means,
        "pdms": 100.0 * means[SCORE_COLUMN] if means else float("nan"),
        "score_std": score_std,
        "score_sem": score_std / math.sqrt(num_valid) if num_valid > 1 else float("nan"),
    }

    if len(average_rows) == 1 and means:
        devkit_average = float(average_rows[SCORE_COLUMN].iloc[0])
        summary["devkit_average_score"] = devkit_average
        summary["matches_devkit_average"] = bool(abs(devkit_average - means[SCORE_COLUMN]) < 1e-9)

    if expected_tokens is not None:
        expected = set(expected_tokens)
        scored = set(per_scenario["token"])
        summary["num_expected"] = len(expected)
        summary["num_missing"] = len(expected - scored)
        summary["num_unexpected"] = len(scored - expected)
    else:
        summary["num_expected"] = None
        summary["num_missing"] = None
        summary["num_unexpected"] = None

    summary["complete"] = bool(
        num_valid > 0
        and summary["num_invalid"] == 0
        and duplicated == 0
        and not summary["num_missing"]
        and not summary["num_unexpected"]
    )
    return summary


def compare_to_reference(
    summary: Mapping[str, Any],
    reference: Mapping[str, Any],
    tolerance_pdms: float,
) -> Dict[str, Any]:
    """Compare a complete full-split run with a published PDMS.

    A verdict is given only for a complete run: an average over a subset of the split, or over
    the scenarios that happened not to fail, is not the published quantity.
    """
    reference_pdms = float(reference["pdms"])
    delta = float(summary["pdms"]) - reference_pdms
    comparable = bool(summary["complete"])
    return {
        "reference_pdms": reference_pdms,
        "reference_source": reference.get("source"),
        "reference_note": reference.get("note"),
        "delta_pdms": delta,
        "tolerance_pdms": float(tolerance_pdms),
        "comparable": comparable,
        "within_tolerance": bool(abs(delta) <= tolerance_pdms) if comparable else None,
    }


def summarize_plans(plans_dir: Path) -> Dict[str, Any]:
    """Summarise the plan log written by ``DegradedAgent``: open-loop L2 and coverage.

    The log has one line per scored token; a token can appear twice if a worker retried it, in
    which case the last line counts. Means are over tokens that have a logged future.
    """
    import json

    files = sorted(Path(plans_dir).glob("*.plans.jsonl")) if Path(plans_dir).is_dir() else []
    if not files:
        return {"num_tokens": 0, "note": f"no plan log in {plans_dir}"}
    records: Dict[str, Dict[str, Any]] = {}
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records[str(record["token"])] = record
    with_l2 = [r for r in records.values() if "ade" in r]
    out: Dict[str, Any] = {"files": [p.name for p in files], "num_tokens": len(records), "num_with_l2": len(with_l2)}
    for key in ("ade", "fde", "l2_1s", "l2_2s", "l2_3s"):
        values = [r[key] for r in with_l2 if key in r]
        out[f"mean_{key}"] = float(sum(values) / len(values)) if values else float("nan")
    effective = {r.get("effective_k") for r in records.values() if r.get("effective_k") is not None}
    if effective:
        out["effective_k"] = sorted(effective)
    return out
