"""BridgeSim closed-loop runs: convert scenarios, evaluate a policy in shards, report.

Usage (normally through ``jobs/bridgesim.sbatch``)::

    python -m wete.cli.bridgesim_run --config <experiment.yaml> --action convert
    python -m wete.cli.bridgesim_run --config <experiment.yaml> --action evaluate --shard 0 --num-shards 4
    python -m wete.cli.bridgesim_run --config <experiment.yaml> --action report

convert   run BridgeSim's OpenScene converter over the whole scene filter (once per split; it is
          skipped when every expected scenario is already converted)
evaluate  run unified_evaluator.py on the scenarios of one shard, several at a time. Resumable:
          scenarios whose summary already exists are not run again
report    collect every scenario's AVERAGE row, average them the way BridgeSim's leaderboard
          does, and compare with the published value only when every expected scenario scored

Nothing here re-implements BridgeSim: its scripts run unmodified as subprocesses.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from wete import config as config_lib
from wete import provenance
from wete.closed_loop import bridgesim

ACTIONS = ("convert", "evaluate", "report")
REQUIRED_KEYS = (
    "evaluator.root",
    "evaluator.navsim_log_path",
    "evaluator.map_root",
    "evaluator.scene_filter",
    "evaluator.run.replan_rate",
    "scenarios.dir",
    "scenarios.expected_list",
    "policy.family",
    "policy.id",
    "policy.checkpoint",
    "experiment.name",
    "experiment.out_dir",
    "run_parallel",
)


def _fail(message: str) -> SystemExit:
    return SystemExit(f"[bridgesim] ERROR: {message}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def expected_scenarios(cfg: Dict[str, Any]) -> List[str]:
    return bridgesim.read_scenario_list(Path(cfg["scenarios"]["expected_list"]))


# ---------------------------------------------------------------------------------------------
def action_convert(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = cfg["evaluator"]
    scenario_dir = Path(cfg["scenarios"]["dir"])
    expected = set(expected_scenarios(cfg))
    done = set(bridgesim.converted_scenarios(scenario_dir))
    if expected and expected <= done:
        print(f"[bridgesim] convert: all {len(expected)} expected scenarios already in {scenario_dir}, skipping.")
        return {"skipped": True, "num_converted": len(done)}

    command = [
        sys.executable, str(Path(ev["root"]) / "converters/openscene/convert_openscene_with_filter.py"),
        "--scene-filter", str(ev["scene_filter"]),
        "--input-dir", str(ev["navsim_log_path"]),
        "--output-dir", str(scenario_dir),
        "--map-root", str(ev["map_root"]),
        "--num-future-frames-extract", str(int(ev["conversion"]["num_future_frames_extract"])),
    ]
    if ev["conversion"].get("interpolate"):
        command.append("--interpolate")
    log_path = scenario_dir.parent / f"convert_{scenario_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bridgesim] convert: {' '.join(command)}", flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ev["root"])
    done = set(bridgesim.converted_scenarios(scenario_dir))
    result = {
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "num_converted": len(done),
        "num_expected": len(expected),
        "missing": sorted(expected - done)[:20],
        "num_missing": len(expected - done),
        "num_extra": len(done - expected),
        "log": str(log_path),
    }
    print(f"[bridgesim] convert: {json.dumps(result)}", flush=True)
    if proc.returncode != 0 or not done:
        raise _fail(f"conversion failed; see {log_path}")
    return result


# ---------------------------------------------------------------------------------------------
def _run_one(cfg: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    ev, policy = cfg["evaluator"], cfg["policy"]
    run_dir = Path(cfg["experiment"]["out_dir"]) / "scenarios" / scenario
    if bridgesim.read_average_row(run_dir) is not None:
        return {"scenario": scenario, "status": "already_done"}
    run_dir.mkdir(parents=True, exist_ok=True)
    command = bridgesim.evaluator_command(
        ev["root"], bridgesim.MODEL_TYPES[policy["family"]], Path(policy["checkpoint"]),
        Path(cfg["scenarios"]["dir"]) / scenario, run_dir, ev["run"],
        # policy.bridgesim_args belong to the model (e.g. DiffusionDrive's anchors), not to the
        # protocol, so unlike extra_evaluator_args they do not disable the leaderboard comparison
        extra=[str(x) for x in (policy.get("bridgesim_args") or [])]
              + [str(x) for x in (cfg.get("extra_evaluator_args") or [])],
        degradation=cfg.get("degradation"),
    )
    start = time.time()
    with open(run_dir / "evaluator.log", "w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ev["root"],
                                  timeout=int(cfg["timeout_s_per_scenario"]))
            returncode: Any = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    ok = bridgesim.read_average_row(run_dir) is not None
    return {"scenario": scenario, "status": "ok" if ok else "failed", "returncode": returncode,
            "seconds": round(time.time() - start, 1)}


def action_evaluate(cfg: Dict[str, Any], shard: int, num_shards: int) -> Dict[str, Any]:
    if not 0 <= shard < num_shards:
        raise _fail(f"--shard must be in [0, {num_shards - 1}]")
    checkpoint = Path(cfg["policy"]["checkpoint"])
    if not checkpoint.is_file():
        raise _fail(f"checkpoint not found: {checkpoint}")
    converted = set(bridgesim.converted_scenarios(Path(cfg["scenarios"]["dir"])))
    expected = expected_scenarios(cfg)
    missing = [s for s in expected if s not in converted]
    if missing:
        raise _fail(f"{len(missing)} expected scenarios are not converted (e.g. {missing[:3]}). Run --action convert.")
    mine = expected[shard::num_shards]
    print(f"[bridgesim] evaluate: policy={cfg['policy']['id']} shard {shard}/{num_shards}: "
          f"{len(mine)} scenarios, {cfg['run_parallel']} at a time", flush=True)
    start = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=int(cfg["run_parallel"])) as pool:
        for row in pool.map(lambda s: _run_one(cfg, s), mine):
            rows.append(row)
            print(f"[bridgesim] {row}", flush=True)
    status = pd.Series([r["status"] for r in rows]).value_counts().to_dict() if rows else {}
    result = {"shard": shard, "num_shards": num_shards, "num_scenarios": len(mine), "status": status,
              "seconds": round(time.time() - start, 1),
              "failed": [r["scenario"] for r in rows if r["status"] == "failed"]}
    _write_json(Path(cfg["experiment"]["out_dir"]) / "shards" / f"shard_{shard}_of_{num_shards}.json", result)
    print(f"[bridgesim] evaluate: {json.dumps(result)}", flush=True)
    # The shard that completes the run writes the report, so no separate report job is needed.
    out_dir = Path(cfg["experiment"]["out_dir"])
    if all(bridgesim.read_average_row(out_dir / "scenarios" / s) is not None for s in expected):
        print("[bridgesim] every expected scenario is scored; writing the report", flush=True)
        action_report(cfg)
    return result


# ---------------------------------------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]], expected: List[str]) -> Dict[str, Any]:
    """Average the scenario rows exactly like BridgeSim's aggregation (mean, NaN skipped), and
    report everything that could make that average differ from the published one."""
    frame = pd.DataFrame(rows)
    scored = set(frame["scenario"]) if len(frame) else set()
    out: Dict[str, Any] = {
        "num_expected": len(expected),
        "num_scored": len(scored),
        "num_missing": len(set(expected) - scored),
        "missing_example": sorted(set(expected) - scored)[:10],
    }
    means: Dict[str, float] = {}
    nan_counts: Dict[str, int] = {}
    sems: Dict[str, float] = {}
    for column in bridgesim.METRICS:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            means[column] = float(values.mean()) if values.notna().any() else float("nan")
            nan_counts[column] = int(values.isna().sum())
            n = int(values.notna().sum())
            sems[column] = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    out["mean"] = means
    out["sem"] = sems
    out["num_blank_values"] = nan_counts
    out["epdms"] = 100.0 * means.get("EPDMS_no_ep", float("nan"))
    out["ds"] = 100.0 * means.get("DS", float("nan"))
    out["rc"] = 100.0 * means.get("RC", float("nan"))
    out["epdms_sem"] = 100.0 * sems.get("EPDMS_no_ep", float("nan"))
    out["ds_sem"] = 100.0 * sems.get("DS", float("nan"))
    out["complete"] = bool(out["num_missing"] == 0 and len(expected) > 0)
    return out


def action_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(cfg["experiment"]["out_dir"])
    expected = expected_scenarios(cfg)
    rows = []
    for scenario in expected:
        row = bridgesim.read_average_row(out_dir / "scenarios" / scenario)
        if row is not None:
            rows.append({"scenario": scenario, **{k: row.get(k) for k in bridgesim.METRICS}})
    result = summarize(rows, expected)

    summary: Dict[str, Any] = {
        "experiment": cfg["experiment"]["name"],
        "created_utc": provenance.utc_now(),
        "evaluator": cfg["evaluator"]["id"],
        "policy": cfg["policy"]["id"],
        "checkpoint": {"path": cfg["policy"]["checkpoint"],
                       "sha256": provenance.sha256_of(Path(cfg["policy"]["checkpoint"]))},
        "protocol": cfg["evaluator"]["run"],
        "degradation": {k: (cfg.get("degradation") or {}).get(k) for k in ("name", "level", "value", "wrap")},
        "extra_evaluator_args": cfg.get("extra_evaluator_args") or [],
        "result": result,
        "code": {
            "repo": provenance.git_commit(Path(cfg["paths"]["repo"]), "WETE_GIT_COMMIT"),
            "bridgesim": provenance.git_commit(Path(cfg["evaluator"]["root"]), "WETE_BRIDGESIM_COMMIT"),
        },
        "machine": provenance.machine(),
    }
    reference = (cfg["evaluator"].get("reference") or {}).get(cfg["policy"]["family"])
    # Compare only runs that follow the leaderboard protocol; a variant with extra evaluator
    # options opts in explicitly when it is a candidate for the protocol the leaderboard used.
    compare = cfg.get("compare_with_reference")
    if compare is None:
        compare = not summary["extra_evaluator_args"]
    if str((cfg.get("degradation") or {}).get("name", "none")) != "none":
        compare = False  # a degraded policy is not the published one
    if reference and result["complete"] and compare:
        summary["comparison"] = {
            "source": cfg["evaluator"]["reference"].get("source"),
            "reference": reference,
            "delta_epdms": result["epdms"] - float(reference["epdms"]),
            "delta_ds": result["ds"] - float(reference["ds"]),
            "delta_rc": result["rc"] - float(reference["rc"]),
            "note": "The leaderboard reports one checkpoint per model; compare seed 0.",
        }
    else:
        summary["comparison"] = None

    pd.DataFrame(rows).to_csv(out_dir / "per_scenario.csv", index=False)
    _write_json(out_dir / "summary.json", summary)
    export_dir = cfg["experiment"].get("export_dir")
    if export_dir:
        export = Path(export_dir)
        export.mkdir(parents=True, exist_ok=True)
        (export / "summary.json").write_text((out_dir / "summary.json").read_text())
        (export / "per_scenario.csv").write_text((out_dir / "per_scenario.csv").read_text())
        config_lib.save(cfg, export / "resolved_config.yaml")
    print("[bridgesim] ===== SUMMARY (paste this block back) =====")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print("[bridgesim] ===== END SUMMARY =====")
    return summary


# ---------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = config_lib.resolve(config_lib.load_config(args.config, args.overrides))
    config_lib.require(cfg, REQUIRED_KEYS)
    if cfg["policy"]["family"] not in bridgesim.MODEL_TYPES:
        raise _fail(f"policy.family must be one of {sorted(bridgesim.MODEL_TYPES)}")
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_lib.save(cfg, out_dir / f"resolved_config_{args.action}.yaml")

    if args.action == "convert":
        action_convert(cfg)
    elif args.action == "evaluate":
        action_evaluate(cfg, args.shard, args.num_shards)
    else:
        action_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
