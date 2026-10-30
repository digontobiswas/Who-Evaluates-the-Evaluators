"""NAVSIM job: score a policy with the unmodified NAVSIM devkit (v1.1 PDMS or v2 EPDMS).

The devkit is chosen by the evaluator config: ``evaluator.devkit_version`` is ``v1`` (default)
or ``v2``; with v2, ``evaluator.two_stage: true`` runs the two-stage pseudo-simulation
(run_pdm_score.py on real + synthetic scenes, e.g. navhard_two_stage) and otherwise the one-stage
EPDMS on real scenes (run_pdm_score_one_stage.py, e.g. navtest). The job script must put the
matching devkit on PYTHONPATH (jobs/navsim_v1_pdms.sbatch: NAVSIM_VERSION=v1|v2).

Usage (normally through ``jobs/navsim_v1_pdms.sbatch``)::

    python -m wete.cli.navsim_v1 --config configs/experiments/<file>.yaml [key=value ...]

Stages, run in this order unless ``--stages`` narrows them:

``select``  choose scenarios and verify that their sensor files are on disk
``cache``   run the devkit's metric caching for those scenarios (skipped if already cached)
``score``   run the devkit's PDM scoring for the configured checkpoint
``report``  aggregate the per-scenario CSV, compare with the published value, export a summary

Population A (graded degradations) uses the same stages. When the config's ``degradation``
block names a degradation, or sets ``wrap: true``, the checkpoint is run inside
``wete.navsim.degraded_agent.DegradedAgent``; otherwise the devkit's own agent runs untouched.

The job never invents a number: everything in ``summary.json`` is computed from the devkit's
CSV, and the published reference value is carried through from the policy config with its source.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from wete import config as config_lib
from wete import provenance
from wete.navsim import devkit_v1, report, selection

STAGES = ("select", "cache", "score", "report")

REQUIRED_KEYS = (
    "paths.maps",
    "paths.openscene",
    "paths.exp",
    "navsim.map_version",
    "evaluator.devkit_root",
    "evaluator.split",
    "evaluator.navsim_log_path",
    "evaluator.sensor_blobs_path",
    "evaluator.metric_cache.dir",
    "evaluator.metric_cache.worker.name",
    "evaluator.scoring.worker.name",
    "policy.id",
    "policy.navsim_agent",
    "policy.checkpoint",
    "policy.sensors.cameras",
    "policy.sensors.lidar",
    "policy.sensors.history_frames",
    "experiment.name",
    "experiment.out_dir",
    "scenarios.mode",
    "degradation.name",
    "degradation.level",
    "workers",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment YAML file.")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of {STAGES}.")
    parser.add_argument("--dry-run", action="store_true", help="Print the devkit commands without running them.")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. policy.seed=1 workers=16.")
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    """Replace NaN and infinity with null so that the files are strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _fail(message: str) -> "SystemExit":
    return SystemExit(f"[wete] ERROR: {message}")


def _version(cfg: Dict[str, Any]) -> str:
    version = str(cfg["evaluator"].get("devkit_version", "v1"))
    if version not in ("v1", "v2"):
        raise _fail(f"evaluator.devkit_version must be v1 or v2, got {version!r}")
    return version


def _two_stage(cfg: Dict[str, Any]) -> bool:
    return _version(cfg) == "v2" and bool(cfg["evaluator"].get("two_stage", False))


def stage_select(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Decide which scenarios to run and prove their sensor files exist."""
    if _two_stage(cfg):
        # Two-stage splits list synthetic scenes that are in no log, so the log-based sensor check
        # does not apply. The devkit scores the split's own definition; completeness is judged from
        # the devkit's valid flags and its pseudo-simulation aggregate.
        synth = Path(cfg["evaluator"]["synthetic_scenes_path"])
        if not synth.is_dir() or not any(synth.iterdir()):
            raise _fail(f"No synthetic scenes in {synth}; download the split (scripts in the v2 devkit's download/).")
        chosen = {"mode": "full", "split": cfg["evaluator"]["split"], "log_names": None, "tokens": None,
                  "expected_tokens": None, "num_expected": None, "num_logs_read": None}
        _write_json(out_dir / "selection.json", chosen)
        print(f"[wete] select: two-stage split {chosen['split']}, scenarios defined by the devkit")
        return chosen
    devkit_root = Path(cfg["evaluator"]["devkit_root"])
    split_filter = selection.load_split_filter(devkit_root, cfg["evaluator"]["split"])
    sensors = cfg["policy"]["sensors"]
    mode = cfg["scenarios"]["mode"]
    if mode not in ("full", "subset"):
        raise _fail(f"scenarios.mode must be 'full' or 'subset', got '{mode}'")

    check = selection.check_sensors(
        navsim_log_path=Path(cfg["evaluator"]["navsim_log_path"]),
        sensor_blobs_path=Path(cfg["evaluator"]["sensor_blobs_path"]),
        split_filter=split_filter,
        cameras=list(sensors["cameras"]),
        lidar=bool(sensors["lidar"]),
        history_frames=devkit_v1.required_history_frames(cfg["degradation"], int(sensors["history_frames"])),
        max_scenarios=cfg["scenarios"].get("max_scenarios") if mode == "subset" else None,
        max_logs=cfg["scenarios"].get("max_logs") if mode == "subset" else None,
    )

    if mode == "subset":
        if not check.available_tokens:
            raise _fail(
                "No scenario of the split has its sensor files on disk. Download at least one camera "
                "shard and one LiDAR shard with scripts/fetch_assets.sh. "
                f"Example missing files: {check.example_missing_files}"
            )
        chosen = {
            "mode": mode,
            "split": split_filter.split,
            "log_names": check.available_log_names,
            "tokens": check.available_tokens,
        }
    else:
        problems = []
        if check.missing_logs:
            problems.append(f"{len(check.missing_logs)} log files missing")
        if check.missing_tokens:
            problems.append(f"{len(check.missing_tokens)} scenarios lack sensor files")
        if check.unseen_tokens:
            problems.append(f"{len(check.unseen_tokens)} split tokens found in no log")
        if problems:
            _write_json(out_dir / "sensor_check_failed.json", check.to_dict())
            raise _fail(
                "Full-split run refused: " + "; ".join(problems) + ". "
                f"Details in {out_dir / 'sensor_check_failed.json'}. "
                f"Example missing files: {check.example_missing_files}"
            )
        # A full run passes no scene-filter overrides: the devkit's own split definition applies.
        chosen = {"mode": mode, "split": split_filter.split, "log_names": None, "tokens": None}

    chosen["expected_tokens"] = check.available_tokens
    chosen["num_expected"] = len(check.available_tokens)
    chosen["num_logs_read"] = check.num_logs_read
    _write_json(out_dir / "selection.json", chosen)
    print(f"[wete] select: mode={mode}, scenarios={chosen['num_expected']}, logs read={check.num_logs_read}")
    return chosen


def _load_selection(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "selection.json"
    if not path.is_file():
        raise _fail(f"{path} not found. Run the 'select' stage first.")
    return _read_json(path)


def stage_cache(cfg: Dict[str, Any], out_dir: Path, chosen: Dict[str, Any], dry_run: bool) -> Optional[float]:
    """Build the metric cache unless every expected scenario is already in it."""
    cache_dir = Path(cfg["evaluator"]["metric_cache"]["dir"])
    expected = set(chosen["expected_tokens"] or [])
    cached = devkit_v1.read_metric_cache_tokens(cache_dir)
    if expected and expected <= cached:
        print(f"[wete] cache: all {len(expected)} scenarios already cached in {cache_dir}, skipping.")
        return None
    if not expected and cached and _two_stage(cfg):
        print(f"[wete] cache: two-stage cache already present in {cache_dir} ({len(cached)} tokens), skipping.")
        return None

    common = dict(
        devkit_root=Path(cfg["evaluator"]["devkit_root"]),
        split=cfg["evaluator"]["split"],
        navsim_log_path=Path(cfg["evaluator"]["navsim_log_path"]),
        cache_dir=cache_dir,
        worker=cfg["evaluator"]["metric_cache"]["worker"],
        num_workers=int(cfg["workers"]),
        log_names=chosen["log_names"],
        tokens=chosen["tokens"],
    )
    if _version(cfg) == "v2":
        synth = cfg["evaluator"].get("synthetic_scenes_path") if _two_stage(cfg) else None
        command = devkit_v1.metric_cache_command_v2(synthetic_scenes_path=Path(synth) if synth else None, **common)
    else:
        command = devkit_v1.metric_cache_command(**common)
    if dry_run:
        print(devkit_v1.format_command(command))
        return None

    elapsed = devkit_v1.run(command, out_dir / "logs" / "metric_caching.log", env=_devkit_env(cfg))
    cached = devkit_v1.read_metric_cache_tokens(cache_dir)
    missing = expected - cached
    if not cached:
        raise _fail(f"Metric caching finished but {cache_dir} holds no tokens. See {out_dir / 'logs' / 'metric_caching.log'}.")
    if missing:
        raise _fail(
            f"Metric caching finished but {len(missing)} of {len(expected)} scenarios are not in the cache "
            f"({cache_dir}). See {out_dir / 'logs' / 'metric_caching.log'}."
        )
    print(f"[wete] cache: {len(expected)} scenarios cached in {elapsed:.0f} s.")
    return elapsed


def stage_score(cfg: Dict[str, Any], out_dir: Path, chosen: Dict[str, Any], dry_run: bool) -> Optional[float]:
    """Run the devkit's PDM scoring for the configured checkpoint."""
    checkpoint = Path(cfg["policy"]["checkpoint"])
    if not checkpoint.is_file() and not dry_run:
        raise _fail(f"Checkpoint not found: {checkpoint}. Fetch it with scripts/fetch_assets.sh.")

    score_dir = out_dir / "pdm_score"
    degradation = cfg["degradation"]
    agent_overrides = list(cfg["policy"].get("navsim_overrides") or [])
    if devkit_v1.is_wrapped(degradation):
        agent = cfg["policy"].get("degraded_agent")
        if not agent:
            raise _fail(f"policy {cfg['policy']['id']} has no 'degraded_agent' entry; it cannot be wrapped.")
        agent_overrides = devkit_v1.rebase_agent_overrides(agent_overrides) + devkit_v1.wrapper_overrides(
            degradation, cfg["policy"]["sensors"].get("used_cameras")
        )
        checkpoint_key = devkit_v1.WRAPPER_CHECKPOINT_KEY
        searchpath = Path(cfg["paths"]["repo"]) / devkit_v1.WRAPPER_CONFIG_DIRS[_version(cfg)]
    else:
        agent = cfg["policy"]["navsim_agent"]
        checkpoint_key = "agent.checkpoint_path"
        searchpath = None
    builder = devkit_v1.pdm_score_command
    extra: Dict[str, Any] = {}
    if _version(cfg) == "v2":
        builder = devkit_v1.pdm_score_command_v2
        two = _two_stage(cfg)
        extra = {"two_stage": two, "traffic_agents": cfg["evaluator"].get("traffic_agents", "non_reactive")}
        if two:
            extra["synthetic_sensor_path"] = Path(cfg["evaluator"]["synthetic_sensor_path"])
            extra["synthetic_scenes_path"] = Path(cfg["evaluator"]["synthetic_scenes_path"])
    command = builder(
        **extra,
        devkit_root=Path(cfg["evaluator"]["devkit_root"]),
        split=cfg["evaluator"]["split"],
        navsim_log_path=Path(cfg["evaluator"]["navsim_log_path"]),
        sensor_blobs_path=Path(cfg["evaluator"]["sensor_blobs_path"]),
        cache_dir=Path(cfg["evaluator"]["metric_cache"]["dir"]),
        output_dir=score_dir,
        experiment_name=cfg["experiment"]["name"],
        agent=agent,
        agent_overrides=agent_overrides,
        checkpoint=checkpoint,
        worker=cfg["evaluator"]["scoring"]["worker"],
        num_workers=int(cfg["workers"]),
        log_names=chosen["log_names"],
        tokens=chosen["tokens"],
        checkpoint_key=checkpoint_key,
        config_searchpath=searchpath,
    )
    if dry_run:
        print(devkit_v1.format_command(command))
        return None

    # Plans are appended line by line; lines from an earlier attempt must not mix in.
    plans_dir = score_dir / "plans"
    if plans_dir.is_dir():
        for stale in plans_dir.glob("*.jsonl"):
            stale.rename(stale.with_suffix(".jsonl.superseded"))

    # Old CSVs from an earlier attempt would be picked up by the report stage; keep them aside.
    if score_dir.is_dir():
        for stale in score_dir.glob("*.csv"):
            stale.rename(stale.with_suffix(".csv.superseded"))

    elapsed = devkit_v1.run(command, out_dir / "logs" / "pdm_score.log", env=_devkit_env(cfg))
    print(f"[wete] score: finished in {elapsed:.0f} s.")
    return elapsed


def stage_report(cfg: Dict[str, Any], out_dir: Path, chosen: Dict[str, Any], timings: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate, compare with the published value where that is meaningful, and export."""
    csv_path = report.find_score_csv(out_dir / "pdm_score")
    if _version(cfg) == "v2":
        scores = report.load_scores_v2(csv_path)
        result = report.summarize_v2(scores, _two_stage(cfg), expected_tokens=chosen["expected_tokens"])
    else:
        scores = report.load_scores(csv_path)
        result = report.summarize(scores, expected_tokens=chosen["expected_tokens"])

    degradation = cfg["degradation"]
    degraded = str(degradation["name"]) != "none" and int(degradation["level"]) > 0
    summary: Dict[str, Any] = {
        "experiment": cfg["experiment"]["name"],
        "created_utc": provenance.utc_now(),
        "evaluator": cfg["evaluator"].get("id"),
        "policy": cfg["policy"]["id"],
        "degradation": {
            "name": degradation["name"],
            "level": int(degradation["level"]),
            "value": degradation.get("value"),
            "wrapped": devkit_v1.is_wrapped(degradation),
        },
        "split": chosen["split"],
        "scenario_mode": chosen["mode"],
        "result": result,
        "timings_s": timings,
        "score_csv": str(csv_path),
        "checkpoint": {
            "path": cfg["policy"]["checkpoint"],
            "size_bytes": Path(cfg["policy"]["checkpoint"]).stat().st_size,
            "sha256": provenance.sha256_of(Path(cfg["policy"]["checkpoint"])),
        },
        "code": {
            "repo": provenance.git_commit(Path(cfg["paths"]["repo"]), "WETE_GIT_COMMIT")
            if cfg["paths"].get("repo")
            else None,
            "navsim_devkit": provenance.git_commit(
                Path(cfg["evaluator"]["devkit_root"]),
                "WETE_NAVSIM_V2_COMMIT" if Path(cfg["evaluator"]["devkit_root"]).name == "navsim_v2"
                else "WETE_NAVSIM_V1_COMMIT",
            ),
            "nuplan_devkit": provenance.git_commit(Path(cfg["evaluator"]["nuplan_devkit_root"]), "WETE_NUPLAN_COMMIT")
            if cfg["evaluator"].get("nuplan_devkit_root")
            else None,
        },
        "machine": provenance.machine(),
    }

    score_seconds = timings.get("score")
    if score_seconds and result["num_scored"]:
        summary["scenarios_per_second"] = result["num_scored"] / score_seconds

    if devkit_v1.is_wrapped(degradation):
        summary["open_loop"] = report.summarize_plans(out_dir / "pdm_score" / "plans")

    reference = cfg["policy"].get("reference") if _version(cfg) == "v1" else None
    if degraded:
        summary["comparison"] = None
        summary["comparison_note"] = "Degraded policy: the published PDMS does not apply."
    elif chosen["mode"] == "full" and reference and reference.get("pdms") is not None:
        summary["comparison"] = report.compare_to_reference(
            result, reference, tolerance_pdms=float(cfg.get("check", {}).get("tolerance_pdms", 1.0))
        )
    else:
        summary["comparison"] = None
        summary["comparison_note"] = (
            "No comparison with the published PDMS: it is defined on the full split, "
            "and this run covers a subset."
        )

    _write_json(out_dir / "summary.json", summary)

    export_dir = cfg["experiment"].get("export_dir")
    if export_dir:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / "summary.json", export_path / "summary.json")
        shutil.copyfile(csv_path, export_path / "per_scenario.csv")
        shutil.copyfile(out_dir / "resolved_config.yaml", export_path / "resolved_config.yaml")
        print(f"[wete] report: exported summary.json, per_scenario.csv, resolved_config.yaml to {export_path}")

    print("[wete] ===== SUMMARY (paste this block back) =====")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    print("[wete] ===== END SUMMARY =====")
    return summary


def _devkit_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    return devkit_v1.devkit_env(
        maps_root=Path(cfg["paths"]["maps"]),
        map_version=cfg["navsim"]["map_version"],
        openscene_root=Path(cfg["paths"]["openscene"]),
        exp_root=Path(cfg["paths"]["exp"]) / "navsim",
    )


def _require_cached(cfg: Dict[str, Any], chosen: Dict[str, Any]) -> None:
    """Scoring without the cache stage: every expected scenario must already be cached.

    The devkit silently skips uncached tokens, and several Population A jobs that each tried to
    build the same cache at once would collide, so jobs that skip caching check first.
    """
    cache_dir = Path(cfg["evaluator"]["metric_cache"]["dir"])
    cached = devkit_v1.read_metric_cache_tokens(cache_dir)
    if chosen["expected_tokens"] is None:
        if not cached:
            raise _fail(f"The metric cache {cache_dir} is empty. Build it first with STAGES=select,cache (one job).")
        return
    missing = set(chosen["expected_tokens"]) - cached
    if missing:
        raise _fail(
            f"{len(missing)} of {len(chosen['expected_tokens'])} scenarios are not in the metric cache "
            f"{cache_dir}. Build it first with STAGES=select,cache (one job), then resubmit."
        )


def _preflight(cfg: Dict[str, Any]) -> None:
    """Stop before computing anything if an input is missing."""
    checks = {
        "NAVSIM devkit": Path(cfg["evaluator"]["devkit_root"]) / devkit_v1.PDM_SCORE_SCRIPT,
        "nuPlan maps": Path(cfg["paths"]["maps"]),
        "OpenScene logs": Path(cfg["evaluator"]["navsim_log_path"]),
        "OpenScene sensors": Path(cfg["evaluator"]["sensor_blobs_path"]),
    }
    if _two_stage(cfg):
        checks["synthetic sensors"] = Path(cfg["evaluator"]["synthetic_sensor_path"])
        checks["synthetic scenes"] = Path(cfg["evaluator"]["synthetic_scenes_path"])
    absent = [f"{name}: {path}" for name, path in checks.items() if not path.exists()]
    if absent:
        raise _fail("Missing inputs:\n  " + "\n  ".join(absent))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise _fail(f"Unknown stages {unknown}; choose from {STAGES}")

    cfg = config_lib.resolve(config_lib.load_config(args.config, args.overrides))
    config_lib.require(cfg, REQUIRED_KEYS)
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_lib.save(cfg, out_dir / "resolved_config.yaml")
    print(f"[wete] experiment={cfg['experiment']['name']} out_dir={out_dir} stages={stages} workers={cfg['workers']}")

    if not args.dry_run:
        _preflight(cfg)
        on_path = os.environ.get("NAVSIM_DEVKIT_ROOT")
        wanted = Path(cfg["evaluator"]["devkit_root"])
        if on_path and Path(on_path).resolve() != wanted.resolve():
            raise _fail(f"The job put {on_path} on PYTHONPATH but the evaluator uses {wanted}. "
                        f"Submit with NAVSIM_VERSION={_version(cfg)}.")

    timings_path = out_dir / "timings.json"
    timings: Dict[str, Any] = _read_json(timings_path) if timings_path.is_file() else {}

    started = time.time()
    if args.dry_run and not (out_dir / "selection.json").is_file():
        # Nothing on disk to select from (for example on a laptop): show the command shapes only.
        subset = cfg["scenarios"]["mode"] == "subset"
        chosen = {
            "mode": cfg["scenarios"]["mode"],
            "split": cfg["evaluator"]["split"],
            "log_names": ["<log_name>"] if subset else None,
            "tokens": ["<token>"] if subset else None,
            "expected_tokens": [],
        }
    elif "select" in stages and not args.dry_run:
        chosen = stage_select(cfg, out_dir)
        timings["select"] = time.time() - started
    else:
        chosen = _load_selection(out_dir)

    if "score" in stages and "cache" not in stages and not args.dry_run:
        _require_cached(cfg, chosen)
    if "cache" in stages:
        elapsed = stage_cache(cfg, out_dir, chosen, args.dry_run)
        if elapsed is not None:
            timings["cache"] = elapsed
    if "score" in stages:
        elapsed = stage_score(cfg, out_dir, chosen, args.dry_run)
        if elapsed is not None:
            timings["score"] = elapsed
    if not args.dry_run:
        _write_json(timings_path, timings)
    if "report" in stages and not args.dry_run:
        stage_report(cfg, out_dir, chosen, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
