"""Stage 1 feasibility probe for BridgeSim as the closed-loop evaluator.

Usage (normally through ``jobs/bridgesim_probe.sbatch`` on one GPU)::

    python -m wete.cli.bridgesim_probe --config configs/experiments/stage1_bridgesim_probe.yaml

Each step records what it found and carries on, so a single job answers every question at once.
BridgeSim's own scripts run unmodified as subprocesses; this module only prepares their inputs
and reads their outputs. Nothing here is a result about policies: a handful of scenarios is not
comparable with the 421-scenario leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from wete import config as config_lib
from wete import provenance

# BridgeSim adapter class for each --model-type we use (bridgesim/evaluation/models/).
ADAPTERS = {
    "transfuser": ("bridgesim.evaluation.models.transfuser_adapter", "TransfuserAdapter"),
    "ltf": ("bridgesim.evaluation.models.ltf_adapter", "LTFAdapter"),
    "egomlp": ("bridgesim.evaluation.models.ego_mlp_adapter", "EgoStatusMLPAdapter"),
}

RENDER_SNIPPET = r"""
import sys
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
if sys.argv[1] != "default":
    loadPrcFileData("", "load-display " + sys.argv[1])
from direct.showbase.ShowBase import ShowBase
base = ShowBase()
base.graphicsEngine.renderFrame()
gsg = base.win.getGsg()
print("PIPE", base.pipe.getType().getName())
print("RENDERER", gsg.getDriverRenderer(), "|", gsg.getDriverVendor())
"""


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------
def step_environment() -> Dict[str, Any]:
    import numpy
    import torch
    from panda3d.core import PandaSystem

    out: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "panda3d": PandaSystem.getVersionString(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def step_render(displays: List[str]) -> Dict[str, Any]:
    """Open an offscreen buffer in a fresh process per display module (ShowBase is a singleton)."""
    results = {}
    for display in displays:
        proc = subprocess.run(
            [sys.executable, "-c", RENDER_SNIPPET, display], capture_output=True, text=True, timeout=300
        )
        found = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in proc.stdout.splitlines()
                 if line.startswith(("PIPE ", "RENDERER "))}
        results[display] = {
            "ok": proc.returncode == 0 and "PIPE" in found,
            "pipe": found.get("PIPE"),
            "renderer": found.get("RENDERER"),
            "error_tail": None if proc.returncode == 0 else _tail(proc.stderr, 12),
        }
    return results


def step_checkpoints(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Load each checkpoint through BridgeSim's adapter, then verify parameter by parameter.

    The adapters call ``load_state_dict(strict=False)``, which silently skips keys that do not
    match. We therefore compare every tensor of the built model with the checkpoint tensor whose
    key ends with the same name: a parameter counts as loaded only if the values are identical.
    """
    import torch

    results = {}
    for check in checks:
        model_type, path = check["model_type"], Path(check["checkpoint"])
        entry: Dict[str, Any] = {"checkpoint": str(path)}
        try:
            module_name, class_name = ADAPTERS[model_type]
            adapter = getattr(importlib.import_module(module_name), class_name)(str(path))
            adapter.load_model()
            model_sd = {k: v.detach().cpu() for k, v in adapter.model.state_dict().items()}
            raw = torch.load(str(path), map_location="cpu")
            ckpt_sd = raw.get("state_dict", raw)
            unmatched = []
            for key, tensor in model_sd.items():
                candidates = [c for c in ckpt_sd if c == key or c.endswith("." + key)]
                if not any(ckpt_sd[c].shape == tensor.shape and torch.equal(ckpt_sd[c].cpu(), tensor)
                           for c in candidates):
                    unmatched.append(key)
            entry.update({
                "model_tensors": len(model_sd),
                "checkpoint_tensors": len(ckpt_sd),
                "loaded_from_checkpoint": len(model_sd) - len(unmatched),
                "not_loaded": unmatched[:15],
                "num_not_loaded": len(unmatched),
                "all_loaded": not unmatched,
            })
        except Exception as error:  # report and continue with the next check
            entry["error"] = f"{type(error).__name__}: {error}"
        results[model_type] = entry
    return results


def step_convert(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Count the NavHard logs on disk and convert the scenarios of the first few."""
    ev = cfg["evaluator"]
    with open(ev["scene_filter"]) as handle:
        navhard = yaml.safe_load(handle)
    log_dir = Path(ev["navsim_log_path"])
    present = [name for name in navhard["log_names"] if (log_dir / f"{name}.pkl").is_file()]
    result: Dict[str, Any] = {
        "navhard_logs_required": len(navhard["log_names"]),
        "navhard_logs_on_disk": len(present),
        "navhard_tokens": len(navhard.get("tokens") or []),
        "missing_logs_example": [n for n in navhard["log_names"] if n not in present][:5],
    }
    chosen = present[: int(cfg["probe"]["convert_logs"])]
    if not chosen:
        result["error"] = "none of the NavHard logs is on disk"
        return result

    subset = {k: navhard[k] for k in ("num_history_frames", "num_future_frames", "frame_interval", "has_route")}
    subset.update({"max_scenes": None, "log_names": chosen, "tokens": navhard.get("tokens")})
    filter_path = out_dir / "navhard_probe_filter.yaml"
    with open(filter_path, "w") as handle:
        yaml.safe_dump(subset, handle)

    scenario_dir = out_dir / "scenarios"
    command = [
        sys.executable, str(Path(ev["root"]) / "converters/openscene/convert_openscene_with_filter.py"),
        "--scene-filter", str(filter_path),
        "--input-dir", str(log_dir),
        "--output-dir", str(scenario_dir),
        "--map-root", str(ev["map_root"]),
        "--num-future-frames-extract", str(int(ev["conversion"]["num_future_frames_extract"])),
    ]
    if ev["conversion"].get("interpolate"):
        command.append("--interpolate")
    start = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=ev["root"])
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "convert.log").write_text(proc.stdout + "\n" + proc.stderr)
    scenarios = sorted(p.name for p in scenario_dir.glob("sd_*") if list((p / f"{p.name}_0").glob("*.pkl")))
    result.update({
        "logs_converted": chosen,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "scenarios": scenarios,
        "num_scenarios": len(scenarios),
        "error_tail": None if proc.returncode == 0 else _tail(proc.stderr),
    })
    return result


def _read_average_row(run_dir: Path) -> Optional[Dict[str, str]]:
    files = sorted(run_dir.rglob("driving_score_summary.csv"))
    if not files:
        return None
    with open(files[-1], newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("frame_id") == "AVERAGE"]
    return rows[-1] if rows else None


def step_evaluate(cfg: Dict[str, Any], out_dir: Path, scenarios: List[str]) -> Dict[str, Any]:
    ev, probe = cfg["evaluator"], cfg["probe"]
    run = ev["run"]
    evaluator_script = Path(ev["root"]) / "bridgesim/evaluation/unified_evaluator.py"
    results: Dict[str, Any] = {}
    for model in probe["models"]:
        rows = []
        for scenario in scenarios[: int(probe["scenarios_per_model"])]:
            run_dir = out_dir / "runs" / model["name"] / scenario
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(evaluator_script),
                "--model-type", model["model_type"],
                "--checkpoint", str(model["checkpoint"]),
                "--scenario-path", str(out_dir / "scenarios" / scenario),
                "--output-dir", str(run_dir),
                "--traffic-mode", run["traffic_mode"],
                "--eval-mode", run["eval_mode"],
                "--controller", run["controller"],
                "--sim-dt", str(run["sim_dt"]),
                "--replan-rate", str(int(run["replan_rate"])),
                "--ego-replay-frames", str(int(run["ego_replay_frames"])),
                "--eval-frames", str(int(run["eval_frames"])),
                "--no-save-perframe",
            ]
            start = time.time()
            try:
                proc = subprocess.run(command, capture_output=True, text=True, cwd=ev["root"],
                                      timeout=int(probe["timeout_s_per_run"]))
                returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as error:
                returncode, stdout, stderr = "timeout", str(error.stdout or ""), str(error.stderr or "")
            (run_dir / "evaluator.log").write_text(f"$ {' '.join(command)}\n\n{stdout}\n{stderr}")
            average = _read_average_row(run_dir)
            rows.append({
                "scenario": scenario,
                "returncode": returncode,
                "seconds": round(time.time() - start, 1),
                "average_row": average,
                "error_tail": None if returncode == 0 else _tail(stderr or stdout),
            })
            print(f"[probe] {model['name']} {scenario}: rc={returncode} "
                  f"{rows[-1]['seconds']} s average={average}", flush=True)
        results[model["name"]] = rows
    return results


# ---------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = config_lib.resolve(config_lib.load_config(args.config, args.overrides))
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_lib.save(cfg, out_dir / "resolved_config.yaml")

    summary: Dict[str, Any] = {
        "experiment": cfg["experiment"]["name"],
        "created_utc": provenance.utc_now(),
        "machine": provenance.machine(),
        "code": {
            "repo": provenance.git_commit(Path(cfg["paths"]["repo"]), "WETE_GIT_COMMIT"),
            "bridgesim": provenance.git_commit(Path(cfg["evaluator"]["root"]), "WETE_BRIDGESIM_COMMIT"),
        },
        "note": "Feasibility probe on a handful of scenarios; not comparable with the leaderboard.",
    }
    steps = [
        ("environment", lambda: step_environment()),
        ("render", lambda: step_render(["default", "p3headlessgl"])),
        ("checkpoints", lambda: step_checkpoints(cfg["probe"]["checkpoint_checks"])),
        ("convert", lambda: step_convert(cfg, out_dir)),
    ]
    for name, func in steps:
        print(f"[probe] step {name}", flush=True)
        start = time.time()
        try:
            summary[name] = func()
        except Exception as error:
            summary[name] = {"error": f"{type(error).__name__}: {error}"}
        summary.setdefault("timings_s", {})[name] = round(time.time() - start, 1)

    scenarios = summary.get("convert", {}).get("scenarios") or []
    if scenarios:
        print("[probe] step evaluate", flush=True)
        start = time.time()
        summary["evaluate"] = step_evaluate(cfg, out_dir, scenarios)
        summary["timings_s"]["evaluate"] = round(time.time() - start, 1)
    else:
        summary["evaluate"] = {"skipped": "no converted scenarios"}

    summary = _json_safe(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    export_dir = cfg["experiment"].get("export_dir")
    if export_dir:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        (Path(export_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("[probe] ===== SUMMARY (paste this block back) =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[probe] ===== END SUMMARY =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
