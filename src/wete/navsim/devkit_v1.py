"""Driver for the unmodified NAVSIM v1.1 devkit scripts.

Stage 1 asks one question: does the published checkpoint reproduce its published PDMS on this
cluster? To answer it without ambiguity we do not re-implement anything. We build the Hydra
command lines for the devkit's own ``run_metric_caching.py`` and ``run_pdm_score.py`` from our
config and run them as subprocesses.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

METRIC_CACHING_SCRIPT = Path("navsim/planning/script/run_metric_caching.py")
PDM_SCORE_SCRIPT = Path("navsim/planning/script/run_pdm_score.py")

# The one entry of `hydra.searchpath` in the devkit's default_run_pdm_score.yaml (v1.1).
DEVKIT_SEARCHPATH = "pkg://navsim.planning.script.config.common"

# Directory with our Hydra agent configs for the v1.1 devkit, relative to the repository.
WRAPPER_CONFIG_DIR = Path("configs/navsim_hydra/v1")
WRAPPER_CHECKPOINT_KEY = "agent.base_agent.checkpoint_path"

# Devkit worker configs (navsim/planning/script/config/common/worker/*.yaml) and the key each
# one uses for its degree of parallelism.
_WORKER_PARALLELISM_KEY = {
    "ray_distributed_no_torch": "threads_per_node",
    "ray_distributed": "threads_per_node",
    "single_machine_thread_pool": "max_workers",
    "sequential": None,
}


class DevkitError(RuntimeError):
    """Raised when a devkit script fails or its output is not what we expect."""


def hydra_quote(value: str) -> str:
    """Quote a string for a Hydra override.

    Scenario tokens are hexadecimal strings. Unquoted, Hydra would read ``12e4567890123456`` as
    a float and ``1234567890123456`` as an integer, and the scene filter would then silently
    match nothing. Paths are quoted for the same reason of never relying on the grammar.
    """
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def hydra_list(values: Iterable[str]) -> str:
    return "[" + ",".join(hydra_quote(v) for v in values) + "]"


def worker_overrides(worker: Mapping[str, Any], num_workers: int) -> List[str]:
    """Hydra overrides that select a devkit worker pool and bound its parallelism.

    The bound matters under SLURM: left alone, the devkit sizes its pool from the number of
    cores on the node, not from the cores allocated to the job.
    """
    name = str(worker["name"])
    if name not in _WORKER_PARALLELISM_KEY:
        raise DevkitError(f"Unknown devkit worker '{name}'. Known: {sorted(_WORKER_PARALLELISM_KEY)}")
    overrides = [f"worker={name}"]
    key = _WORKER_PARALLELISM_KEY[name]
    if key is not None:
        overrides.append(f"worker.{key}={int(num_workers)}")
    if name == "single_machine_thread_pool":
        use_processes = bool(worker.get("use_process_pool", False))
        overrides.append(f"worker.use_process_pool={'true' if use_processes else 'false'}")
    return overrides


def scene_filter_overrides(log_names: Optional[Sequence[str]], tokens: Optional[Sequence[str]]) -> List[str]:
    """Restrict the split's scene filter to a subset. ``None`` leaves the split untouched."""
    overrides: List[str] = []
    if log_names is not None:
        overrides.append(f"train_test_split.scene_filter.log_names={hydra_list(log_names)}")
    if tokens is not None:
        overrides.append(f"train_test_split.scene_filter.tokens={hydra_list(tokens)}")
    return overrides


def metric_cache_command(
    devkit_root: Path,
    split: str,
    navsim_log_path: Path,
    cache_dir: Path,
    worker: Mapping[str, Any],
    num_workers: int,
    log_names: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> List[str]:
    return [
        sys.executable,
        str(devkit_root / METRIC_CACHING_SCRIPT),
        f"train_test_split={split}",
        f"cache.cache_path={hydra_quote(str(cache_dir))}",
        f"navsim_log_path={hydra_quote(str(navsim_log_path))}",
        *worker_overrides(worker, num_workers),
        *scene_filter_overrides(log_names, tokens),
    ]


def pdm_score_command(
    devkit_root: Path,
    split: str,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    cache_dir: Path,
    output_dir: Path,
    experiment_name: str,
    agent: str,
    agent_overrides: Sequence[str],
    checkpoint: Path,
    worker: Mapping[str, Any],
    num_workers: int,
    log_names: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
    checkpoint_key: str = "agent.checkpoint_path",
    config_searchpath: Optional[Path] = None,
) -> List[str]:
    """``config_searchpath`` adds a directory of extra Hydra configs (our wrapped agents). The
    devkit's own search path entry is kept: overriding ``hydra.searchpath`` replaces the list."""
    searchpath = []
    if config_searchpath is not None:
        searchpath = [
            "hydra.searchpath="
            + hydra_list([DEVKIT_SEARCHPATH, "file://" + str(Path(config_searchpath).resolve())])
        ]
    return [
        sys.executable,
        str(devkit_root / PDM_SCORE_SCRIPT),
        *searchpath,
        f"train_test_split={split}",
        f"agent={agent}",
        f"{checkpoint_key}={hydra_quote(str(checkpoint))}",
        *agent_overrides,
        f"experiment_name={experiment_name}",
        f"navsim_log_path={hydra_quote(str(navsim_log_path))}",
        f"sensor_blobs_path={hydra_quote(str(sensor_blobs_path))}",
        f"metric_cache_path={hydra_quote(str(cache_dir))}",
        f"output_dir={hydra_quote(str(output_dir))}",
        *worker_overrides(worker, num_workers),
        *scene_filter_overrides(log_names, tokens),
    ]


def read_metric_cache_tokens(cache_dir: Path) -> Set[str]:
    """Tokens present in a metric cache, read the way the devkit's ``MetricCacheLoader`` does.

    The devkit writes ``<cache>/metadata/<name>_metadata_node_<k>.csv`` with one cache file
    path per line after a header; the token is the parent directory of each file.
    """
    metadata_dir = cache_dir / "metadata"
    if not metadata_dir.is_dir():
        return set()
    tokens: Set[str] = set()
    for metadata_file in sorted(metadata_dir.glob("*.csv")):
        with open(metadata_file, newline="") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]:
            if row and row[0]:
                tokens.add(Path(row[0]).parent.name)
    return tokens


def run(command: Sequence[str], log_path: Path, env: Optional[Mapping[str, str]] = None) -> float:
    """Run a devkit command, mirror its output to ``log_path`` and stdout, return seconds taken."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    print(f"[wete] running: {format_command(command)}", flush=True)
    start = time.time()
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            sys.stdout.write(line)
        return_code = process.wait()
    elapsed = time.time() - start
    if return_code != 0:
        raise DevkitError(f"Command failed with exit code {return_code}; see {log_path}")
    return elapsed


def format_command(command: Sequence[str]) -> str:
    """Render a command so that it can be pasted into a shell."""
    parts = []
    for arg in command:
        parts.append("'" + arg.replace("'", "'\\''") + "'" if any(c in arg for c in ' "[]$*?()') else arg)
    return " ".join(parts)


def devkit_env(maps_root: Path, map_version: str, openscene_root: Path, exp_root: Path) -> Dict[str, str]:
    """Environment variables the devkit reads directly."""
    return {
        "NUPLAN_MAPS_ROOT": str(maps_root),
        "NUPLAN_MAP_VERSION": map_version,
        "OPENSCENE_DATA_ROOT": str(openscene_root),
        "NAVSIM_EXP_ROOT": str(exp_root),
    }


def is_wrapped(degradation: Mapping[str, Any]) -> bool:
    """Does this run go through ``wete.navsim.degraded_agent.DegradedAgent``?"""
    return str(degradation.get("name", "none")) != "none" or bool(degradation.get("wrap", False))


def rebase_agent_overrides(overrides: Sequence[str]) -> List[str]:
    """Move ``agent.X=...`` overrides written for the plain agent onto the wrapped base agent."""
    out = []
    for item in overrides:
        if item.startswith("agent."):
            out.append("agent.base_agent." + item[len("agent."):])
        else:
            out.append(item)
    return out


def wrapper_overrides(degradation: Mapping[str, Any], used_cameras: Optional[Sequence[str]]) -> List[str]:
    """Hydra overrides that configure the wrapper itself."""
    out = [
        f"agent.degradation={degradation.get('name', 'none')}",
        f"agent.level={int(degradation.get('level', 0))}",
    ]
    if degradation.get("value") is not None:
        out.append(f"agent.value={float(degradation['value'])!r}")
    if used_cameras:
        out.append(f"agent.used_cameras={hydra_list(used_cameras)}")
    return out


def required_history_frames(degradation: Mapping[str, Any], policy_history_frames: int, frame_dt: float = 0.5) -> int:
    """Frames, counted back from the current one, whose sensor files the run reads.

    stale_obs reads one older frame; history_removal reads all four. Both need those files on
    disk, so the sensor preflight must look that far back.
    """
    name = str(degradation.get("name", "none"))
    level = int(degradation.get("level", 0))
    if name == "none" or level == 0:
        return policy_history_frames
    if name == "history_removal":
        return max(policy_history_frames, 4)
    if name == "stale_obs":
        from wete import degradations as D

        value = degradation.get("value")
        seconds = float(value) if value is not None else D.level_param(name, level)
        shift = int(round(seconds / frame_dt))
        return max(policy_history_frames, min(4, shift + 1))
    return policy_history_frames
