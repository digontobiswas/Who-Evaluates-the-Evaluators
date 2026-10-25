"""Scenario selection and sensor preflight for NAVSIM splits.

Two uses:

* Smoke test. Only part of the sensor data may be on disk (one or two download shards). We
  pick the first N scenarios of the split whose sensor files are actually present.
* Full run. We verify that every scenario of the split has its sensor files before scoring.
  The devkit turns a missing file into an invalid scenario and then averages over the valid
  ones, which would bias the reported score without any error.

This module reads the devkit's split definition and the OpenScene log pickles directly, so it
has no dependency on the devkit being importable.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

SPLIT_CONFIG_DIR = Path("navsim/planning/script/config/common/train_test_split")


class SelectionError(RuntimeError):
    """Raised when a split definition or the data on disk is not usable."""


@dataclass(frozen=True)
class SplitFilter:
    """Scene filter of a NAVSIM split as shipped with the devkit."""

    split: str
    data_split: str
    log_names: Optional[List[str]]
    tokens: Optional[List[str]]
    num_history_frames: int


@dataclass
class SensorCheck:
    """Which scenarios of a split have their sensor files on disk."""

    available_tokens: List[str] = field(default_factory=list)
    available_log_names: List[str] = field(default_factory=list)
    missing_tokens: List[str] = field(default_factory=list)
    missing_logs: List[str] = field(default_factory=list)  # log pickles not found at all
    unseen_tokens: List[str] = field(default_factory=list)  # split tokens found in no log pickle
    example_missing_files: List[str] = field(default_factory=list)
    num_logs_read: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_split_filter(devkit_root: Path, split: str) -> SplitFilter:
    """Read ``train_test_split/<split>.yaml`` and the scene filter it points to."""
    config_dir = devkit_root / SPLIT_CONFIG_DIR
    split_file = config_dir / f"{split}.yaml"
    if not split_file.is_file():
        raise SelectionError(f"Split config not found: {split_file}")
    with open(split_file) as handle:
        split_cfg = yaml.safe_load(handle)

    filter_name = None
    for entry in split_cfg.get("defaults", []):
        if isinstance(entry, dict) and "scene_filter" in entry:
            filter_name = entry["scene_filter"]
    if filter_name is None:
        raise SelectionError(f"{split_file} does not name a scene_filter in its defaults list.")

    filter_file = config_dir / "scene_filter" / f"{filter_name}.yaml"
    with open(filter_file) as handle:
        filter_cfg = yaml.safe_load(handle)

    def as_str_list(values: Optional[Sequence[Any]]) -> Optional[List[str]]:
        return None if values is None else [str(v) for v in values]

    return SplitFilter(
        split=split,
        data_split=str(split_cfg["data_split"]),
        log_names=as_str_list(filter_cfg.get("log_names")),
        tokens=as_str_list(filter_cfg.get("tokens")),
        num_history_frames=int(filter_cfg.get("num_history_frames", 4)),
    )


def _frame_sensor_paths(frame: Dict[str, Any], cameras: Sequence[str], lidar: bool) -> List[str]:
    """Relative sensor paths a frame needs, mirroring ``Cameras.from_camera_dict`` / ``Lidar.from_paths``."""
    cams = {name.lower(): spec for name, spec in frame["cams"].items()}
    paths = []
    for camera in cameras:
        if camera not in cams:
            raise SelectionError(f"Camera '{camera}' not in log frame; available: {sorted(cams)}")
        paths.append(str(cams[camera]["data_path"]))
    if lidar:
        paths.append(str(frame["lidar_path"]))
    return paths


def check_sensors(
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    split_filter: SplitFilter,
    cameras: Sequence[str],
    lidar: bool,
    history_frames: int = 1,
    max_scenarios: Optional[int] = None,
    max_logs: Optional[int] = None,
) -> SensorCheck:
    """Find the scenarios of a split whose sensor files exist.

    :param history_frames: frames the policy loads sensors for, counted back from the current
        frame. The NAVSIM TransFuser baselines load only the current frame (1).
    :param max_scenarios: stop after this many available scenarios (smoke test).
    :param max_logs: stop after this many logs have contributed a scenario (smoke test).
    """
    if not navsim_log_path.is_dir():
        raise SelectionError(f"Log directory not found: {navsim_log_path}")
    if not sensor_blobs_path.is_dir():
        raise SelectionError(f"Sensor directory not found: {sensor_blobs_path}")
    if split_filter.tokens is None or split_filter.log_names is None:
        raise SelectionError(
            f"Split '{split_filter.split}' does not list log names and tokens explicitly; "
            "subset selection is only implemented for token-defined splits such as navtest."
        )
    if history_frames < 1 or history_frames > split_filter.num_history_frames:
        raise SelectionError(
            f"history_frames must be in [1, {split_filter.num_history_frames}], got {history_frames}"
        )

    wanted = set(split_filter.tokens)
    seen = set()
    result = SensorCheck()

    for log_name in sorted(split_filter.log_names):
        if max_scenarios is not None and len(result.available_tokens) >= max_scenarios:
            break
        if max_logs is not None and len(result.available_log_names) >= max_logs:
            break

        log_file = navsim_log_path / f"{log_name}.pkl"
        if not log_file.is_file():
            result.missing_logs.append(log_name)
            continue
        with open(log_file, "rb") as handle:
            frames = pickle.load(handle)
        result.num_logs_read += 1

        contributed = False
        for index, frame in enumerate(frames):
            token = frame["token"]
            if token not in wanted:
                continue
            seen.add(token)

            first = index - (history_frames - 1)
            needed: List[str] = []
            for frame_index in range(max(first, 0), index + 1):
                needed.extend(_frame_sensor_paths(frames[frame_index], cameras, lidar))
            absent = [p for p in needed if not (sensor_blobs_path / p).is_file()]

            if first < 0 or absent:
                result.missing_tokens.append(token)
                if absent and len(result.example_missing_files) < 5:
                    result.example_missing_files.append(absent[0])
                continue

            result.available_tokens.append(token)
            contributed = True
            if max_scenarios is not None and len(result.available_tokens) >= max_scenarios:
                break
        if contributed:
            result.available_log_names.append(log_name)

    stopped_early = max_scenarios is not None or max_logs is not None
    if not stopped_early:
        result.unseen_tokens = sorted(wanted - seen)
    return result
