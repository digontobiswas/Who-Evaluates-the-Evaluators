import pickle
from pathlib import Path

import pytest
import yaml

from wete.navsim import selection

CAMERAS = ["cam_f0", "cam_l0"]


def make_devkit(root: Path, log_names, tokens):
    split_dir = root / selection.SPLIT_CONFIG_DIR
    (split_dir / "scene_filter").mkdir(parents=True)
    (split_dir / "navtest.yaml").write_text("defaults:\n  - scene_filter: navtest\n\ndata_split: test\n")
    (split_dir / "scene_filter" / "navtest.yaml").write_text(
        yaml.safe_dump({"num_history_frames": 4, "log_names": log_names, "tokens": tokens})
    )


def make_log(logs: Path, sensors: Path, log_name: str, tokens, missing=()):
    """Write a log pickle shaped like OpenScene's, and the sensor files it refers to."""
    frames = []
    for token in tokens:
        frame = {
            "token": token,
            "cams": {
                "CAM_F0": {"data_path": f"{log_name}/CAM_F0/{token}.jpg"},
                "CAM_L0": {"data_path": f"{log_name}/CAM_L0/{token}.jpg"},
            },
            "lidar_path": f"{log_name}/MergedPointCloud/{token}.pcd",
        }
        frames.append(frame)
        if token in missing:
            continue
        for rel in [spec["data_path"] for spec in frame["cams"].values()] + [frame["lidar_path"]]:
            path = sensors / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"{log_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)


@pytest.fixture()
def dataset(tmp_path):
    devkit, logs, sensors = tmp_path / "devkit", tmp_path / "logs", tmp_path / "sensors"
    sensors.mkdir()
    # Split tokens: t1, t2 in log_a; t4, t5 in log_b; t7 in log_c (whose log file is absent).
    make_devkit(devkit, ["log_a", "log_b", "log_c"], ["t1", "t2", "t4", "t5", "t7"])
    make_log(logs, sensors, "log_a", ["t0", "t1", "t2", "t3"])
    make_log(logs, sensors, "log_b", ["t4", "t5", "t6"], missing={"t5"})
    return devkit, logs, sensors


def test_load_split_filter(dataset):
    devkit, _, _ = dataset
    split = selection.load_split_filter(devkit, "navtest")
    assert split.data_split == "test"
    assert split.log_names == ["log_a", "log_b", "log_c"]
    assert split.tokens == ["t1", "t2", "t4", "t5", "t7"]
    assert split.num_history_frames == 4


def test_full_check_reports_every_gap(dataset):
    devkit, logs, sensors = dataset
    split = selection.load_split_filter(devkit, "navtest")

    check = selection.check_sensors(logs, sensors, split, CAMERAS, lidar=True)

    assert check.available_tokens == ["t1", "t2", "t4"]
    assert check.missing_tokens == ["t5"]
    assert check.missing_logs == ["log_c"]
    assert check.unseen_tokens == ["t7"]
    assert check.example_missing_files and check.example_missing_files[0].startswith("log_b/")


def test_subset_stops_at_limits(dataset):
    devkit, logs, sensors = dataset
    split = selection.load_split_filter(devkit, "navtest")

    by_count = selection.check_sensors(logs, sensors, split, CAMERAS, lidar=True, max_scenarios=1)
    assert by_count.available_tokens == ["t1"]
    assert by_count.available_log_names == ["log_a"]

    by_logs = selection.check_sensors(logs, sensors, split, CAMERAS, lidar=True, max_logs=1)
    assert by_logs.available_tokens == ["t1", "t2"]
    assert by_logs.available_log_names == ["log_a"]


def test_history_needs_earlier_frames(dataset):
    devkit, logs, sensors = dataset
    split = selection.load_split_filter(devkit, "navtest")

    # t4 is the first frame of log_b, so it has no earlier frame to load.
    check = selection.check_sensors(logs, sensors, split, CAMERAS, lidar=True, history_frames=2)

    assert check.available_tokens == ["t1", "t2"]
    assert set(check.missing_tokens) == {"t4", "t5"}


def test_no_camera_policy_needs_no_files(dataset):
    devkit, logs, sensors = dataset
    split = selection.load_split_filter(devkit, "navtest")

    check = selection.check_sensors(logs, sensors, split, cameras=[], lidar=False)

    assert check.available_tokens == ["t1", "t2", "t4", "t5"]


def test_unknown_camera_is_an_error(dataset):
    devkit, logs, sensors = dataset
    split = selection.load_split_filter(devkit, "navtest")
    with pytest.raises(selection.SelectionError, match="cam_b0"):
        selection.check_sensors(logs, sensors, split, ["cam_b0"], lidar=False)
