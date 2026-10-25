from pathlib import Path

import pytest

from wete.navsim import devkit_v1


def test_tokens_are_always_quoted():
    # Unquoted, Hydra would parse the first as a float and the second as an int.
    assert devkit_v1.hydra_list(["12e4567890123456", "1234567890123456"]) == '["12e4567890123456","1234567890123456"]'


def test_quote_escapes_backslash_and_quote():
    assert devkit_v1.hydra_quote('a"b\\c') == '"a\\"b\\\\c"'


def test_worker_overrides_bound_parallelism():
    assert devkit_v1.worker_overrides({"name": "ray_distributed_no_torch"}, 8) == [
        "worker=ray_distributed_no_torch",
        "worker.threads_per_node=8",
    ]
    assert devkit_v1.worker_overrides({"name": "single_machine_thread_pool", "use_process_pool": True}, 4) == [
        "worker=single_machine_thread_pool",
        "worker.max_workers=4",
        "worker.use_process_pool=true",
    ]
    assert devkit_v1.worker_overrides({"name": "sequential"}, 4) == ["worker=sequential"]
    with pytest.raises(devkit_v1.DevkitError):
        devkit_v1.worker_overrides({"name": "nope"}, 1)


def test_full_split_run_passes_no_scene_filter_overrides():
    command = devkit_v1.pdm_score_command(
        devkit_root=Path("/dk"),
        split="navtest",
        navsim_log_path=Path("/d/logs"),
        sensor_blobs_path=Path("/d/sensors"),
        cache_dir=Path("/c"),
        output_dir=Path("/o"),
        experiment_name="exp",
        agent="transfuser_agent",
        agent_overrides=["agent.config.latent=false"],
        checkpoint=Path("/w/tf.ckpt"),
        worker={"name": "single_machine_thread_pool"},
        num_workers=2,
    )
    assert command[1] == "/dk/navsim/planning/script/run_pdm_score.py"
    assert 'agent.checkpoint_path="/w/tf.ckpt"' in command
    assert not any("scene_filter" in arg for arg in command)


def test_subset_run_restricts_logs_and_tokens():
    command = devkit_v1.metric_cache_command(
        devkit_root=Path("/dk"),
        split="navtest",
        navsim_log_path=Path("/d/logs"),
        cache_dir=Path("/c"),
        worker={"name": "sequential"},
        num_workers=1,
        log_names=["log_a"],
        tokens=["aa11"],
    )
    assert 'train_test_split.scene_filter.log_names=["log_a"]' in command
    assert 'train_test_split.scene_filter.tokens=["aa11"]' in command


def test_read_metric_cache_tokens(tmp_path):
    assert devkit_v1.read_metric_cache_tokens(tmp_path) == set()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "cache_metadata_node_0.csv").write_text(
        "file_name\n/x/cache/log_a/unknown/tok1/metric_cache.pkl\n/x/cache/log_a/unknown/tok2/metric_cache.pkl\n"
    )
    assert devkit_v1.read_metric_cache_tokens(tmp_path) == {"tok1", "tok2"}


def test_wrapper_detection_and_override_rebasing():
    assert not devkit_v1.is_wrapped({"name": "none", "level": 0})
    assert devkit_v1.is_wrapped({"name": "none", "level": 0, "wrap": True})
    assert devkit_v1.is_wrapped({"name": "stale_obs", "level": 1})
    assert devkit_v1.rebase_agent_overrides(["agent.config.latent=true", "worker=sequential"]) == [
        "agent.base_agent.config.latent=true",
        "worker=sequential",
    ]
    assert devkit_v1.wrapper_overrides({"name": "camera_dropout", "level": 2, "value": None}, ["cam_f0"]) == [
        "agent.degradation=camera_dropout",
        "agent.level=2",
        'agent.used_cameras=["cam_f0"]',
    ]
    assert "agent.value=0.25" in devkit_v1.wrapper_overrides({"name": "late_braking", "level": 1, "value": 0.25}, None)


def test_sensor_preflight_looks_back_far_enough():
    f = devkit_v1.required_history_frames
    assert f({"name": "none", "level": 0}, 1) == 1
    assert f({"name": "camera_dropout", "level": 3}, 1) == 1
    assert [f({"name": "stale_obs", "level": k}, 1) for k in (1, 2, 3)] == [2, 3, 4]
    assert f({"name": "stale_obs", "level": 1, "value": 1.0}, 1) == 3
    assert f({"name": "history_removal", "level": 1}, 1) == 4


def test_pdm_score_command_keeps_devkit_searchpath(tmp_path):
    command = devkit_v1.pdm_score_command(
        devkit_root=tmp_path,
        split="navtest",
        navsim_log_path=tmp_path,
        sensor_blobs_path=tmp_path,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        experiment_name="x",
        agent="degraded_transfuser_agent",
        agent_overrides=[],
        checkpoint=tmp_path / "a.ckpt",
        worker={"name": "sequential"},
        num_workers=1,
        checkpoint_key=devkit_v1.WRAPPER_CHECKPOINT_KEY,
        config_searchpath=tmp_path,
    )
    searchpath = [c for c in command if c.startswith("hydra.searchpath=")]
    assert searchpath == [f'hydra.searchpath=["{devkit_v1.DEVKIT_SEARCHPATH}","file://{tmp_path.resolve()}"]']
    assert any(c.startswith("agent.base_agent.checkpoint_path=") for c in command)
    plain = devkit_v1.pdm_score_command(
        devkit_root=tmp_path, split="navtest", navsim_log_path=tmp_path, sensor_blobs_path=tmp_path,
        cache_dir=tmp_path, output_dir=tmp_path, experiment_name="x", agent="transfuser_agent",
        agent_overrides=[], checkpoint=tmp_path / "a.ckpt", worker={"name": "sequential"}, num_workers=1,
    )
    assert not any(c.startswith("hydra.searchpath") for c in plain)
    assert any(c.startswith("agent.checkpoint_path=") for c in plain)


def test_wrapper_hydra_configs_exist_for_every_policy():
    import yaml

    repo = Path(__file__).resolve().parents[1]
    for policy_file in (repo / "configs" / "policies").glob("*.yaml"):
        agent = yaml.safe_load(policy_file.read_text())["policy"]["degraded_agent"]
        config = repo / devkit_v1.WRAPPER_CONFIG_DIR / "agent" / f"{agent}.yaml"
        assert config.is_file(), config
        assert yaml.safe_load(config.read_text())["_target_"] == "wete.navsim.degraded_agent.DegradedAgent"
