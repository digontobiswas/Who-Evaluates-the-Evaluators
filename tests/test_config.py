import pytest

from wete import config as config_lib


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_extends_merges_in_order_and_child_wins(tmp_path):
    write(tmp_path / "base.yaml", "a: 1\nb: {x: 1, y: 2}\n")
    write(tmp_path / "other.yaml", "b: {y: 20, z: 30}\n")
    child = write(tmp_path / "sub" / "child.yaml", "extends: [../base.yaml, ../other.yaml]\nb: {z: 300}\nc: ${a}\n")

    cfg = config_lib.resolve(config_lib.load_config(child))

    assert cfg == {"a": 1, "b": {"x": 1, "y": 20, "z": 300}, "c": 1}


def test_override_propagates_through_interpolation(tmp_path):
    path = write(tmp_path / "c.yaml", "policy: {seed: 0, id: 'tf_seed${policy.seed}'}\n")

    cfg = config_lib.resolve(config_lib.load_config(path, ["policy.seed=2"]))

    assert cfg["policy"]["id"] == "tf_seed2"


def test_env_interpolation_and_missing_env_fails(tmp_path, monkeypatch):
    path = write(tmp_path / "c.yaml", "root: ${oc.env:WETE_TEST_ROOT}\n")

    monkeypatch.setenv("WETE_TEST_ROOT", "/somewhere")
    assert config_lib.resolve(config_lib.load_config(path))["root"] == "/somewhere"

    monkeypatch.delenv("WETE_TEST_ROOT")
    with pytest.raises(config_lib.ConfigError):
        config_lib.resolve(config_lib.load_config(path))


def test_cyclic_extends_is_reported(tmp_path):
    write(tmp_path / "a.yaml", "extends: b.yaml\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\n")

    with pytest.raises(config_lib.ConfigError, match="Cyclic"):
        config_lib.load_config(tmp_path / "a.yaml")


def test_bad_override_and_missing_file(tmp_path):
    path = write(tmp_path / "c.yaml", "a: 1\n")
    with pytest.raises(config_lib.ConfigError):
        config_lib.load_config(path, ["no_equals_sign"])
    with pytest.raises(config_lib.ConfigError):
        config_lib.load_config(tmp_path / "absent.yaml")


def test_require_lists_missing_keys():
    with pytest.raises(config_lib.ConfigError, match="a.c"):
        config_lib.require({"a": {"b": 1, "c": None}}, ["a.b", "a.c"])


def test_git_commit_falls_back_to_submit_time_value(tmp_path, monkeypatch):
    from wete import provenance

    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    monkeypatch.setenv("WETE_TEST_COMMIT", "abc123")
    assert provenance.git_commit(not_a_repo, "WETE_TEST_COMMIT") == "submit:abc123"
    monkeypatch.delenv("WETE_TEST_COMMIT")
    assert provenance.git_commit(not_a_repo, "WETE_TEST_COMMIT") is None
    assert provenance.git_commit(not_a_repo) is None
