"""The parameter-level checkpoint check of wete.cli.bridgesim_probe, with a stand-in adapter.

BridgeSim's adapters load with strict=False, so a checkpoint whose keys do not match loads
"successfully" with random weights. The probe must catch that.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from wete.cli import bridgesim_probe as P  # noqa: E402


class _Adapter:
    """Mimics BridgeSim's TransfuserAdapter: strips a prefix, loads with strict=False."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):
        torch.manual_seed(1)
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        raw = torch.load(self.checkpoint_path, map_location="cpu")
        clean = {k.replace("agent._model.", ""): v for k, v in raw["state_dict"].items()}
        self.model.load_state_dict(clean, strict=False)


@pytest.fixture
def fake_adapter(monkeypatch):
    module = types.ModuleType("fake_bridgesim_adapter")
    module.Adapter = _Adapter
    monkeypatch.setitem(sys.modules, "fake_bridgesim_adapter", module)
    monkeypatch.setitem(P.ADAPTERS, "fake", ("fake_bridgesim_adapter", "Adapter"))


def _save(path, drop=0):
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    items = list(ref.state_dict().items())[drop:]
    torch.save({"state_dict": {"agent._model." + k: v for k, v in items}}, path)


def test_complete_checkpoint_counts_every_tensor(fake_adapter, tmp_path):
    _save(tmp_path / "ok.ckpt")
    out = P.step_checkpoints([{"model_type": "fake", "checkpoint": str(tmp_path / "ok.ckpt")}])["fake"]
    assert out["all_loaded"] and out["model_tensors"] == 4 and out["loaded_from_checkpoint"] == 4


def test_missing_keys_are_caught_despite_strict_false(fake_adapter, tmp_path):
    _save(tmp_path / "bad.ckpt", drop=2)  # first layer's weight and bias absent
    out = P.step_checkpoints([{"model_type": "fake", "checkpoint": str(tmp_path / "bad.ckpt")}])["fake"]
    assert not out["all_loaded"] and out["num_not_loaded"] == 2
    assert set(out["not_loaded"]) == {"0.weight", "0.bias"}


def test_errors_are_reported_not_raised(fake_adapter, tmp_path):
    out = P.step_checkpoints([{"model_type": "fake", "checkpoint": str(tmp_path / "missing.ckpt")}])["fake"]
    assert "error" in out


def test_average_row_reader(tmp_path):
    run = tmp_path / "run" / "transfuser_rr5" / "sd_x"
    run.mkdir(parents=True)
    (run / "driving_score_summary.csv").write_text(
        "frame_id,valid,NC,DS,EPDMS_no_ep,RC\n0,True,1.0,,,\n1,True,1.0,,,\nAVERAGE,,1.0,0.5,0.7,0.6\n"
    )
    row = P._read_average_row(tmp_path / "run")
    assert row["DS"] == "0.5" and row["EPDMS_no_ep"] == "0.7"
    assert P._read_average_row(tmp_path / "empty") is None
