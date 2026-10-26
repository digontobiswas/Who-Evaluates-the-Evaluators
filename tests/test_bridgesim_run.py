"""wete.cli.bridgesim_run: scenario bookkeeping and the leaderboard-style average."""

import math

from wete.closed_loop import bridgesim
from wete.cli import bridgesim_run as R


def _summary_csv(run_dir, ds, epdms, rc):
    run_dir.mkdir(parents=True)
    (run_dir / "driving_score_summary.csv").write_text(
        "frame_id,valid,NC,DAC,DS,EPDMS_no_ep,RC\n0,True,1,1,,,\n"
        f"AVERAGE,,1.0,1.0,{ds},{epdms},{rc}\n"
    )


def test_scenario_list_and_conversion_detection(tmp_path):
    (tmp_path / "list.txt").write_text("sd_b  sd_a\tsd_c\nsd_a\n")
    assert bridgesim.read_scenario_list(tmp_path / "list.txt") == ["sd_a", "sd_b", "sd_c"]
    (tmp_path / "scen" / "sd_a" / "sd_a_0").mkdir(parents=True)
    (tmp_path / "scen" / "sd_a" / "sd_a_0" / "sd_a.pkl").write_bytes(b"x")
    (tmp_path / "scen" / "sd_b" / "sd_b_0").mkdir(parents=True)  # started, no pkl
    assert bridgesim.converted_scenarios(tmp_path / "scen") == ["sd_a"]


def test_summary_is_the_plain_mean_and_flags_missing(tmp_path):
    _summary_csv(tmp_path / "a", 0.5, 0.8, 0.6)
    _summary_csv(tmp_path / "b", 0.3, 0.6, "")  # blank RC is skipped, as in BridgeSim
    rows = [{"scenario": s, **bridgesim.read_average_row(tmp_path / s)} for s in ("a", "b")]
    out = R.summarize(rows, ["a", "b", "c"])
    assert math.isclose(out["ds"], 40.0) and math.isclose(out["epdms"], 70.0) and math.isclose(out["rc"], 60.0)
    assert out["num_blank_values"]["RC"] == 1
    assert out["num_missing"] == 1 and out["missing_example"] == ["c"] and not out["complete"]
    assert R.summarize(rows, ["a", "b"])["complete"]


def test_evaluator_command_carries_the_protocol(tmp_path):
    run = {"traffic_mode": "log_replay", "eval_mode": "closed_loop", "controller": "pure_pursuit",
           "sim_dt": 0.1, "replan_rate": 5, "ego_replay_frames": 20, "eval_frames": 80}
    cmd = bridgesim.evaluator_command(tmp_path, "ltf", tmp_path / "c.ckpt", tmp_path / "s", tmp_path / "o", run)
    joined = " ".join(cmd)
    assert "--model-type ltf" in joined and "--replan-rate 5" in joined and "--eval-frames 80" in joined
    assert cmd[1].endswith("bridgesim/evaluation/unified_evaluator.py")
