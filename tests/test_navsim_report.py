import pandas as pd
import pytest

from wete.navsim import report

REFERENCE = {"pdms": 84.0, "source": "paper", "note": "n"}


def devkit_csv(tmp_path, rows):
    """Write a CSV the way run_pdm_score.py does: per-token rows, then an 'average' row."""
    frame = pd.DataFrame(rows)
    average = frame.drop(columns=["token", "valid"]).mean(skipna=True)
    average["token"] = "average"
    average["valid"] = frame["valid"].all()
    frame.loc[len(frame)] = average
    path = tmp_path / "2026.01.01.00.00.00.csv"
    frame.to_csv(path)
    return path


def row(token, score, valid=True):
    if not valid:
        return {"token": token, "valid": False}
    values = {column: 1.0 for column in report.SUBMETRIC_COLUMNS}
    return {"token": token, "valid": True, **values, "score": score}


def test_summary_of_a_complete_run(tmp_path):
    path = devkit_csv(tmp_path, [row("a", 0.80), row("b", 0.89), row("0123", 0.845)])

    result = report.summarize(report.load_scores(path), expected_tokens=["a", "b", "0123"])

    assert result["num_scored"] == 3 and result["num_valid"] == 3 and result["num_invalid"] == 0
    assert result["pdms"] == pytest.approx(84.5)
    assert result["matches_devkit_average"] is True
    assert result["num_missing"] == 0 and result["num_unexpected"] == 0
    assert result["complete"] is True

    comparison = report.compare_to_reference(result, REFERENCE, tolerance_pdms=1.0)
    assert comparison["delta_pdms"] == pytest.approx(0.5)
    assert comparison["within_tolerance"] is True


def test_invalid_scenarios_make_the_run_incomplete_and_withhold_a_verdict(tmp_path):
    path = devkit_csv(tmp_path, [row("a", 0.84), row("b", 0.84), row("c", 0.0, valid=False)])

    result = report.summarize(report.load_scores(path), expected_tokens=["a", "b", "c"])

    assert result["num_invalid"] == 1
    assert result["pdms"] == pytest.approx(84.0)  # average over the survivors only
    assert result["complete"] is False
    comparison = report.compare_to_reference(result, REFERENCE, tolerance_pdms=1.0)
    assert comparison["comparable"] is False and comparison["within_tolerance"] is None


def test_missing_scenarios_make_the_run_incomplete(tmp_path):
    path = devkit_csv(tmp_path, [row("a", 0.84)])

    result = report.summarize(report.load_scores(path), expected_tokens=["a", "b"])

    assert result["num_missing"] == 1 and result["complete"] is False


def test_out_of_tolerance(tmp_path):
    path = devkit_csv(tmp_path, [row("a", 0.80)])
    result = report.summarize(report.load_scores(path), expected_tokens=["a"])
    comparison = report.compare_to_reference(result, REFERENCE, tolerance_pdms=1.0)
    assert comparison["within_tolerance"] is False


def test_find_score_csv_picks_newest(tmp_path):
    with pytest.raises(report.ReportError):
        report.find_score_csv(tmp_path)
    older = tmp_path / "a.csv"
    newer = tmp_path / "b.csv"
    older.write_text("x")
    newer.write_text("x")
    import os

    os.utime(older, (1, 1))
    assert report.find_score_csv(tmp_path) == newer


def test_summary_json_is_strict(tmp_path):
    import json

    from wete.cli import navsim_v1

    path = devkit_csv(tmp_path, [row("a", 0.80)])  # one scenario: the standard deviation is undefined
    result = report.summarize(report.load_scores(path), expected_tokens=["a"])
    navsim_v1._write_json(tmp_path / "s.json", {"result": result})

    loaded = json.loads((tmp_path / "s.json").read_text())
    assert loaded["result"]["score_std"] is None
    assert loaded["result"]["pdms"] == pytest.approx(80.0)


def test_summarize_plans_last_line_per_token_wins(tmp_path):
    import json

    from wete.navsim import report as rep

    plans = tmp_path / "plans"
    plans.mkdir()
    lines = [
        {"token": "a", "ade": 1.0, "fde": 2.0, "l2_1s": 0.5, "effective_k": 3},
        {"token": "b", "ade": 3.0, "fde": 4.0, "effective_k": 3},
        {"token": "a", "ade": 2.0, "fde": 2.0, "effective_k": 3},  # retried token: last line counts
        {"token": "c", "l2_error": "IndexError"},
    ]
    (plans / "x.plans.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    out = rep.summarize_plans(plans)
    assert out["num_tokens"] == 3 and out["num_with_l2"] == 2
    assert out["mean_ade"] == 2.5 and out["effective_k"] == [3]
    assert rep.summarize_plans(tmp_path / "missing")["num_tokens"] == 0
