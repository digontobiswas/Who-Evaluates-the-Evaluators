"""Collect the summary.json files under results/ into one summary CSV per experiment group.

    python -m wete.cli.export_tables [--results results] [--out results/tables]

Writes, for whatever exists:
    stage1_navsim_navtest.csv    one row per (policy, seed): PDMS, sub-metrics, validity, published value
    popA_navsim_navtest.csv      one row per (policy, degradation, level): PDMS, open-loop L2, validity
    bridgesim_navhard.csv        one row per BridgeSim run: EPDMS, DS, RC, sub-metrics, published value
    README.txt                   what each file is and where its numbers come from

Every number is copied from a summary.json written by a cluster job; nothing is computed here
except unit conversion (fractions to percent where the column name says so). Smoke and toy runs
are left out on purpose: they are pipeline checks, not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def stage1_rows(results: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(results.glob("stage1_pdms_*_navtest/summary.json")):
        s = _load(path)
        r, c = s["result"], s.get("comparison") or {}
        row = {
            "experiment": s["experiment"],
            "policy": s["policy"],
            "split": s.get("split"),
            "pdms": r["pdms"],
            "pdms_sem": 100 * r["score_sem"] if r.get("score_sem") is not None else None,
            "num_valid": r["num_valid"],
            "num_expected": r["num_expected"],
            "complete": r["complete"],
            "published_pdms": c.get("reference_pdms"),
            "delta_vs_published": c.get("delta_pdms"),
            "published_source": c.get("reference_source"),
        }
        row.update({f"mean_{k}": v for k, v in (r.get("mean") or {}).items()})
        row["checkpoint_sha256"] = (s.get("checkpoint") or {}).get("sha256")
        row["source_file"] = str(path)
        rows.append(row)
    return rows


def popa_rows(results: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(results.glob("popA/*/*/summary.json")):
        s = _load(path)
        r, d, ol = s["result"], s["degradation"], s.get("open_loop") or {}
        rows.append({
            "policy": s["policy"],
            "degradation": d["name"],
            "level": d["level"],
            "parameter": d.get("value"),
            "pdms": r["pdms"],
            "pdms_sem": 100 * r["score_sem"] if r.get("score_sem") is not None else None,
            "num_valid": r["num_valid"],
            "num_expected": r["num_expected"],
            "complete": r["complete"],
            "open_loop_ade_m": ol.get("mean_ade"),
            "open_loop_fde_m": ol.get("mean_fde"),
            "open_loop_l2_1s_m": ol.get("mean_l2_1s"),
            "open_loop_l2_2s_m": ol.get("mean_l2_2s"),
            "open_loop_l2_3s_m": ol.get("mean_l2_3s"),
            "cameras_blanked": (ol.get("effective_k") or [None])[0],
            **{f"mean_{k}": v for k, v in (r.get("mean") or {}).items()},
            "source_file": str(path),
        })
    return sorted(rows, key=lambda x: (x["policy"], x["degradation"] != "none", x["degradation"], x["level"]))


def bridgesim_rows(results: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(results.glob("stage1_bridgesim_navhard*/summary.json")):
        s = _load(path)
        r, c = s["result"], s.get("comparison") or {}
        ref = c.get("reference") or {}
        rows.append({
            "experiment": s["experiment"],
            "policy": s["policy"],
            "extra_evaluator_args": " ".join(s.get("extra_evaluator_args") or []) or "none",
            "epdms": r["epdms"],
            "epdms_sem": r.get("epdms_sem"),
            "ds": r["ds"],
            "ds_sem": r.get("ds_sem"),
            "rc": r["rc"],
            "num_scored": r["num_scored"],
            "num_expected": r["num_expected"],
            "complete": r["complete"],
            "published_epdms": ref.get("epdms"),
            "published_ds": ref.get("ds"),
            "published_rc": ref.get("rc"),
            "delta_epdms": c.get("delta_epdms"),
            "delta_ds": c.get("delta_ds"),
            "delta_rc": c.get("delta_rc"),
            **{f"mean_{k}": v for k, v in (r.get("mean") or {}).items()},
            "source_file": str(path),
        })
    return rows


README = """Summary tables built from the summary.json files of the cluster jobs (wete.cli.export_tables).

stage1_navsim_navtest.csv
  NAVSIM v1 PDMS on navtest (12,146 scenarios) for the published navsim_baselines checkpoints,
  three training seeds each. published_pdms: NAVSIM paper (arXiv:2406.15349v2), Table 1.
  The stage 1 check compares the three-seed mean with the published value.

popA_navsim_navtest.csv
  Population A: TransFuser and Latent TransFuser (seed 0) with graded degradations (level 0 =
  clean, levels 1-3 = increasing severity), scored by NAVSIM v1 PDMS on navtest, plus open-loop
  L2 of the executed plan against the logged trajectory (metres). Descriptive only: whether each
  degradation is a valid ground-truth ordering is decided by the monotonicity test on the
  closed-loop reference evaluator, which has not run yet.

bridgesim_navhard.csv
  Closed-loop evaluation in BridgeSim (MetaDrive) on the 421 BridgeSim-NavHard scenarios.
  EPDMS, DS (driving score) and RC (route completion) in percent, averaged over scenarios the way
  BridgeSim's leaderboard does. published_*: BridgeSim README leaderboard (commit ea5d45f).
  extra_evaluator_args shows protocol variants (e.g. runtime LiDAR on).

Per-scenario data for every run: results/<experiment>/per_scenario.csv (and results/popA/...).
Columns mean_* are the evaluator's sub-metrics as fractions in [0, 1].
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args(argv)
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in (("stage1_navsim_navtest.csv", stage1_rows(results)),
                       ("popA_navsim_navtest.csv", popa_rows(results)),
                       ("bridgesim_navhard.csv", bridgesim_rows(results))):
        if rows:
            pd.DataFrame(rows).to_csv(out / name, index=False, float_format="%.4f")
            written.append(f"{name}: {len(rows)} rows")
    (out / "README.txt").write_text(README)
    print("\n".join(written) or "no summary.json files found")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
