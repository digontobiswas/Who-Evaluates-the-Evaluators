"""Collect native evaluator outputs into the long-format tables of wete.analysis.

Directory layout (experiment.out_dir of configs/experiments/popA_*.yaml):

  <root>/<evaluator>/A/<base>/<spec>/...     Population A; spec = none__L0 (or clean) | <deg>__L<k>
  <root>/<evaluator>/B/<policy>/...           Population B (public checkpoints)
  <root>/<evaluator>/C/<run>/e<epoch>/...     Population C (training checkpoints)

Evaluator payloads:
  navsim_v1    NAVSIM v1.1 run_pdm_score.py CSV (<timestamp>.csv), one row per token plus a
               final row whose token is "average"
  open_l2      <...>/plans/<agent>.plans.jsonl written by DegradedAgent (navtest run)
  navsim_v2    per_scenario.csv from wete_navsim.run_pdm_score_per_scenario
  hugsim       <scene>_<mode>/eval.json (+ wete_trace.jsonl)
  world_model  scores.csv with columns scenario, score[, consistency]
  bridgesim    per_scenario.csv from wete.cli.bridgesim_run (EPDMS_no_ep per scenario)

Outputs: scores.csv (long format), features.csv, behavior.csv (plan displacement
level 3 vs clean, for behaviour-conditioned reliability labels), collect_report.json (counts of
invalid / missing scenarios per policy: failures that correlate with the
degradation level would bias results if silently dropped -- inspect them).

Usage: python -m wete.collect --root $WETE_EXP_ROOT/runs --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

SPEC_RE = re.compile(r"^(?P<deg>[a-z_]+?)__L(?P<lvl>\d)$")
CLEAN_SPECS = ("clean", "none__L0")
# Summary rows the devkits append to their per-token CSV: "average" in v1.1, the rest in v2.
NAVSIM_SUMMARY_TOKENS = {"average", "average_all_frames", "extended_pdm_score_stage_one",
                         "extended_pdm_score_stage_two", "extended_pdm_score_combined"}


def _is_clean(spec_name: str) -> bool:
    return spec_name in CLEAN_SPECS


def _clean_dir(base: Path) -> Optional[Path]:
    for name in CLEAN_SPECS:
        if (base / name).is_dir():
            return base / name
    return None


def _as_bool(values: pd.Series) -> pd.Series:
    """The devkit writes True/False; pandas may read them as bool or as strings."""
    return values.astype(str).str.strip().str.lower().eq("true")


def iter_policies(ev_dir: Path) -> Iterator[Tuple[Dict, Path]]:
    for pop_dir in sorted(p for p in ev_dir.iterdir() if p.is_dir()):
        pop = pop_dir.name
        if pop == "A":
            for base in sorted(p for p in pop_dir.iterdir() if p.is_dir()):
                for spec in sorted(p for p in base.iterdir() if p.is_dir()):
                    if _is_clean(spec.name):
                        deg, lvl = "none", 0
                    else:
                        m = SPEC_RE.match(spec.name)
                        if not m:
                            continue
                        deg, lvl = m["deg"], int(m["lvl"])
                    yield {"population": "A", "base": base.name, "degradation": deg, "level": lvl,
                           "policy": f"{base.name}:{spec.name}"}, spec
        elif pop == "B":
            for pol in sorted(p for p in pop_dir.iterdir() if p.is_dir()):
                yield {"population": "B", "base": pol.name, "degradation": "none", "level": 0,
                       "policy": pol.name}, pol
        elif pop == "C":
            for run in sorted(p for p in pop_dir.iterdir() if p.is_dir()):
                for ep in sorted(p for p in run.iterdir() if p.is_dir()):
                    e = int(re.sub(r"\D", "", ep.name) or 0)
                    yield {"population": "C", "base": run.name, "degradation": "none", "level": e,
                           "policy": f"{run.name}:{ep.name}"}, ep


def _latest(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def read_navsim_v1(d: Path) -> pd.DataFrame:
    csv = _latest([p for p in d.rglob("*.csv") if p.name != "per_scenario.csv"])
    if csv is None:
        return pd.DataFrame()
    df = pd.read_csv(csv, dtype={"token": str})
    df = df[~df.token.isin(NAVSIM_SUMMARY_TOKENS)].copy()
    df.loc[~_as_bool(df["valid"]), "score"] = np.nan
    return pd.DataFrame({"scenario": df.token.astype(str), "score": df.score.astype(float)})


def _plan_log(d: Path) -> pd.DataFrame:
    f = _latest(list(d.rglob("*.plans.jsonl")) or list(d.rglob("*.l2.jsonl")))
    if f is None:
        return pd.DataFrame()
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return pd.DataFrame(recs).drop_duplicates("token", keep="last")


def read_open_l2(d: Path) -> pd.DataFrame:
    df = _plan_log(d)
    if df.empty or "ade" not in df:
        return pd.DataFrame()
    df = df.dropna(subset=["ade"])
    return pd.DataFrame({"scenario": df.token.astype(str), "score": df.ade.astype(float)})


def plans_navsim(d: Path) -> Dict[str, np.ndarray]:
    df = _plan_log(d)
    if df.empty or "poses" not in df:
        return {}
    return {str(t): np.asarray(p, float) for t, p in zip(df.token, df.poses)}


def plans_hugsim(d: Path) -> Dict[str, np.ndarray]:
    out = {}
    for tr in d.rglob("wete_trace.jsonl"):
        for line in tr.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("step") == 0 and "plan0" in r:
                out[tr.parent.name] = np.asarray(r["plan0"], float)
                break
    return out


PLAN_READERS = {"navsim_v1": plans_navsim, "navsim_v2": plans_navsim, "hugsim": plans_hugsim}
NO_BEHAVIOUR_AT_STEP0 = {"stale_obs", "latency"}      # identity at the first closed-loop step


def behaviour_table(root: Path) -> pd.DataFrame:
    """Per (base, degradation, scenario): mean waypoint displacement between the
    level-3 and clean executed plans on the same input."""
    rows = []
    for ev, reader in PLAN_READERS.items():
        ev_dir = root / ev / "A"
        if not ev_dir.exists():
            continue
        for base in sorted(p for p in ev_dir.iterdir() if p.is_dir()):
            clean_dir = _clean_dir(base)
            clean = reader(clean_dir) if clean_dir is not None else {}
            if not clean:
                continue
            for spec in sorted(p for p in base.iterdir() if p.is_dir()):
                m = SPEC_RE.match(spec.name)
                if not m or int(m["lvl"]) != 3:
                    continue
                if ev == "hugsim" and m["deg"] in NO_BEHAVIOUR_AT_STEP0:
                    continue
                deg_plans = reader(spec)
                for sc, P in deg_plans.items():
                    C = clean.get(sc)
                    if C is None or C.shape != P.shape:
                        continue
                    rows.append({"base": base.name, "degradation": m["deg"], "scenario": sc,
                                 "delta": float(np.linalg.norm(P[:, :2] - C[:, :2], axis=1).mean()),
                                 "source": ev})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["base", "degradation", "scenario"]) if not df.empty else df


def read_navsim_v2(d: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = _latest(list(d.rglob("per_scenario.csv")))
    if f is None:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(f)
    sc = pd.DataFrame({"scenario": df.scenario.astype(str), "score": df.combined.astype(float)})
    ft = pd.DataFrame({"scenario": df.scenario.astype(str), "stage_gap": df.stage_gap.astype(float)})
    return sc, ft


LOGGED_PATHS: Dict[str, np.ndarray] = {}   # scene -> (M, 2|3) recorded ego positions


def _mean_dev(ego_pos: List, path: np.ndarray) -> float:
    """Mean distance from each simulated ego position to the nearest recorded
    ego position (same world frame as info['ego_pos'])."""
    P = np.asarray([p for p in ego_pos if p is not None], float)
    if len(P) == 0:
        return float("nan")
    k = min(P.shape[1], path.shape[1])
    d = np.linalg.norm(P[:, None, :k] - path[None, :, :k], axis=-1).min(1)
    return float(d.mean())


def read_hugsim(d: Path, key: str = "hdscore") -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, feats = [], []
    for ev in sorted(d.rglob("eval.json")):
        scene = ev.parent.name
        res = json.loads(ev.read_text())
        v = res.get(key)
        v = float(np.mean(v)) if isinstance(v, list) else (np.nan if v is None else float(v))
        rows.append({"scenario": scene, "score": v})
        tr = ev.parent / "wete_trace.jsonl"
        if tr.exists():
            steps = [json.loads(l) for l in tr.read_text().splitlines() if l.strip()]
            f = {"scenario": scene, "n_steps": len(steps)}
            base_scene = scene.rsplit("_", 1)[0]          # '<scene>_<mode>' -> '<scene>'
            path = LOGGED_PATHS.get(scene, LOGGED_PATHS.get(base_scene))
            if path is not None:
                f["deviation"] = _mean_dev([s_["ego_pos"] for s_ in steps], path)
            feats.append(f)
    return pd.DataFrame(rows), pd.DataFrame(feats)


def read_world_model(d: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = d / "scores.csv"
    if not f.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(f)
    df["scenario"] = df.scenario.astype(str)
    ft = df[["scenario", "consistency"]] if "consistency" in df else pd.DataFrame()
    return df[["scenario", "score"]], ft


# Which BridgeSim column is the closed-loop score (set with --bridgesim-score).
#   DS           driving score = EPDMS (without ego progress) x route completion. Default.
#   EPDMS_no_ep  BridgeSim's EPDMS column. It leaves out ego progress, so a policy that drives less
#                and therefore commits fewer infractions scores higher; observed for image noise
#                on LTF (27 Oct). Kept for comparison, not as the reference.
BRIDGESIM_SCORE = "DS"


def read_bridgesim(d: Path) -> pd.DataFrame:
    """per_scenario.csv written by wete.cli.bridgesim_run (report), column BRIDGESIM_SCORE, in
    [0, 1]. Only a complete run's file is written there."""
    f = d / "per_scenario.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    return pd.DataFrame({"scenario": df.scenario.astype(str),
                         "score": pd.to_numeric(df[BRIDGESIM_SCORE], errors="coerce")})


READERS = {"navsim_v1": read_navsim_v1, "navsim_v2": read_navsim_v2, "bridgesim": read_bridgesim,
           "hugsim": read_hugsim, "world_model": read_world_model}
# derived evaluators: (name, source evaluator dir, reader)
DERIVED = [("open_l2", "navsim_v1", read_open_l2)]   # L2 is logged during the navtest run


def collect(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    score_rows, feat_rows, report = [], [], {}
    for ev_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in READERS):
        ev = ev_dir.name
        report[ev] = {}
        for meta, d in iter_policies(ev_dir):
            out = READERS[ev](d)
            sc, ft = out if isinstance(out, tuple) else (out, pd.DataFrame())
            if sc.empty:
                report[ev][meta["policy"]] = {"missing": True}
                continue
            report[ev][meta["policy"]] = {"n": int(len(sc)), "n_invalid": int(sc.score.isna().sum())}
            score_rows.append(sc.assign(evaluator=ev, **meta))
            for dname, src, reader in DERIVED:
                if src == ev:
                    dsc = reader(d)
                    if not dsc.empty:
                        score_rows.append(dsc.assign(evaluator=dname, **meta))
            # features: taken from each base's CLEAN run only (policy-agnostic proxy)
            if not ft.empty and meta["population"] == "A" and meta["level"] == 0:
                feat_rows.append(ft.assign(evaluator=ev, _base=meta["base"]))
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    if feat_rows:
        f = pd.concat(feat_rows, ignore_index=True)
        # average the clean-run features over bases
        features = f.drop(columns="_base").groupby(["evaluator", "scenario"], as_index=False).mean()
    else:
        features = pd.DataFrame()
    return scores, features, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bridgesim-score", default="DS", choices=["DS", "EPDMS_no_ep", "RC"],
                    help="BridgeSim column used as the closed-loop score (default DS)")
    ap.add_argument("--hugsim-logged-paths", help="JSON {scene: [[x,y(,z)],...]} of recorded ego "
                    "positions in the frame of info['ego_pos']; enables the deviation feature")
    a = ap.parse_args(argv)
    global BRIDGESIM_SCORE
    BRIDGESIM_SCORE = a.bridgesim_score
    print(f"bridgesim score column: {BRIDGESIM_SCORE}")
    if a.hugsim_logged_paths:
        LOGGED_PATHS.update({k: np.asarray(v, float) for k, v in
                             json.loads(Path(a.hugsim_logged_paths).read_text()).items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores, features, report = collect(Path(a.root))
    scores.to_csv(out / "scores.csv", index=False)
    if not features.empty:
        features.to_csv(out / "features.csv", index=False)
    (out / "collect_report.json").write_text(json.dumps(report, indent=2))
    beh = behaviour_table(Path(a.root))
    if not beh.empty:
        beh.to_csv(out / "behavior.csv", index=False)
    inval = {e: sum(v.get("n_invalid", 0) for v in r.values()) for e, r in report.items()}
    print(f"{len(scores)} score rows; invalid per evaluator: {inval}")


if __name__ == "__main__":
    main()
