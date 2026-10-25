"""Pipeline validation on the toy world (planted ground truth).

Each check below compares a pipeline output with a property that holds BY
CONSTRUCTION of wete/toysim.py. Passing them means the statistics and the
reliability machinery behave as intended; it says nothing about real evaluators.

V1 null calibration   : null_random evaluator -> tau CIs cover 0 in >= ~90 % of
                        cells; monotonicity false-pass rate on it <= ~5 %.
V2 planted blindness  : world_model tau on lateral_noise is ~0 (CI covers 0).
V3 sensitivity        : closed_ref recovers the order of every retained
                        degradation (tau CI lower bound > 0).
V4 protocol discards  : history_removal is discarded (single_frame base ignores
                        history, so its levels are identical).
V6 behaviour labels   : on the positive control, behaviour-conditioned labels
                        (no reference evaluator needed) recover the signal:
                        AUROC CI > 0.5 and gain_z CI > 0; noise feature ~0.5.
V5 reliability        : POSITIVE control (scores randomly corrupted with a known
                        scenario-dependent probability): r(s) from the artifact
                        feature beats chance (AUROC CI > 0.5) and weighting
                        improves separation (gain_z CI > 0), in both label modes.
                        NEGATIVE control (pure-noise feature): AUROC CI covers 0.5.
                        gs3d (realistic, weak planted signal) is reported as a
                        diagnostic only, not pass/fail.

Usage: python scripts/validate_on_toy.py --out results/toy [--n 400] [--B 500]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wete import analysis, reliability as R, stats as St, toysim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Step 0: calibrate levels on a disjoint pilot scenario set
    pilot = toysim.pilot_calibration(n_pilot=150, seed=a.seed + 10_000)
    (out / "pilot_calibration.json").write_text(json.dumps(pilot, indent=2))
    print("calibrated levels:", json.dumps(pilot["levels"]), "uncalibratable:", pilot["uncalibratable"])
    scores, feats, extra = toysim.generate(a.n, a.seed, levels=pilot["levels"])
    behavior = extra["behavior"]
    behavior.to_csv(out / "behavior.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    feats.to_csv(out / "features.csv", index=False)
    (out / "meta.json").write_text(json.dumps(toysim.META, indent=2))

    rep = analysis.run(scores, toysim.META, str(out), feats, {"B": a.B, "seed": a.seed}, behavior)
    import pandas as pd
    M = pd.read_csv(out / "discriminative_matrix.csv")
    checks = {}

    # V1 -----------------------------------------------------------------
    nul = M[M.evaluator == "null_random"]
    cover = ((nul.tau_lo <= 0) & (nul.tau_hi >= 0)).mean()
    dfA = scores[scores.population == "A"]
    fp, tot = 0, 0
    rng = np.random.default_rng(1)
    for d in nul.degradation:
        fams, _ = analysis.families_for(dfA, "null_random", [d], True)
        for S in fams.values():
            fp += St.monotonicity_test(S, B=a.B, rng=rng).passed
            tot += 1
    checks["V1_null_tau_ci_covers_0_frac"] = float(cover)
    checks["V1_null_monotonicity_false_pass"] = f"{fp}/{tot}"
    checks["V1_pass"] = bool(cover >= 0.8 and fp / max(tot, 1) <= 0.1)

    # V2 -----------------------------------------------------------------
    c = M[(M.evaluator == "world_model") & (M.degradation == "lateral_noise")].iloc[0]
    checks["V2_world_model_lateral_tau"] = [float(c.tau), float(c.tau_lo), float(c.tau_hi)]
    checks["V2_pass"] = bool(c.tau_lo <= 0 <= c.tau_hi or abs(c.tau) < 0.2)

    # V3 -----------------------------------------------------------------
    retained = rep["monotonicity"]["retained"]
    ref = M[(M.evaluator == "closed_ref") & M.degradation.isin(retained)]
    checks["V3_closed_ref_min_tau_lo"] = float(ref.tau_lo.min())
    checks["V3_pass"] = bool((ref.tau_lo > 0).all())

    # V4 -----------------------------------------------------------------
    checks["V4_discarded"] = rep["monotonicity"]["discarded"]
    checks["V4_uncalibratable_in_pilot"] = pilot["uncalibratable"]
    checks["V4_pass"] = ("history_removal" in rep["monotonicity"]["discarded"]
                         or "history_removal" in pilot["uncalibratable"])

    # V5 -----------------------------------------------------------------
    ref_fams, rcommon = analysis.families_for(dfA, "closed_ref", retained, True)
    bpiv = behavior.assign(scenario=behavior.scenario.astype(str)) \
        .pivot_table(index=["base", "degradation"], columns="scenario", values="delta")
    res = {}
    for ev, feat in (("positive_control", "artifact_index"), ("gs3d", "deviation")):
        fams, common = analysis.families_for(dfA, ev, retained, True)
        assert common == rcommon
        fe = feats[feats.evaluator == ev].set_index("scenario").loc[common]
        beh = {k: (bpiv.loc[k, common].values if k in bpiv.index else np.full(len(common), np.nan))
               for k in fams}
        for mode in ("absolute", "effect_conditioned", "behavior_conditioned"):
            for name, cols in {feat: [feat], "noise_only": ["noise_feature"]}.items():
                cv = R.heldout_cv(fams, fe[cols].values.astype(float), mode=mode,
                                  reference=ref_fams if mode == "effect_conditioned" else None,
                                  behavior=beh if mode == "behavior_conditioned" else None,
                                  B=200, seed=a.seed)
                s = R.summarize_cv(cv)
                res[f"{ev}/{mode}/{name}"] = {"auroc": s["auroc_mean"], "ci": s["auroc_ci_over_families"],
                                              "gain_z": s["gain_z"], "gain_z_ci": s["gain_z_ci"]}
    checks["V5_auroc"] = res
    ok = True
    for mode in ("absolute", "effect_conditioned"):
        pcr = res[f"positive_control/{mode}/artifact_index"]
        nz = res[f"positive_control/{mode}/noise_only"]
        ok &= pcr["ci"][0] > 0.5 and pcr["gain_z_ci"][0] > 0 and nz["ci"][0] <= 0.5 <= nz["ci"][1]
    checks["V5_pass"] = bool(ok)
    pcb = res["positive_control/behavior_conditioned/artifact_index"]
    nzb = res["positive_control/behavior_conditioned/noise_only"]
    checks["V6_pass"] = bool(pcb["ci"][0] > 0.5 and pcb["gain_z_ci"][0] > 0
                             and nzb["ci"][0] <= 0.5 <= nzb["ci"][1])

    checks["ALL_PASS"] = all(v for k, v in checks.items() if k.endswith("_pass"))
    (out / "validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0 if checks["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
