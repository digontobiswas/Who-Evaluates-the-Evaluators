Summary tables built from the summary.json files of the cluster jobs (wete.cli.export_tables).

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

popA_bridgesim_navhard.csv
  Population A in BridgeSim (closed loop, 421 NavHard scenarios): the same policies and
  degradations as popA_navsim_navtest.csv, plus latency, which exists only in closed loop.
  EPDMS, DS and RC in percent with per-scenario standard errors. BridgeSim is the reference
  evaluator: the monotonicity test that validates each degradation runs on these scores.

bridgesim_navhard.csv
  Closed-loop evaluation in BridgeSim (MetaDrive) on the 421 BridgeSim-NavHard scenarios.
  EPDMS, DS (driving score) and RC (route completion) in percent, averaged over scenarios the way
  BridgeSim's leaderboard does. published_*: BridgeSim README leaderboard (commit ea5d45f).
  extra_evaluator_args shows protocol variants (e.g. runtime LiDAR on).

popB_navsim_navtest.csv, popB_bridgesim_navhard.csv
  Population B: the public navsim_baselines checkpoints (TransFuser, LTF, Ego Status MLP, seeds
  0-2), undegraded, on NAVSIM v1 navtest (PDMS and open-loop L2) and in BridgeSim NavHard (EPDMS,
  DS, RC). TransFuser and LTF seed 0 are the Population A clean runs (same numbers).

popA_navsim_v2_navtest.csv, popB_navsim_v2_navtest.csv
  The same populations scored by NAVSIM v2 EPDMS, one stage, on navtest (column pdms holds EPDMS).

popA_navsim_v2_navhard.csv, popB_navsim_v2_navhard.csv
  NAVSIM v2 two-stage pseudo-simulation on navhard_two_stage: pdms holds the devkit's combined
  EPDMS, with stage one and stage two separately. Latent TransFuser and Ego Status MLP only: the
  synthetic scenes have no LiDAR, which TransFuser needs.

popA_bridgesim_heldout.csv, popB_bridgesim_heldout.csv
  BridgeSim on held-out navtest scenes that are not in NavHard (wete.closed_loop.heldout: the first
  420 converted candidates in a fixed hash order). Same protocol and policies as NavHard. NavHard
  is used to validate degradations; every evaluator is measured on these scenes (analysis
  --heldout). Population B rows for TransFuser, LTF and DiffusionDrive seed 0 are the clean runs.

The NAVSIM v1 tables are unchanged; v1 and v2 are separate evaluators, reported side by side.

Column arm: "specificity" marks the benign perturbations tail_lateral and tail_stretch, which apply
lateral_noise / late_braking only beyond the first 1.5 s and 10 m of the plan (the part BridgeSim's
replanning controller never executes). A correct closed-loop score should not move under them.

Per-scenario data for every run: results/<experiment>/per_scenario.csv (and results/popA/...).
Columns mean_* are the evaluator's sub-metrics as fractions in [0, 1].
