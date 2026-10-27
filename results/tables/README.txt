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

Per-scenario data for every run: results/<experiment>/per_scenario.csv (and results/popA/...).
Columns mean_* are the evaluator's sub-metrics as fractions in [0, 1].
