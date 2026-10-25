# Review of the senior team's `wete` package

Received 25 Oct 2026 as `wete.zip`. Reviewed against the two NAVSIM devkits pinned in
`configs/third_party.env` (v1.1 at `3e8291b`, v2 at `0a380a9`) and the cluster's pinned package set
(`requirements/navsim-cp39-manylinux2014.lock.txt`). Nothing here was run on PARAM Shakti.

## Verdict

The statistics, reliability and analysis code is sound as software and runs unchanged under the
cluster's Python 3.9 package set: all 37 of its tests pass there. The NAVSIM wrapper and the job
generator would have failed on the cluster: the wrapper at the first instantiation, the job
generator at config composition. Both are fixed. The package is not a complete study yet. Its
reference evaluator (HUGSIM) cannot be installed on the cluster, and the world model is an
interface only.

## Defects found and fixed

| # | Where | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | `adapters/navsim/wete_navsim/degraded_agent.py`, agent YAMLs | Written for the NAVSIM v2 API (`AbstractAgent(trajectory_sampling, ...)`, `TransfuserAgent(trajectory_sampling=...)`). Our navtest PDMS uses v1.1, where neither argument exists. | Instantiating it with the real v1.1 code raises `TypeError: __init__() got an unexpected keyword argument 'trajectory_sampling'`. | `src/wete/navsim/degraded_agent.py` detects the API from `AbstractAgent`'s signature; the v1.1 Hydra configs in `configs/navsim_hydra/v1/agent/` mirror the devkit's own. |
| 2 | `scripts/make_jobs.py` | Overrides `hydra.searchpath` with only its own folder. That replaces the devkit's search path, so the devkit's configs are no longer found. | With the real v1.1 config tree, composition stops at `Could not load 'default_evaluation'`. | Our command builder keeps `pkg://navsim.planning.script.config.common` and appends ours. |
| 3 | `scripts/make_jobs.py` | Its `navsim_v1` row calls v2's `run_pdm_score_one_stage.py`, which uses the v2 scorer (extended metrics), so its numbers are not the v1 PDMS that stage 1 reproduces (84.0 etc.). | Read in the v2 source. | Population A runs through the same v1.1 job as stage 1. |
| 4 | `adapters/collect.py` | Filters only v2's summary rows; v1.1's final row (token `average`) would have been read as a scenario. | Row name read in v1.1 `run_pdm_score.py`. | Filter covers both; test added. |
| 5 | wrapper, camera dropout | TransFuser and LTF load all eight cameras in v1.1 but use three, so the logged `effective_k` said 4 at level 3. The blanking order itself was right. | v1.1 `get_sensor_config` and feature builder. | Wrapper takes `used_cameras` from the policy config. |
| 6 | wrapper, stale observation | Loaded all four history frames of every sensor, four times the sensor I/O. | Code reading. | Loads only the frame that is served as current. |

How the fixed wrapper was checked: 81 unit tests pass under Python 3.9 with the cluster's pinned
packages (the wrapper tests run against stubs of both the v1.1 and the v2 API). In addition, the
devkit's own v1.1 pdm-scoring config was composed with the exact overrides our job generates, the
wrapper was instantiated around the real v1.1 TransFuser and LTF classes, and one synthetic input
was run through it for clean, camera dropout, image noise, stale observation and late braking.
That check used random weights and torch 2.8 (the PyTorch wheel index was not reachable from the
sandbox), so the smoke job on the cluster remains the real test.

## What could not be fixed in code

- **No reference evaluator on the cluster.** The analysis validates degradations (the monotonicity
  test) on the evaluator marked `reference`, which is HUGSIM. HUGSIM needs three CUDA extensions
  built from source and an `open3d` wheel for glibc 2.27, so it does not install on PARAM Shakti
  (docs/FEASIBILITY.md, section 4). Until a closed-loop evaluator runs, the analysis can only produce
  a provisional summary (`--no-reference`). Nothing in that summary counts as a validated ordering.
- **HUGSIM relay**: tested with a fake environment over real FIFOs, never with HUGSIM.
- **World model**: interface only; no model chosen; `score_rollout` uses simplified geometry.
- **NAVSIM v2 per-scenario EPDMS** (`run_pdm_score_per_scenario.py`): not carried over yet. The v2
  functions it calls exist at `0a380a9`, but it is not wired into a job and has never run.

## Decisions for the team

1. **Degradation list for NAVSIM.** `history_removal` is a no-op for TransFuser and LTF (single
   frame), and `latency` is closed-loop only. The default grid leaves both out (5 degradations ×
   3 levels + clean = 16 jobs per base policy).
2. **Reference evaluator.** Which closed-loop evaluator serves as E\* (BridgeSim on the cluster,
   HUGSIM off the cluster, or another), since G2 depends on it.

## Not taken over

`scripts/make_jobs.py` and `configs/study.example.yaml` are replaced by the config-driven job and
`scripts/submit_population_a.sh`. `results/toy/` is synthetic output and was left out on purpose.
The package's README is kept as `docs/senior_wete_README.md` for its design notes.
