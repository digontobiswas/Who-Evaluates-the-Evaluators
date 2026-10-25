> README of the `wete` package as received from the senior team on 25 Oct 2026, kept for its design notes (section 4). Paths in it refer to that package's layout; the code now lives in `src/wete/`. The toy-world numbers in it validate the pipeline only and are not results.

# wete — Who Evaluates the Evaluators?

Implementation of the CVPR 2027 project *"Who Evaluates the Evaluators? Ground-Truth
Ordering Tests and Reliability-Weighted Scoring for End-to-End Driving Benchmarks"*.

**Status (9 Oct 2026).** The harness, statistics, reliability method and evaluator
adapters are implemented and tested. **No experiment on a real evaluator has been
run.** Nothing in this repository is evidence about NAVSIM, HUGSIM or any world
model. The toy-world numbers below validate the *machinery* on planted ground
truth; they must never appear in the paper as findings.

---

## 1. What is implemented

| Proposal component | Where | State |
|---|---|---|
| 7 graded degradations (Population A), common random numbers across levels | `wete/degradations.py` | done, unit-tested |
| Level calibration on a disjoint pilot set (**new step**, §4.1) | `degradations.calibrate_levels` | done, unit-tested |
| Monotonicity test (paired bootstrap, intersection–union) | `wete/stats.py` | done; type-I error ≈ 5 % per pair, conjunction ≤ 5 % |
| Ordering recovery (τ, pair accuracy, P(full order)), bootstrap CIs | `wete/stats.py` | done |
| Agreement (τ, ρ, MMRV, exact permutation test) for Populations B/C | `wete/stats.py` | done |
| Sample efficiency (scenarios to stable ranking) | `wete/stats.py` | done |
| Reliability labels, r(s), weighted score S_w, held-out CV (base × degradation × scenario folds) | `wete/reliability.py` | done; 3 label modes (§4.3) |
| Label-noise ceiling for H3 (**new**) | `reliability.label_ceiling_auroc` | done |
| Fallback discriminative subset | `reliability.discriminative_subset_cv` | done |
| H1–H4 pre-registered verdicts (supported / contradicted / inconclusive) | `wete/analysis.py` | done |
| LaTeX tables that drop into `main.tex` (`tab:matrix`, `tab:reliability`) | `wete/analysis.py` | done |
| NAVSIM v1 (navtest PDMS) + open-loop L2: agent wrapper | `adapters/navsim/wete_navsim/degraded_agent.py` | done; **stub-tested only** |
| NAVSIM v2 (navhard EPDMS) **per-scenario** scores | `adapters/navsim/wete_navsim/run_pdm_score_per_scenario.py` | done; **not run** |
| HUGSIM closed loop: pipe relay (no changes to HUGSIM or the policy) | `adapters/hugsim/` | done; protocol-tested with fake env/policy over real FIFOs |
| World model: interface, closed-loop rollout, temporal-consistency feature | `adapters/world_model/wete_wm/rollout.py` | interface only — model not chosen |
| Job generation, output collection | `scripts/make_jobs.py`, `adapters/collect.py` | done, tested on synthetic native outputs |
| Toy world with planted blind spots + validation checks V1–V6 | `wete/toysim.py`, `scripts/validate_on_toy.py` | done; all checks pass |

Pinned upstreams the adapters were written against: NAVSIM `0a380a9` (27 Oct 2025),
HUGSIM `62c690d` (8 Nov 2025).

## 2. Install and test

```bash
pip install -e ".[test]"
pytest -q                                   # 37 tests, ~20 s
python scripts/validate_on_toy.py --out results/toy --n 400 --B 500   # ~8 min CPU
```

## 3. Running the real study

1. **Smoke-test the NAVSIM wrapper** in your NAVSIM environment (the unit tests use
   stubs because NAVSIM pins numpy 1.23.4, which does not build on Python 3.13):
   `pip install -e .` there, add `adapters/navsim` to `PYTHONPATH`, run one navmini
   job with `agent=degraded_transfuser_agent agent.degradation=camera_dropout agent.level=1`
   and confirm that the clean run reproduces the reported PDMS within ±1.
2. **Pilot calibration** on HUGSIM (the reference evaluator E\*): reserve ~20 % of
   scenes as pilot scenes, *never reused*. Sweep each degradation on a fine grid
   (see `toysim.PILOT_GRIDS` for the pattern) for each base policy, then call
   `calibrate_levels` per base and take the most severe choice per level. Save as
   `levels.json`. Degradations that return `None` are reported as undetectable, not forced.
3. `python scripts/make_jobs.py --config configs/study.yaml --levels levels.json > jobs.sh`
   and run the lines with your scheduler (each line is independent).
4. `python adapters/collect.py --root runs --out results/real [--hugsim-logged-paths paths.json]`
   → `scores.csv`, `features.csv`, `behavior.csv`, `collect_report.json`.
   **Inspect `collect_report.json`**: invalid scenarios that become more frequent at
   higher levels would bias results if silently dropped.
5. `python -m wete.analysis --scores results/real/scores.csv --meta configs/meta.example.json
   --features results/real/features.csv --behavior results/real/behavior.csv --out results/real`
   → `report.json`, `discriminative_matrix.csv/png`, `monotonicity.csv`,
   `sample_efficiency.png`, `tab_matrix.tex`, `tab_reliability.tex`.

`configs/meta.example.json` marks each evaluator's orientation and family, and HUGSIM as
`"reference": true` (see `wete/toysim.META` for the format).

## 4. What the implementation revealed (affects the proposal and paper)

These come from building and validating the pipeline. Items marked *toy* were
observed in the toy world: they show a failure mode is *possible*, not that it
occurs on real evaluators. Check each on real data.

### 4.1 Degradation levels must be calibrated on a disjoint pilot set
With the proposal's fixed level table, the monotonicity test discarded three of
seven degradations in the toy world, for three different reasons:
late braking **saturated** (level 1 already collided in nearly every lead-vehicle
scenario, so levels 2–3 could not score lower); one step of **latency improved**
the score for two bases; and keeping fewer history frames **helped** history-using
policies (shorter baseline, faster reaction to a braking lead). After per-base
pilot calibration on separate scenarios, all degradations except the planted
history failure were retained. Calibrating on the main scenarios would select
levels on the same data the monotonicity test uses, inflating its pass rate.

### 4.2 Facts about the real policies that constrain the design
* **TransFuser and LTF read only the current frame** (`history_steps = [3]`), so
  history removal is a no-op for both and will be discarded. Testing memory
  failures (part of H1) needs at least one base policy that consumes history.
* **TransFuser reads 3 cameras (l0, f0, r0) and fuses LiDAR.** "4 cameras blanked"
  is effectively 3 (logged as `effective_k`), and camera-only dropout may barely
  hurt it. LTF is camera-only. Report which.
* **NAVSIM keeps 1.5 s of history at 2 Hz**, so stale observations above 1.5 s are impossible there.
* **Latency is closed-loop only**, so H1 compares open-loop vs closed-loop on
  stale observations (and memory, if a history-using base passes).
* **The official NAVSIM v2 CSV cannot give per-scenario EPDMS**: the stage-2
  weights are dropped before writing. `run_pdm_score_per_scenario.py` re-runs
  upstream's own aggregation and asserts that the mean of the per-scenario scores
  reproduces the official EPDMS to 1e-6.

### 4.3 The reliability label in the proposal is confounded (*toy*, but mechanistic)
Per-scenario concordance mostly measures **how much the degradation matters in
that scenario**, not how faithful the evaluator is there (toy: ρ = +0.72 with the
reference evaluator's effect size). On a positive control with a known, strong
reliability signal:

| Label definition | AUROC of r(s) [95 % CI] | Gain in separation z |
|---|---|---|
| Absolute (proposal) | 0.546 [0.520, 0.570] | +0.21 |
| Behaviour-conditioned (new; no reference needed) | 0.591 [0.563, 0.618] | +0.25 |
| Effect-conditioned (needs E\* on the same scenarios) | **0.779** [0.750, 0.807] | +0.30 |
| Pure-noise feature (negative control) | 0.48–0.49, CIs cover 0.5 | ≈ 0 |

Effect-conditioned labels work best but need the reference evaluator on the
*same* scenarios, which HUGSIM does not provide for navtest/navhard. Behaviour-
conditioned labels (keep a scenario only if the degradation changes the policy's
plan) are evaluator-independent and partially fix the problem. Recommendation:
report all three; treat absolute-label results as a lower bound.

On the toy's realistic 3DGS analogue, reliability weighting from the deviation
feature *hurt* separation (gain z = −0.18 [−0.19, −0.16], behaviour-conditioned).
The H4 test catches this, but it means a negative H4 is a live outcome.

### 4.4 The ordering test measures sensitivity, not specificity (*toy*)
With 400 scenarios, τ saturates at 1.00 for almost every sensitive evaluator, so
the τ matrix (proposal `tab:matrix`) may not discriminate. The sample-efficiency
metric (scenarios needed for P(full order) ≥ 0.9) does. But it ranked **open-loop
L2 as the most efficient detector of nearly every degradation** (10–50 scenarios),
ahead of the closed-loop reference: every degradation moves the plan away from the
expert, which is exactly what L2 measures. An evaluator can pass Population A and
still mis-rank real policies.

**Recommended protocol addition:** a *specificity* arm of benign perturbations
(for example a different but safe speed or lane position, verified by E\* as not worse)
where a good evaluator should show **no** drop, tested with an equivalence test
(TOST). Open-loop L2 would fail it, which is the known failure the paper wants to
expose. Without it, a reviewer can argue the ordering test rewards plan-distance metrics.

### 4.5 H3's fixed 0.7 threshold needs a reference point
With 4 levels a scenario label rests on 6 pairwise comparisons per family. In a
unit test where the predictor *is* the true planted variable, AUROC was ≈ 0.67.
The analysis now reports a split-half **label-noise ceiling** next to every AUROC;
interpret H3 relative to it.

### 4.6 Statistics notes
* Bootstrap unit = scenarios. CV folds share scenarios, so treating folds as
  independent gave a falsely narrow AUROC CI for a pure-noise feature
  ([0.505, 0.530]); the scenario bootstrap fixes it (coverage 11/12 replicates).
* The monotonicity test is an intersection–union test: no multiplicity correction
  is needed (it is conservative).
* Verdicts are three-way. "Inconclusive" (CI too wide) is not evidence of absence;
  "contradicted" requires the CI to exclude the hypothesised effect beyond a 0.1 τ margin.

## 5. Toy validation (pipeline checks on planted ground truth)

`scripts/validate_on_toy.py`, n = 400 scenarios, B = 500, all pass:

| Check | Planted truth | Result |
|---|---|---|
| V1 null evaluator | random scores | τ CI covers 0 in 6/6 cells; monotonicity false-pass 0/18 |
| V2 planted blindness | world model ignores lateral position | τ = −0.11 [−0.39, 0.56] |
| V3 sensitivity | reference sees all retained degradations | min τ lower bound 0.94 |
| V4 protocol discards | one base ignores history | history removal discarded (uncalibratable in pilot) |
| V5 reliability, positive/negative control | known corruption probability | see table in §4.3 |
| V6 behaviour-conditioned labels | same control, no reference | AUROC 0.591 [0.563, 0.618], gain > 0 |

H1 and H2 come out "supported" on the toy. That reflects how the toy was built and
says nothing about the hypotheses.

## 6. Not done / not verified

* No run on NAVSIM, HUGSIM or a world model. Compute, data and licences are on your side.
* The NAVSIM wrapper is tested against stubs that mirror the dataclass fields at
  `0a380a9`; a real smoke test (step 1) is required.
* The HUGSIM relay is tested with a fake env and policy over real FIFOs; it has
  not run with HUGSIM. Latency is a plan-lag approximation (no re-projection into
  the current ego frame); check it on 2–3 scenes.
* HUGSIM per-scenario deviation needs the recorded ego path per scene
  (`--hugsim-logged-paths`); extracting it is dataset-specific.
* World model: model not chosen; `score_rollout` needs agent layout from the model
  or a fixed detector, and that choice must be reported.
* Specificity arm (§4.4): recommended, not implemented.

## Layout

```
wete/                 core library (no simulator dependencies)
adapters/navsim/      DegradedAgent, per-scenario v2 runner, Hydra agent configs
adapters/hugsim/      pipe relay, relay.sh, launcher
adapters/world_model/ interface + rollout + consistency feature
adapters/collect.py   native outputs -> long-format tables
scripts/              make_jobs.py, validate_on_toy.py
configs/              study.example.yaml
tests/                37 unit/integration tests
results/toy/          toy validation outputs (pipeline check only)
```
