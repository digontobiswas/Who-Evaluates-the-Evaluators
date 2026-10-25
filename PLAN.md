# Plan: 8 October to 16 November 2026

Target: CVPR 2027. Paper registration 10 Nov, paper deadline 16 Nov 2026 Anywhere on Earth, which
is 17 Nov 17:59 in Dhaka. That is 39 days from today.

Working rule: stages run in order, and each ends at a gate. I stop at every gate and wait for your
go or no-go. Every number in this plan that is not a date or a count read from code is marked as
an assumption; the measured value replaces it as soon as a job reports one.

## 1. Schedule

| Stage | Dates | Gate | Go/no-go check | If the check fails |
|---|---|---|---|---|
| 0. Feasibility and environment | Thu 8 to Fri 9 Oct | G0, Fri 9 Oct | Probe output back; venv builds on the login node; smoke job (64 navtest scenarios) finishes with every scenario valid. | Fix the environment using the offline options in `setup_env.sh`; nothing else starts until the smoke job passes. |
| 1. Set up evaluators | Thu 8 to Wed 14 Oct | G1, Wed 14 Oct | Each of three baselines (TransFuser, LTF, Ego Status MLP) within about 1 PDMS point of its published navtest score; the same three scored on navhard (EPDMS) against the values in the v2 paper; one closed-loop evaluator and one world model each produce a score for one policy on a handful of scenarios. | Replace the simulator that will not run (proposal, Risks). The replacement order is fixed in `docs/FEASIBILITY.md` sections 4 and 5. |
| 2. Build policy populations | Wed 14 to Tue 20 Oct | G2, Tue 20 Oct | Population A: for each kept degradation, every step lowers the closed-loop score, bootstrap P(drop > 0) > 0.95. Populations B and C: every checkpoint loads and scores. | Drop the degradation and report it as non-monotone; the check is part of the protocol. |
| 3. Run all evaluations | Wed 14 to Tue 27 Oct | G3, Tue 27 Oct | No missing (policy, evaluator) pair; per-scenario scores saved for every pair. | Reduce scenario counts for the GPU evaluators, justified by the sample-efficiency curve. |
| 4. Measure ranking quality | Wed 21 to Tue 27 Oct | G4, Tue 27 Oct | Evaluators differ in ordering recovery beyond their bootstrap intervals. | Reframe as an agreement result (proposal, stage 4). |
| 5. Reliability-weighted scoring | Wed 28 Oct to Tue 3 Nov | G5, Tue 3 Nov | AUROC of r(s) above 0.7 on held-out data. | Switch to the discriminative-subset fallback. |
| 6. Held-out validation and paper | Wed 21 Oct to Mon 16 Nov | G6, Fri 13 Nov (results frozen) | Weighted score beats uniform weighting on unseen policies and failure types, bootstrap interval excluding zero. | Report the negative result; contributions 1 and 2 stand without it. |

Writing starts on 21 Oct alongside analysis, as in the proposal. Registration on 10 Nov needs a
title, abstract and author list; I will draft them by 6 Nov.

Stage 3 overlaps stage 2 in the proposal's chart. Under the gate rule I read that as: evaluations
of Populations B and C (which need no degradation code) may start after G1, and Population A
evaluations start after G2.

### Stage 1 day by day

| Day | You run | I prepare |
|---|---|---|
| Thu 8 Oct | `hpc_probe.sh`, the node probe job, `setup_env.sh`, `fetch_assets.sh` (smoke set), smoke job. Start the full sensor download in `screen`. | Fixes from your logs. |
| Fri 9 Oct | Full navtest run, TransFuser seed 0. | NAVSIM v2 (navhard) job and its fetch step. |
| Sat 10 to Sun 11 Oct | Remaining seeds and baselines on navtest; navhard runs. | Closed-loop evaluator trial (BridgeSim or HUGSIM, per your decision); Vista trial job. |
| Mon 12 to Tue 13 Oct | Closed-loop and world-model trial jobs. | DiffusionDrive reproduction against the authors' per-scenario file. |
| Wed 14 Oct | | Stage 1 report for G1. |

## 2. Compute

What is known:

- NAVSIM v1 and v2 scoring is **CPU work**. The devkit runs the agent's forward pass on CPU, so
  these two evaluator rows use the 384 CPU nodes, not the 22 GPU nodes.
- navtest has 12,146 scenarios (read from the devkit config).
- The proposal plans about 64 policy variants, so the NAVSIM v1 row is about 64 × 12,146 ≈ 777,000
  scenario evaluations, plus one metric-cache build.
- GPUs are V100 16 GB, two per node, 44 on the cluster (public documentation). The proposal asked for
  "4 to 8 modern GPUs for 3 to 4 weeks"; what you can actually hold at once is unknown until the
  probe shows the per-user limits.

What is not known, and where the number will come from:

| Quantity | Source |
|---|---|
| NAVSIM v1 scenarios per second on one 40-core node | `scenarios_per_second` in the smoke job's summary |
| Metric-cache build time | `timings_s.cache` in the same summary |
| Seconds per closed-loop scenario, and GPU memory | Stage 1 closed-loop trial |
| Seconds per world-model sample on a V100, and whether it fits in 16 GB | Stage 1 world-model trial |
| How many jobs and GPUs you may run at once | `sacctmgr` output in the probe |

Estimate per stage. CPU figures are given as a formula because the throughput is unmeasured; the
two worked values are illustrations of the formula, not predictions.

| Stage | CPU | GPU | Disk added |
|---|---|---|---|
| 0 | One smoke job, 64 scenarios. | One 5-minute probe job. | About 15 GB: a 6 GB venv (measured in a sandbox), one sensor shard pair (about 7 GB by the documented total), maps and logs. |
| 1 | 9 navtest runs (3 baselines × 3 seeds) = 109,314 scenario evaluations, plus the same policies on navhard. | Trial runs only: a handful of scenarios in the closed-loop evaluator and the world model. | 217 GB test sensors, 31 GB navhard, 61 GB if HUGSIM scenes are used. |
| 2 | Population A wrappers on subsets. | Monotonicity check on the closed-loop evaluator: bases × kept degradations × 4 levels × N scenarios, with N chosen after the stage 1 timing. Optional: one LTF training run for Population C, unknown GPU-days on V100s; the default is to use the public seeds instead. | 300 to 445 GB only if we train. |
| 3 | The bulk: about 777,000 v1 scenario evaluations. At r scenarios per second per node this is 777,000 / r seconds of node time: 216 node-hours if r = 1, 22 node-hours if r = 10. NAVSIM v2 on top. | Closed-loop evaluator and world model on scenario subsets, sized from stage 1 timings so that they finish by 27 Oct. | Per-scenario score files, small. |
| 4 | Minutes: 1,000 bootstrap resamples over the score matrix. | None. | None. |
| 5 | Minutes: logistic regression or a small MLP. | None. | None. |
| 6 | Re-scoring of held-out policies; same unit costs as stage 3. | Same evaluators on held-out policies. | None. |

Total disk for the study without training: roughly 310 GB plus environments, against a documented
2 TB soft quota on scratch (to be confirmed with `myquota`).

## 3. Open risks

Ordered by how much of the paper each one can take down.

| # | Risk | Status | Fallback and decision date |
|---|---|---|---|
| 1 | **The closed-loop evaluator.** HUGSIM cannot be installed on the cluster under the no-compilation rule, and it is the proposal's reference for the stage 2 monotonicity check. | Open | BridgeSim on the cluster, HUGSIM off the cluster, or prebuilt HUGSIM wheels (`docs/FEASIBILITY.md` section 4). **Your decision by Sun 11 Oct.** |
| 2 | **The world-model row.** V100s have 16 GB, and no open model is both multi-camera and action-conditioned, so a closed loop with a three-camera policy is not available. | Open | Vista as a single-step critic on a subset; ReSim if driver and memory allow; otherwise report the row as not obtainable. H2 must be re-read against the single-step design. **Decision at G1.** |
| 3 | **Degradations versus the baseline architecture.** TransFuser and LTF are single-frame and use three cameras, so "history removal" does not apply and camera dropout tops out at three cameras. | Open | Redefine dropout over the three used cameras; drop history removal or add a temporal policy from Population B. **Decision at G1, before any wrapper is written.** |
| 4 | **Different evaluators host different policies.** HUGSIM hosts only LTF, with nuScenes cameras and a fixed driving command; world models see one camera. The cross-evaluator matrix is only as wide as the policies every evaluator can host. | Open | Define the common policy set explicitly at G1 and state it as a limitation. |
| 5 | **Overlap with BridgeSim** (CoRL 2026) and its September follow-ups, which the proposal does not cite. | Open | Cite and differentiate: they measure and close the open-loop to closed-loop gap for policies; we test evaluators against a known ordering. |
| 6 | **Time.** 39 days, shared queues of unknown length, about 220 GB to download through a login node. | Open | Sensor download starts today. Queue depth comes from the probe. GPU rows shrink to subsets before anything else slips. |
| 7 | **Reproduction target.** The published TransFuser PDMS has a seed standard deviation of 0.56; one seed can miss 84.0 by more than half a point without anything being wrong. | Known | Run all three seeds and compare the mean. The per-seed numbers are reported either way. |
| 8 | **Environment on CentOS 7.** The pinned wheel set is verified to exist for glibc 2.17 and to work in a sandbox, not on the cluster. Ray may not start on compute nodes. | Open until G0 | Offline wheelhouse option; process-pool worker instead of Ray (one config line). |
| 9 | **Population C by training** needs navtrain and an unknown number of V100 days. | Open | Use the three public seeds per baseline, which the proposal allows. |
| 10 | **Scratch retention.** Files older than 60 days are deleted (public documentation). | Low | The project fits inside the window; results that matter are committed to git. |

## 4. Decisions I need from you

1. At G0: the filled `configs/hpc/cluster.env` values, from the probe.
2. By Sun 11 Oct: which closed-loop evaluator to pursue (risk 1).
3. At G1: the world-model design (risk 2), the degradation list (risk 3) and the common policy set (risk 4).
