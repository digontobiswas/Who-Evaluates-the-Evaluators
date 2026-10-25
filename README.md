# Who Evaluates the Evaluators?

Code for a study of whether the benchmarks used to rank end-to-end driving policies rank them
correctly. We build policy populations whose quality order is known by construction, run them
through several evaluator families, and measure which evaluators recover the order. Target: CVPR 2027
(paper deadline 16 Nov 2026 AoE).

**Status (8 Oct 2026): stage 0 done, stage 1 started.** The repository holds the scaffold, the
cluster environment scripts and the first job (reproduce a published NAVSIM v1 PDMS). No experiment
has been run yet and there are no results.

- [PLAN.md](PLAN.md): the six stages mapped to dates, with go/no-go checks, compute and risks.
- [docs/FEASIBILITY.md](docs/FEASIBILITY.md): what can run on PARAM Shakti and what needs a fallback.

## How work moves

Code is written on a PC, pushed to GitHub, pulled on PARAM Shakti and run with `sbatch`. Nothing is
executed on the cluster except through the scripts here, and every number that is ever quoted comes
from a result file a job wrote.

Cluster rules the code is built around:

- `python3` and `venv` only, no conda. The interpreter is a pinned relocatable CPython 3.9, because
  the cluster's own Python 3 cannot reach PyPI (no working `ssl` module).
- Wheels only. Nothing is compiled on the cluster.
- Compute nodes have no internet. Everything is fetched on the login node first.
- Jobs are submitted from scratch. `scripts/submit.sh` takes care of that.
- No partition, quota or path is written in code. They live in `configs/hpc/cluster.env`, which you
  fill from commands you ran.

## Layout

```
configs/
  default.yaml                 paths (all from environment variables) and shared defaults
  third_party.env              pinned commits and the pinned Python build
  hpc/cluster.env.example      template for the cluster settings (copy to cluster.env)
  policies/                    one file per policy family: checkpoint, sensors, published score
  evaluators/                  one file per evaluator
  experiments/                 one file per job; each `extends` the three above
  navsim_hydra/v1/agent/       Hydra configs of the wrapped agents, for the v1.1 devkit
  meta.*.json                  evaluator orientation and family for the analysis
docs/                          feasibility report
adapters/                      HUGSIM relay and world-model interface (not runnable on the cluster yet)
jobs/                          sbatch scripts, free of cluster-specific settings
requirements/                  pinned wheel set for the NAVSIM environment
results/                       small result files copied here by jobs, for committing
scripts/
  env.sh                       sourced by everything; the only place the directory layout is decided
  hpc_probe.sh                 read-only fact gathering on the login node
  setup_env.sh                 Python, venv, wheels, pinned devkit checkouts
  fetch_assets.sh              NAVSIM data, one checkpoint, backbone weights
  submit.sh                    sbatch wrapper that reads cluster.env
src/wete/
  config.py                    YAML loading with `extends`, interpolation and overrides
  policies/base.py             PolicyWrapper interface (no implementation yet)
  evaluators/base.py           Evaluator interface (no implementation yet)
  navsim/                      driver for the unmodified NAVSIM v1.1 devkit; degraded_agent.py wraps agents
  cli/navsim_v1.py             entry point of the stage 1 and Population A jobs
  degradations.py, wrappers.py Population A operators (open loop and closed loop)
  stats.py, reliability.py     ordering recovery, monotonicity, agreement, reliability weighting
  analysis.py, collect.py      run outputs -> score table -> report, tables, figures
  toysim.py                    toy world with planted blind spots, for pipeline checks only
tests/                         unit tests for the code above; no data or devkit needed
```

On the cluster everything large lives under one workspace on scratch, `$WETE_ROOT`:

```
$WETE_ROOT/
  tools/         relocatable CPython
  venvs/navsim   the virtual environment
  third_party/   navsim_v1, navsim_v2, nuplan-devkit at pinned commits
  data/          OpenScene logs and sensors, nuPlan maps
  checkpoints/   pretrained weights
  cache/         pip, Hugging Face and torch caches
  exp/           metric caches and experiment outputs
  slurm_logs/    job output files
```

## First run on the cluster

```bash
# 1. Facts first. Read-only; paste both outputs back.
bash scripts/hpc_probe.sh 2>&1 | tee hpc_probe_login.txt
cd $SCRATCH && sbatch --partition=<GPU_PARTITION> --gres=<GRES> --time=00:05:00 \
    --output=wete-probe-%j.out <repo>/jobs/probe_node.sbatch

# 2. Cluster settings, from the probe output.
cp configs/hpc/cluster.env.example configs/hpc/cluster.env
nano configs/hpc/cluster.env

# 3. Environment (login node, about 6 GB).
bash scripts/setup_env.sh

# 4. Smoke-test assets: maps, logs, one sensor shard pair, TransFuser seed 0, backbone weights.
bash scripts/fetch_assets.sh --accept-licenses

# 5. Smoke job: 64 navtest scenarios through the same code path as the full run.
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch \
    CONFIG=configs/experiments/stage1_pdms_transfuser_smoke.yaml

# 6. Full sensor set (217 GB; run inside screen), then the real check.
screen -S fetch
bash scripts/fetch_assets.sh --accept-licenses --what sensors --sensor-shards all
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch \
    CONFIG=configs/experiments/stage1_pdms_transfuser_navtest.yaml
```

`submit.sh` prints the exact `sbatch` command it runs; `--dry-run` prints it without submitting.
Job output goes to `$WETE_ROOT/slurm_logs/<job-name>-<job-id>.out`. The job ends by printing a
`SUMMARY` block, which is the part to paste back.

## Population A, analysis and the merged library

The degradation operators, statistics, reliability method and analysis came from the senior
team's `wete` package and now live in `src/wete/` (review: [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md)).
Population A on NAVSIM v1 runs through the same job as stage 1; a `degradation` block in the
config puts the checkpoint inside `wete.navsim.degraded_agent.DegradedAgent`.

```bash
# 0. Pipeline check in the cluster environment: unit tests + toy world with planted ground truth.
#    Toy numbers validate the code only; they are never results.
bash scripts/submit.sh cpu jobs/validate_pipeline.sbatch

# 1. Wrapper smoke test on the stage 1 smoke subset (64 scenarios, same metric cache):
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/popA_transfuser_smoke.yaml
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/popA_transfuser_smoke.yaml \
    OVERRIDES="degradation.name=stale_obs degradation.level=3"

# 2. Full grid for one base policy (clean + 5 degradations x 3 levels = 16 jobs). Needs the full
#    navtest metric cache from stage 1. Check with --dry-run first.
bash scripts/submit_population_a.sh --dry-run configs/experiments/popA_transfuser_navtest.yaml

# 3. Collect and analyse everything under $WETE_EXP_ROOT/runs. Without a closed-loop reference
#    the summary is provisional: monotonicity is not tested and H1-H4 are skipped.
bash scripts/submit.sh cpu jobs/analysis.sbatch NAME=popA_navsim_v1 NO_REFERENCE=1
```

Runs land in `$WETE_EXP_ROOT/runs/navsim_v1/A/<policy>/<degradation>__L<level>/`; each exports
`summary.json` (with open-loop L2 from the plan log) and `per_scenario.csv` to `results/popA/`.

`adapters/hugsim/` and `adapters/world_model/` are kept as written by the senior team. Neither can
run on PARAM Shakti as things stand (docs/FEASIBILITY.md, sections 4 and 5).

## Configuration

An experiment is one YAML file. It may list other files under `extends`; those merge first, in
order, and the file itself merges last. `${paths.exp}` refers to another key and `${oc.env:NAME}`
to an environment variable. Overrides are dotted `key=value` pairs:

```bash
# another seed of the same baseline
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch \
    CONFIG=configs/experiments/stage1_pdms_transfuser_navtest.yaml OVERRIDES="policy.seed=1"

# only re-run the report stage
bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch \
    CONFIG=configs/experiments/stage1_pdms_transfuser_navtest.yaml STAGES=report
```

To see the devkit commands a config produces without running anything (works on a PC too, once the
environment variables in `configs/default.yaml` are set to any value):

```bash
python -m wete.cli.navsim_v1 --config configs/experiments/stage1_pdms_transfuser_smoke.yaml --dry-run
```

## Results

A job writes everything to `$WETE_ROOT/exp/<experiment>/` and copies three small files to
`results/<experiment>/` in the repository: `summary.json`, `per_scenario.csv` and
`resolved_config.yaml`. Those are what gets committed and what any reported number is read from.

The summary of a full-split run includes a comparison with the published score. It gives a verdict
only if every scenario of the split was scored and valid. A subset run never gets a verdict: the
published number is defined on the full split.

## Tests

```bash
pip install -e ".[dev]"     # any Python 3.9+ on a PC
pytest
```

## When something fails on the cluster

| Symptom | Cause and fix |
|---|---|
| `setup_env.sh` cannot download CPython | The login node cannot reach GitHub release assets. Download the file named in `configs/third_party.env` on a PC, copy it over, and run `bash scripts/setup_env.sh --python-tarball <file>`. |
| pip cannot reach PyPI even with the new interpreter | Build a wheelhouse on a PC from the lock file, copy it over, and run `bash scripts/setup_env.sh --wheelhouse <dir>`. Ask me for the exact `pip download` commands. |
| `Unpacked CPython does not run here` | glibc is older than 2.17. Send `ldd --version`. |
| Job fails while constructing the model with a Hugging Face error | The backbone weights were not cached. Run `bash scripts/fetch_assets.sh --what backbone` on the login node. |
| Metric caching hangs or dies at "Starting ray" | Ray cannot start on the node. In `configs/evaluators/navsim_v1_pdms.yaml` set the metric-cache worker to `single_machine_thread_pool` with `use_process_pool: true`. |
| `Full-split run refused: N scenarios lack sensor files` | The sensor download is incomplete. Re-run the `--sensor-shards all` fetch; it resumes. |
| `cluster.env has Windows line endings` | The file was edited on Windows. Run the `sed` command the message prints. |
| `Permission denied` running a script | Run it as `bash scripts/<name>.sh`. |
