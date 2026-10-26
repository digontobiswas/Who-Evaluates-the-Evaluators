#!/bin/bash
#SBATCH --job-name=wete-navsim-v1-pdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Stage 1 job: reproduce a published NAVSIM v1 PDMS with the unmodified devkit.
# CPU only: in NAVSIM v1.1 the agent's forward pass runs on the CPU during scoring.
#
# Submit through the wrapper, which supplies partition, cores and time from cluster.env:
#
#   smoke test (64 scenarios, needs one camera shard and one LiDAR shard):
#     bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/stage1_pdms_transfuser_smoke.yaml
#
#   full navtest (about 12k scenarios, needs all 32 sensor shards):
#     bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/stage1_pdms_transfuser_navtest.yaml
#
# Optional variables:
#   STAGES     comma-separated subset of select,cache,score,report (default: all four)
#   OVERRIDES  space-separated config overrides, e.g. OVERRIDES="policy.seed=1"

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
: "${CONFIG:?Pass CONFIG=configs/experiments/<file>.yaml}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
wete_use_navsim v1

# Compute nodes have no internet. Every weight file is already in the caches filled by
# scripts/fetch_assets.sh; these switches turn an attempted download into an immediate error
# instead of a long timeout.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Parallelism comes from the devkit's worker pool (one scenario per worker). Keep each worker
# single-threaded so that workers x math-library threads does not exceed the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Node-local temporary directory with a short path (Ray keeps UNIX sockets under it).
JOB_TMP="${SLURM_TMPDIR:-/tmp}/wete-${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export RAY_TMPDIR="$JOB_TMP"
cleanup() { rm -rf "$JOB_TMP"; }
trap cleanup EXIT

case "$CONFIG" in
    /*) CONFIG_PATH="$CONFIG" ;;
    *) CONFIG_PATH="$WETE_REPO/$CONFIG" ;;
esac

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none} cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] repo=$WETE_REPO commit=${WETE_GIT_COMMIT:-unknown} (recorded at submission)"
echo "[job] python=$(command -v python) config=$CONFIG_PATH"
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$WETE_REPO"
# OVERRIDES is split on spaces on purpose.
# shellcheck disable=SC2086
python -m wete.cli.navsim_v1 \
    --config "$CONFIG_PATH" \
    --stages "${STAGES:-select,cache,score,report}" \
    ${OVERRIDES:-}

echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
