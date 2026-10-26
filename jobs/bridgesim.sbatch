#!/bin/bash
#SBATCH --job-name=wete-bridgesim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# BridgeSim closed-loop runs. One job per step:
#
#   convert (CPU, once per split):
#     bash scripts/submit.sh cpu jobs/bridgesim.sbatch ACTION=convert CONFIG=configs/experiments/stage1_bridgesim_navhard.yaml
#   evaluate (GPU, one job per shard):
#     bash scripts/submit.sh gpu jobs/bridgesim.sbatch ACTION=evaluate SHARD=0 NUM_SHARDS=4 CONFIG=...
#   report (CPU, after every shard finished):
#     bash scripts/submit.sh cpu jobs/bridgesim.sbatch ACTION=report CONFIG=...
#
# OVERRIDES selects the policy, e.g. OVERRIDES="policy.family=ltf policy.seed=1".

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
: "${ACTION:?Pass ACTION=convert|evaluate|report}"
: "${CONFIG:?Pass CONFIG=configs/experiments/<file>.yaml}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_use_bridgesim

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Several evaluator processes share the node; keep each one's math libraries small.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
JOB_TMP="${SLURM_TMPDIR:-/tmp}/wete-${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
cleanup() { rm -rf "$JOB_TMP"; }
trap cleanup EXIT

case "$CONFIG" in
    /*) CONFIG_PATH="$CONFIG" ;;
    *) CONFIG_PATH="$WETE_REPO/$CONFIG" ;;
esac

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none} cpus=${SLURM_CPUS_PER_TASK:-unset} gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "[job] repo=$WETE_REPO commit=${WETE_GIT_COMMIT:-unknown} (recorded at submission)"
echo "[job] action=$ACTION shard=${SHARD:-0}/${NUM_SHARDS:-1} config=$CONFIG_PATH overrides=${OVERRIDES:-}"
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$WETE_REPO"
# OVERRIDES is split on spaces on purpose.
# shellcheck disable=SC2086
python -m wete.cli.bridgesim_run --config "$CONFIG_PATH" --action "$ACTION" \
    --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" ${OVERRIDES:-}

echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
