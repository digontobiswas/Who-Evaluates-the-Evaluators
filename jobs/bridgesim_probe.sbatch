#!/bin/bash
#SBATCH --job-name=wete-bridgesim-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Stage 1 feasibility probe for BridgeSim (closed-loop evaluator) on one GPU. About 30 minutes.
# Needs scripts/setup_bridgesim.sh to have run on the login node.
#
#     bash scripts/submit.sh gpu jobs/bridgesim_probe.sbatch
#
# Optional: CONFIG (default configs/experiments/stage1_bridgesim_probe.yaml), OVERRIDES.

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
CONFIG="${CONFIG:-configs/experiments/stage1_bridgesim_probe.yaml}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_use_bridgesim

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/wete-${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
cleanup() { rm -rf "$JOB_TMP"; }
trap cleanup EXIT

case "$CONFIG" in
    /*) CONFIG_PATH="$CONFIG" ;;
    *) CONFIG_PATH="$WETE_REPO/$CONFIG" ;;
esac

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none} gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "[job] repo=$WETE_REPO commit=${WETE_GIT_COMMIT:-unknown} (recorded at submission)"
echo "[job] DISPLAY=${DISPLAY:-unset} python=$(command -v python) config=$CONFIG_PATH"
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$WETE_REPO"
# OVERRIDES is split on spaces on purpose.
# shellcheck disable=SC2086
python -m wete.cli.bridgesim_probe --config "$CONFIG_PATH" ${OVERRIDES:-}

echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
