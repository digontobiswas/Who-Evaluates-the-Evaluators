#!/bin/bash
#SBATCH --job-name=wete-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Checks the analysis code in the cluster's own environment: the unit tests, then the toy-world
# validation (planted ground truth, checks V1-V6). CPU only, about 10-20 minutes.
#
#     bash scripts/submit.sh cpu jobs/validate_pipeline.sbatch
#
# The toy numbers validate the machinery only. They are not results about NAVSIM, HUGSIM or any
# world model and must never be reported as findings.

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
export PYTHONPATH="$WETE_REPO/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT="$WETE_EXP_ROOT/toy_validation"
EXPORT="$WETE_REPO/results/toy_validation"
mkdir -p "$OUT" "$EXPORT"

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] repo commit=$(cd "$WETE_REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[job] python=$(command -v python) $(python --version 2>&1)"

cd "$WETE_REPO"
python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -n 25

python scripts/validate_on_toy.py --out "$OUT" --n "${N:-400}" --B "${B:-500}"
cp "$OUT/validation.json" "$EXPORT/validation.json"
echo "[job] ===== VALIDATION (paste this block back) ====="
cat "$OUT/validation.json"
echo "[job] ===== END VALIDATION ====="
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
