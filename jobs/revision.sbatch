#!/bin/bash
#SBATCH --job-name=wete-revision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Reviewer analyses (python -m wete.cli.revision) on the scores.csv of an analysis run.
#
#     bash scripts/submit.sh cpu jobs/revision.sbatch NAME=revision
#
#   NAME   the analysis whose $WETE_EXP_ROOT/analysis/<NAME>/scores.csv is read (run analysis.sbatch
#          with the same NAME first; scripts/submit_revision_analysis.sh chains the two)
# Writes $WETE_EXP_ROOT/analysis/<NAME>/revision and copies it to results/analysis/<NAME>/revision.

set -euo pipefail
: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
: "${NAME:?Pass NAME=<analysis name>}"
# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
export PYTHONPATH="$WETE_REPO/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SCORES="$WETE_EXP_ROOT/analysis/$NAME/scores.csv"
OUT="$WETE_EXP_ROOT/analysis/$NAME/revision"
EXPORT="$WETE_REPO/results/analysis/$NAME/revision"
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} commit=${WETE_GIT_COMMIT:-unknown}"
[ -f "$SCORES" ] || { echo "[job] ERROR: $SCORES missing; run analysis.sbatch NAME=$NAME first" >&2; exit 1; }

cd "$WETE_REPO"
python -m wete.cli.revision --scores "$SCORES" --results "$WETE_REPO/results" --out "$OUT" \
    --B "${B:-1000}" --R "${R:-1000}"
mkdir -p "$EXPORT"
cp "$OUT"/*.csv "$OUT"/summary.json "$EXPORT/"
echo "[job] exported to $EXPORT"
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
