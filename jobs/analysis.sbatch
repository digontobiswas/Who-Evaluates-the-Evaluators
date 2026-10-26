#!/bin/bash
#SBATCH --job-name=wete-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Collect every finished run under $WETE_EXP_ROOT/<RUNS> into one score table and run the
# analysis (ordering recovery, monotonicity, agreement, sample efficiency, reliability).
# CPU only, minutes to an hour depending on the number of runs and bootstrap resamples.
#
#     bash scripts/submit.sh cpu jobs/analysis.sbatch NAME=popA_navsim_v1 NO_REFERENCE=1
#
# Variables:
#   NAME          output name (required); results go to $WETE_EXP_ROOT/analysis/$NAME and the
#                 small files are copied to results/analysis/$NAME for committing
#   RUNS          run root under $WETE_EXP_ROOT (default: runs)
#   META          evaluator meta file (default: configs/meta.navsim_v1.json)
#   NO_REFERENCE  1 = provisional summary without a reference evaluator; monotonicity is then
#                 not tested and H1-H4 are skipped. Required while only NAVSIM rows exist.
#   B             bootstrap resamples (default 1000)

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
: "${NAME:?Pass NAME=<analysis name>}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
export PYTHONPATH="$WETE_REPO/src"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

RUN_ROOT="$WETE_EXP_ROOT/${RUNS:-runs}"
OUT="$WETE_EXP_ROOT/analysis/$NAME"
EXPORT="$WETE_REPO/results/analysis/$NAME"
META_PATH="${META:-configs/meta.navsim_v1.json}"
case "$META_PATH" in /*) ;; *) META_PATH="$WETE_REPO/$META_PATH" ;; esac

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] repo commit=${WETE_GIT_COMMIT:-unknown} (recorded at submission)"
echo "[job] runs=$RUN_ROOT meta=$META_PATH out=$OUT"
[ -d "$RUN_ROOT" ] || { echo "[job] ERROR: $RUN_ROOT does not exist" >&2; exit 1; }
mkdir -p "$OUT" "$EXPORT"

cd "$WETE_REPO"
python -m wete.collect --root "$RUN_ROOT" --out "$OUT"

EXTRA=()
[ "${NO_REFERENCE:-0}" = 1 ] && EXTRA+=(--no-reference)
FEATURES=()
[ -f "$OUT/features.csv" ] && FEATURES=(--features "$OUT/features.csv")
BEHAVIOR=()
[ -f "$OUT/behavior.csv" ] && BEHAVIOR=(--behavior "$OUT/behavior.csv")

python -m wete.analysis --scores "$OUT/scores.csv" --meta "$META_PATH" --out "$OUT" \
    --B "${B:-1000}" ${FEATURES[@]+"${FEATURES[@]}"} ${BEHAVIOR[@]+"${BEHAVIOR[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}

# Small files only; scores.csv stays on scratch.
for f in collect_report.json report.json monotonicity.csv discriminative_matrix.csv \
         discriminative_matrix.png sample_efficiency.png tab_matrix.tex tab_reliability.tex; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$EXPORT/"
done
echo "[job] exported to $EXPORT"
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
