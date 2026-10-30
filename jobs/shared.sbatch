#!/bin/bash
#SBATCH --job-name=wete-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Compare evaluators on the scenarios they share (NavHard scenarios that are also in navtest),
# using the score table of an existing analysis (wete.collect output). CPU only, minutes.
#
#     bash scripts/submit.sh cpu jobs/shared.sbatch NAME=main_ds
#
# Variables:
#   NAME  analysis whose scores.csv is used ($WETE_EXP_ROOT/analysis/$NAME/scores.csv)
#   META  evaluator meta file (default: configs/meta.bridgesim.json)
#   B     bootstrap resamples (default 1000)

set -euo pipefail

: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
: "${NAME:?Pass NAME=<analysis name>}"

# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
export PYTHONPATH="$WETE_REPO/src"

SCORES="$WETE_EXP_ROOT/analysis/$NAME/scores.csv"
OUT="$WETE_EXP_ROOT/analysis/$NAME/shared"
EXPORT="$WETE_REPO/results/analysis/$NAME/shared"
META_PATH="${META:-configs/meta.bridgesim.json}"
case "$META_PATH" in /*) ;; *) META_PATH="$WETE_REPO/$META_PATH" ;; esac

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} commit=${WETE_GIT_COMMIT:-unknown}"
[ -f "$SCORES" ] || { echo "[job] ERROR: $SCORES not found; run jobs/analysis.sbatch NAME=$NAME first" >&2; exit 1; }
mkdir -p "$OUT" "$EXPORT"

cd "$WETE_REPO"
python -m wete.shared --scores "$SCORES" --meta "$META_PATH" --out "$OUT" --B "${B:-1000}"

# Noise ceiling: BridgeSim clean runs (Population A level 0) against their wrapped reruns.
RERUN="$WETE_EXP_ROOT/runs_variants/bridgesim/wrapped_rerun"
if [ -d "$RERUN" ]; then
    ORIG="$OUT/retest_original"
    rm -rf "$ORIG" && mkdir -p "$ORIG"
    for d in "$RERUN"/*/; do
        pol="$(basename "$d")"
        src="$WETE_EXP_ROOT/runs/bridgesim/A/$pol/none__L0"
        [ -f "$src/per_scenario.csv" ] && ln -s "$src" "$ORIG/$pol"
    done
    python -m wete.shared --scores "$SCORES" --meta "$META_PATH" --out "$OUT" --B "${B:-1000}" \
        --retest "$ORIG" "$RERUN"
fi

for f in shared_report.json shared_agreement.csv family_contrasts.csv contrast_correlation.csv popA_shared.csv retest.csv; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$EXPORT/"
done
echo "[job] exported to $EXPORT"
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
