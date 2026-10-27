#!/bin/bash
# Submit the Population A grid for one base policy on NAVSIM v1: the clean run through the
# wrapper plus levels 1-3 of each degradation, one CPU job each, through scripts/submit.sh.
#
#     bash scripts/submit_population_a.sh [--dry-run] <experiment.yaml> [degradation ...]
#
# Examples:
#     bash scripts/submit_population_a.sh --dry-run configs/experiments/popA_transfuser_navtest.yaml
#     bash scripts/submit_population_a.sh configs/experiments/popA_ltf_navtest.yaml camera_dropout
#
# Default degradations: those that can change a single-frame NAVSIM agent. history_removal is
# left out (TransFuser and LTF read only the current frame, so it is a no-op for them) and
# latency is closed-loop only. Pass names explicitly to override.
#
# SKIP_CLEAN=1 leaves out the clean run (use it when adding perturbations to an existing grid),
# e.g. the specificity arm:
#     SKIP_CLEAN=1 bash scripts/submit_population_a.sh configs/experiments/popA_ltf_navtest.yaml \
#         tail_lateral tail_stretch
#
# Jobs skip the metric-cache stage, so the cache must exist first (the stage 1 run of the same
# split builds it). Each job checks this and stops with a clear message if it does not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then
    DRY=(--dry-run)
    shift
fi

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "Usage: submit_population_a.sh [--dry-run] <experiment.yaml> [degradation ...]" >&2; exit 1; }
shift

if [ $# -gt 0 ]; then
    DEGRADATIONS=("$@")
else
    DEGRADATIONS=(camera_dropout image_noise stale_obs lateral_noise late_braking)
fi
LEVELS=(1 2 3)

submit_one() {
    # submit_one "<overrides>"
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch \
        "CONFIG=$CONFIG" "STAGES=select,score,report" "OVERRIDES=$1"
}

count=0
if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    submit_one "degradation.name=none degradation.level=0"
    count=$((count + 1))
fi
for deg in "${DEGRADATIONS[@]}"; do
    for level in "${LEVELS[@]}"; do
        submit_one "degradation.name=$deg degradation.level=$level"
        count=$((count + 1))
    done
done
echo "[popA] $count jobs for $CONFIG (degradations: ${DEGRADATIONS[*]})"
