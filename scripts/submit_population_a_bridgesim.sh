#!/bin/bash
# Submit Population A in BridgeSim for the given base policies: levels 1-3 of each degradation,
# one GPU job per run (421 scenarios, 6 at a time), each writing its own report at the end.
# The clean level 0 already exists (stage 1); one short CPU job exports it into the same table.
#
#     bash scripts/submit_population_a_bridgesim.sh [--dry-run] <family> [family ...]
#     e.g. bash scripts/submit_population_a_bridgesim.sh transfuser ltf
#
# 18 GPU jobs per family. Your QoS runs 4 GPUs at a time and allows 100 queued jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=configs/experiments/popA_bridgesim_navhard.yaml
DEGRADATIONS=(camera_dropout image_noise stale_obs lateral_noise late_braking latency)

DRY=()
if [ "${1:-}" = "--dry-run" ]; then
    DRY=(--dry-run)
    shift
fi
[ $# -gt 0 ] || { echo "Usage: submit_population_a_bridgesim.sh [--dry-run] <transfuser|ltf> ..." >&2; exit 1; }

count=0
for family in "$@"; do
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/bridgesim.sbatch ACTION=report CONFIG=$CONFIG \
        "OVERRIDES=policy.family=$family degradation.name=none degradation.level=0" -- --time=00:30:00
    for deg in "${DEGRADATIONS[@]}"; do
        for level in 1 2 3; do
            bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate CONFIG=$CONFIG \
                "OVERRIDES=policy.family=$family degradation.name=$deg degradation.level=$level" -- --time=06:00:00
            count=$((count + 1))
        done
    done
done
echo "[popA-bridgesim] $count GPU jobs for: $*"
