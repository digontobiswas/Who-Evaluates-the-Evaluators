#!/bin/bash
# Wave 2: NAVSIM v2 for the populations that matter, after scripts/submit_review_fixes.sh has
# built both v2 metric caches and its two v2 test runs have passed.
#
#     bash scripts/submit_navsim_v2.sh [--dry-run] <navtest|navhard>
#
# navhard_two_stage: the synthetic scenes are distributed with cameras only (no MergedPointCloud),
# so TransFuser, which reads LiDAR, cannot run there (job 1218340, 31 Oct). Latent TransFuser and
# the Ego Status MLP read no LiDAR; only they are scored on navhard (18 jobs).
#
# Per split (navtest: 33 CPU jobs):
#   Population A, TF and LTF seed 0: clean, lateral_noise, latency, tail_lateral, tail_stretch L1-3
#   Population B: TF and LTF seeds 1-2, Ego Status MLP seeds 0-2 (seed 0 of TF and LTF is linked
#   from the Population A clean run, as for NAVSIM v1 and BridgeSim)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
SPLIT="${1:-}"
case "$SPLIT" in
    navtest) EV=navsim_v2 ;;
    navhard) EV=navsim_v2_ph ;;
    *) echo "Usage: submit_navsim_v2.sh [--dry-run] <navtest|navhard>" >&2; exit 1 ;;
esac
RUNS="$WETE_EXP_ROOT/runs"
if [ "$SPLIT" = navhard ]; then FAMS_A=(ltf); FAMS_B=(ltf ego_status_mlp); else FAMS_A=(transfuser ltf); FAMS_B=(transfuser ltf ego_status_mlp); fi

job() {
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
        "CONFIG=$1" "STAGES=select,score,report" "OVERRIDES=$2"
}

count=0
for fam in "${FAMS_A[@]}"; do
    cfg="configs/experiments/popA_${fam}_${SPLIT}_v2.yaml"
    if [ ! -d "$RUNS/$EV/A/${fam}_seed0/none__L0" ] || [ ! -f "$RUNS/$EV/A/${fam}_seed0/none__L0/summary.json" ]; then
        job "$cfg" "degradation.name=none degradation.level=0"
        count=$((count + 1))
    fi
    for deg in lateral_noise latency tail_lateral tail_stretch; do
        for level in 1 2 3; do
            job "$cfg" "degradation.name=$deg degradation.level=$level"
            count=$((count + 1))
        done
    done
done
for fam in "${FAMS_B[@]}"; do
    for seed in 0 1 2; do
        if [ "$fam" != ego_status_mlp ] && [ "$seed" = 0 ]; then
            src="$RUNS/$EV/A/${fam}_seed0/none__L0" dst="$RUNS/$EV/B/${fam}_seed0"
            if [ -e "$dst" ]; then echo "[v2] exists: $dst"
            elif [ ${#DRY[@]} -gt 0 ]; then echo "[v2] would link $dst -> $src"
            else mkdir -p "$RUNS/$EV/B" && ln -s "$src" "$dst" && echo "[v2] linked $dst -> $src"; fi
            continue
        fi
        job "configs/experiments/popB_${fam}_${SPLIT}_v2.yaml" "policy.seed=$seed"
        count=$((count + 1))
    done
done
echo "[v2] $count jobs for $SPLIT"
