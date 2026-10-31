#!/bin/bash
# Replication in NAVSIM v2 (senior's point 4, applied to v2): seeds 1-2 of the Population A bases for
# the degradations that matter (lateral_noise and latency, L1-3), as already done for NAVSIM v1 and
# BridgeSim (scripts/submit_review_fixes.sh). Submit after scripts/submit_navsim_v2.sh has finished:
# the clean runs of seeds 1-2 are its Population B runs, linked into runs/<evaluator>/A/.
#
#     bash scripts/submit_navsim_v2_seeds.sh [--dry-run] <navtest|navhard>
#
# navtest: TransFuser and LTF, 24 CPU jobs. navhard: LTF only (no LiDAR in the synthetic scenes), 12.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
SPLIT="${1:-}"
case "$SPLIT" in
    navtest) EV=navsim_v2; FAMS=(transfuser ltf) ;;
    navhard) EV=navsim_v2_ph; FAMS=(ltf) ;;
    *) echo "Usage: submit_navsim_v2_seeds.sh [--dry-run] <navtest|navhard>" >&2; exit 1 ;;
esac
RUNS="$WETE_EXP_ROOT/runs"

count=0
for fam in "${FAMS[@]}"; do
    for seed in 1 2; do
        id="${fam}_seed${seed}"
        src="$RUNS/$EV/B/$id" dst="$RUNS/$EV/A/$id/none__L0"
        if [ ! -f "$src/summary.json" ]; then
            echo "[v2-seeds] ERROR: $src has no summary.json; wait for scripts/submit_navsim_v2.sh $SPLIT to finish" >&2
            exit 1
        fi
        if [ -e "$dst" ]; then echo "[v2-seeds] exists: $dst"
        elif [ ${#DRY[@]} -gt 0 ]; then echo "[v2-seeds] would link $dst -> $src"
        else mkdir -p "$RUNS/$EV/A/$id" && ln -s "$src" "$dst" && echo "[v2-seeds] linked $dst -> $src"; fi
        for deg in lateral_noise latency; do
            for level in 1 2 3; do
                bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
                    "CONFIG=configs/experiments/popA_${fam}_${SPLIT}_v2.yaml" "STAGES=select,score,report" \
                    "OVERRIDES=policy.seed=$seed degradation.name=$deg degradation.level=$level"
                count=$((count + 1))
            done
        done
    done
done
echo "[v2-seeds] $count jobs for $SPLIT"
