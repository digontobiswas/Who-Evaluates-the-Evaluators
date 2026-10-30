#!/bin/bash
# Submit Population B: the public navsim_baselines checkpoints (TransFuser, LTF, Ego Status MLP x
# seeds 0-2), undegraded, on NAVSIM v1 navtest (CPU, plans logged for open-loop L2) and in BridgeSim
# on NavHard (GPU).
#
#     bash scripts/submit_population_b.sh [--dry-run]
#
# TransFuser and LTF seed 0 already exist as the Population A clean runs. They are linked into
# runs/<evaluator>/B/ (symbolic links, nothing copied or recomputed), so 7 policies are submitted
# per evaluator: 7 CPU jobs and 7 GPU jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then
    DRY=(--dry-run)
    shift
fi

RUNS="$WETE_EXP_ROOT/runs"
FAMILIES=(transfuser ltf ego_status_mlp)
SEEDS=(0 1 2)
REUSED=(transfuser_seed0 ltf_seed0)

link_clean() {
    # link_clean <evaluator> <policy id>
    local src="$RUNS/$1/A/$2/none__L0" dst="$RUNS/$1/B/$2"
    if [ ! -d "$src" ]; then
        echo "[popB] ERROR: $src missing; the Population A clean run is needed" >&2
        exit 1
    fi
    if [ -e "$dst" ]; then
        echo "[popB] exists: $dst"
    elif [ ${#DRY[@]} -gt 0 ]; then
        echo "[popB] would link $dst -> $src"
    else
        mkdir -p "$RUNS/$1/B"
        ln -s "$src" "$dst"
        echo "[popB] linked $dst -> $src"
    fi
}

for id in "${REUSED[@]}"; do
    link_clean navsim_v1 "$id"
    link_clean bridgesim "$id"
done

count=0
for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        id="${family}_seed${seed}"
        case " ${REUSED[*]} " in *" $id "*) continue ;; esac
        bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch \
            "CONFIG=configs/experiments/popB_${family}_navtest.yaml" "STAGES=select,score,report" \
            "OVERRIDES=policy.seed=$seed"
        bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate \
            CONFIG=configs/experiments/popB_bridgesim_navhard.yaml \
            "OVERRIDES=policy.family=$family policy.seed=$seed" -- --time=06:00:00
        count=$((count + 2))
    done
done
echo "[popB] $count jobs (NAVSIM v1 CPU + BridgeSim GPU)"
