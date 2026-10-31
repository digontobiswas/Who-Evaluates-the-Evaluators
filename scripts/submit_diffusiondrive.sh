#!/bin/bash
# DiffusionDrive runs (after scripts/setup_diffusiondrive.sh): NAVSIM v1 and BridgeSim, each with
# the clean run and lateral_noise, latency, tail_lateral, tail_stretch at L1-3 (13 runs per
# evaluator, 26 jobs). The clean runs are linked into Population B.
#
#     bash scripts/submit_diffusiondrive.sh [--dry-run]
#
# DiffusionDrive is a held-out base: run the final analysis with SELECTION_BASES=transfuser,ltf so
# that degradation families stay validated on the pre-registered bases only.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
for f in diffusiondrive_navsim_88p1_PDMS kmeans_navsim_traj_20.npy; do
    [ -s "$WETE_CKPT_ROOT/diffusiondrive/$f" ] || { echo "[dd] ERROR: $f missing; run scripts/setup_diffusiondrive.sh" >&2; exit 1; }
done
RUNS="$WETE_EXP_ROOT/runs"
ID=diffusiondrive_seed0

count=0
specs=("none 0")
for deg in lateral_noise latency tail_lateral tail_stretch; do
    for level in 1 2 3; do specs+=("$deg $level"); done
done
for spec in "${specs[@]}"; do
    set -- $spec
    # NAVSIM v1 on CPU: the diffusion decoder is slower than TransFuser, so allow 12 h.
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch \
        CONFIG=configs/experiments/popA_diffusiondrive_navtest.yaml "STAGES=select,score,report" \
        "OVERRIDES=degradation.name=$1 degradation.level=$2" -- --time=12:00:00
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate \
        CONFIG=configs/experiments/popA_bridgesim_diffusiondrive.yaml \
        "OVERRIDES=degradation.name=$1 degradation.level=$2" -- --time=06:00:00
    count=$((count + 2))
done

for ev in navsim_v1 bridgesim; do
    src="$RUNS/$ev/A/$ID/none__L0" dst="$RUNS/$ev/B/$ID"
    if [ -e "$dst" ]; then echo "[dd] exists: $dst"
    elif [ ${#DRY[@]} -gt 0 ]; then echo "[dd] would link $dst -> $src"
    else mkdir -p "$RUNS/$ev/B" && ln -s "$src" "$dst" && echo "[dd] linked $dst -> $src"; fi
done
echo "[dd] $count jobs"
