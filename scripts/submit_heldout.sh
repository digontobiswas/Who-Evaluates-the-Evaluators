#!/bin/bash
# Held-out closed-loop scenes (DECISIONS.md item 19). Everything is queued in one go:
#
#     bash scripts/submit_heldout.sh [--dry-run]
#
#   1 CPU   convert: BridgeSim's converter on configs/bridgesim/navtest_heldout_filter.yaml; writes
#           the held-out list (first 420 converted candidates in the fixed hash order)
#  30 GPU   Population A after the conversion: TransFuser, LTF and DiffusionDrive seed 0, clean +
#           latency, lateral_noise and tail_lateral at levels 1-3
#   7 GPU   Population B: TransFuser and LTF seeds 1-2, Ego Status MLP seeds 0-2 (seed-0 clean runs
#           of TransFuser, LTF and DiffusionDrive are linked, not rerun)
#   4 CPU   after every GPU job has ended: analyses heldout (v1 + L2) and heldout_v2 (+ NAVSIM v2
#           one-stage), the shared-scene comparison on the held-out scenes, and the tables
#
# NAVSIM v1/v2 and L2 need no new runs: they already scored every navtest scene.
# Afterwards push results/ (git add results && git commit -m "held-out results" && git push).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi

for f in diffusiondrive_navsim_88p1_PDMS kmeans_navsim_traj_20.npy; do
    [ -s "$WETE_CKPT_ROOT/diffusiondrive/$f" ] || { echo "[heldout] ERROR: $f missing; run scripts/setup_diffusiondrive.sh" >&2; exit 1; }
done

submit() {
    # submit <cpu|gpu> <job> <vars...>: prints the job id
    local out
    out="$(bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} "$@")"
    echo "$out" >&2
    if [ ${#DRY[@]} -gt 0 ]; then echo 0; else echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p'; fi
}

conv="$(submit cpu jobs/bridgesim.sbatch ACTION=convert CONFIG=configs/experiments/stage1_bridgesim_heldout.yaml)"
after=("--dependency=afterok:$conv" "--time=06:00:00")
gpu=()

specs=("none 0")
for deg in latency lateral_noise tail_lateral; do
    for level in 1 2 3; do specs+=("$deg $level"); done
done
for spec in "${specs[@]}"; do
    set -- $spec
    for family in transfuser ltf; do
        gpu+=("$(submit gpu jobs/bridgesim.sbatch ACTION=evaluate CONFIG=configs/experiments/popA_bridgesim_heldout.yaml \
            "OVERRIDES=policy.family=$family degradation.name=$1 degradation.level=$2" -- "${after[@]}")")
    done
    gpu+=("$(submit gpu jobs/bridgesim.sbatch ACTION=evaluate CONFIG=configs/experiments/popA_bridgesim_heldout_diffusiondrive.yaml \
        "OVERRIDES=degradation.name=$1 degradation.level=$2" -- "${after[@]}")")
done

for id in transfuser_seed1 transfuser_seed2 ltf_seed1 ltf_seed2 ego_status_mlp_seed0 ego_status_mlp_seed1 ego_status_mlp_seed2; do
    family="${id%_seed*}"; seed="${id##*_seed}"
    gpu+=("$(submit gpu jobs/bridgesim.sbatch ACTION=evaluate CONFIG=configs/experiments/popB_bridgesim_heldout.yaml \
        "OVERRIDES=policy.family=$family policy.seed=$seed" -- "${after[@]}")")
done

RUNS="$WETE_EXP_ROOT/runs/bridgesim_ext"
for id in transfuser_seed0 ltf_seed0 diffusiondrive_seed0; do
    src="$RUNS/A/$id/none__L0" dst="$RUNS/B/$id"
    if [ -e "$dst" ] || [ -L "$dst" ]; then echo "[heldout] exists: $dst"
    elif [ ${#DRY[@]} -gt 0 ]; then echo "[heldout] would link $dst -> $src"
    else mkdir -p "$RUNS/B" && ln -s "$src" "$dst" && echo "[heldout] linked $dst -> $src"; fi
done

dep="--dependency=afterany:$(IFS=:; echo "${gpu[*]}")"
common=(HELDOUT=1 SELECTION_BASES=transfuser,ltf BRIDGESIM_SCORE=DS)
a1="$(submit cpu jobs/analysis.sbatch NAME=heldout META=configs/meta.heldout.json "${common[@]}" -- "$dep")"
submit cpu jobs/analysis.sbatch NAME=heldout_v2 META=configs/meta.heldout_all.json "${common[@]}" -- "$dep" >/dev/null
submit cpu jobs/shared.sbatch NAME=heldout META=configs/meta.heldout_shared.json TAG=ext -- "--dependency=afterok:$a1" >/dev/null
submit cpu jobs/export_tables.sbatch -- "$dep" >/dev/null
echo "[heldout] queued: 1 convert, ${#gpu[@]} GPU, 4 analysis/table jobs"
