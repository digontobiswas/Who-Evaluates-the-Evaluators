#!/bin/bash
# Runs requested by the review (Nov 2026). Nothing is retrained; every run reuses the pipeline.
#
#     bash scripts/submit_revision.sh [--dry-run]            # everything (84 jobs)
#     ONLY=projection bash scripts/submit_revision.sh        # item 1: 43 CPU jobs
#     ONLY=head       bash scripts/submit_revision.sh        # item 2: 15 CPU jobs
#     ONLY=replan     bash scripts/submit_revision.sh        # item 3: 26 GPU jobs
#
# Item 1, projected NAVSIM v1 (runs/navsim_v1_proj): TransFuser and LTF seed 0 (clean, tail_lateral,
#   tail_stretch, lateral_noise, late_braking L1-3), DiffusionDrive (clean, tail_lateral,
#   tail_stretch, lateral_noise L1-3) and Population B (TF/LTF seeds 1-2, Ego MLP seeds 0-2; seed-0
#   clean runs linked). The harmful twins give the projected score its own benign-to-harmful ratio.
# Item 2, head-only twins (head_lateral L1-3) in the existing evaluators: NAVSIM v1 for TF, LTF, DD
#   and NAVSIM v2 for TF and LTF (v2 has no DD). L2 comes from the v1 runs.
# Item 3, BridgeSim replanning every 1.0 s on NavHard (runs/bridgesim_rr10, _rr10_t20): TF and LTF
#   seed 0, clean + old twins (cut 1.5 s) and re-derived twins (cut 2.0 s), tail_lateral and
#   tail_stretch L1-3.
# After every job has ended: bash scripts/submit_revision_analysis.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
RUNS="$WETE_EXP_ROOT/runs"
count=0

want() { [ -z "${ONLY:-}" ] || [ "${ONLY}" = "$1" ]; }

link() {
    # link <target dir> <link path>
    if [ -e "$2" ] || [ -L "$2" ]; then echo "[revision] exists: $2"
    elif [ ${#DRY[@]} -gt 0 ]; then echo "[revision] would link $2 -> $1"
    else mkdir -p "$(dirname "$2")" && ln -s "$1" "$2" && echo "[revision] linked $2 -> $1"; fi
}

navsim() {
    # navsim <config> <hours> <overrides> [NAVSIM_VERSION=v2]
    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch ${4:+"$4"} \
        "CONFIG=configs/experiments/$1.yaml" "STAGES=select,score,report" "OVERRIDES=$3" -- "--time=$2:00:00"
    count=$((count + 1))
}

if want projection; then
    for fam in transfuser ltf diffusiondrive; do
        hours=06; [ "$fam" = diffusiondrive ] && hours=12
        degs="tail_lateral tail_stretch lateral_noise late_braking"
        [ "$fam" = diffusiondrive ] && degs="tail_lateral tail_stretch lateral_noise"
        navsim "popA_${fam}_navtest_proj" "$hours" "degradation.name=none degradation.level=0"
        for deg in $degs; do
            for level in 1 2 3; do
                navsim "popA_${fam}_navtest_proj" "$hours" "degradation.name=$deg degradation.level=$level"
            done
        done
    done
    for id in transfuser_seed1 transfuser_seed2 ltf_seed1 ltf_seed2 ego_status_mlp_seed0 ego_status_mlp_seed1 ego_status_mlp_seed2; do
        navsim "popB_${id%_seed*}_navtest_proj" 06 "policy.seed=${id##*_seed}"
    done
    for id in transfuser_seed0 ltf_seed0 diffusiondrive_seed0; do
        link "$RUNS/navsim_v1_proj/A/$id/none__L0" "$RUNS/navsim_v1_proj/B/$id"
    done
fi

if want head; then
    for fam in transfuser ltf diffusiondrive; do
        hours=06; [ "$fam" = diffusiondrive ] && hours=12
        for level in 1 2 3; do
            navsim "popA_${fam}_navtest" "$hours" "degradation.name=head_lateral degradation.level=$level"
            [ "$fam" = diffusiondrive ] || \
                navsim "popA_${fam}_navtest_v2" 06 "degradation.name=head_lateral degradation.level=$level" NAVSIM_VERSION=v2
        done
    done
fi

if want replan; then
    for fam in transfuser ltf; do
        bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate \
            CONFIG=configs/experiments/popA_bridgesim_rr10.yaml "OVERRIDES=policy.family=$fam" -- --time=06:00:00
        count=$((count + 1))
        for cfg in popA_bridgesim_rr10 popA_bridgesim_rr10_t20; do
            for pert in tail_lateral tail_stretch; do
                for level in 1 2 3; do
                    bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate \
                        "CONFIG=configs/experiments/$cfg.yaml" \
                        "OVERRIDES=policy.family=$fam degradation.name=$pert degradation.level=$level" -- --time=06:00:00
                    count=$((count + 1))
                done
            done
        done
        link "$RUNS/bridgesim_rr10/A/${fam}_seed0/none__L0" "$RUNS/bridgesim_rr10_t20/A/${fam}_seed0/none__L0"
    done
fi
echo "[revision] $count jobs ${ONLY:+(ONLY=$ONLY)}"
