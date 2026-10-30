#!/bin/bash
# Wave 1 after the senior's review (31 Oct 2026): replication, latency in NAVSIM, and the start of
# NAVSIM v2. 58 jobs, below the QoS limit of 100 queued jobs.
#
#     bash scripts/submit_review_fixes.sh [--dry-run]
#
#   NAVSIM v1 (CPU, 30):  latency L1-3 for TF and LTF seed 0; lateral_noise and latency L1-3 for
#                         TF and LTF seeds 1-2. Clean runs of seeds 1-2 are the Population B runs,
#                         linked into runs/navsim_v1/A/<policy>/none__L0.
#   BridgeSim (GPU, 24):  lateral_noise and latency L1-3 for TF and LTF seeds 1-2 (clean runs linked
#                         from Population B the same way). Seed 0 already has both.
#   NAVSIM v2 (CPU, 4):   metric caches for navtest and navhard_two_stage; then, after each cache,
#                         one job that tests the v2 pipeline: 64 navtest scenarios, and TF seed 0 on
#                         navhard_two_stage (a real run, kept). Wave 2 (scripts/submit_navsim_v2.sh)
#                         waits until both have passed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
RUNS="$WETE_EXP_ROOT/runs"

link_b_to_a() {
    # link_b_to_a <evaluator> <policy>: Population B run -> Population A clean run of that base
    local src="$RUNS/$1/B/$2" dst="$RUNS/$1/A/$2/none__L0"
    [ -d "$src" ] || { echo "[review] ERROR: $src missing (run Population B first)" >&2; exit 1; }
    if [ -e "$dst" ]; then echo "[review] exists: $dst"
    elif [ ${#DRY[@]} -gt 0 ]; then echo "[review] would link $dst -> $src"
    else mkdir -p "$RUNS/$1/A/$2" && ln -s "$src" "$dst" && echo "[review] linked $dst -> $src"; fi
}

# submit_capture <args for submit.sh>: prints the job id (or DRYRUN)
submit_capture() {
    local out
    out="$(bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} "$@")"
    echo "$out" >&2
    if [ ${#DRY[@]} -gt 0 ]; then echo DRYRUN; else echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p'; fi
}

count=0
for fam in transfuser ltf; do
    for seed in 1 2; do
        link_b_to_a navsim_v1 "${fam}_seed${seed}"
        link_b_to_a bridgesim "${fam}_seed${seed}"
    done
done

# ---- NAVSIM v1 -------------------------------------------------------------------------------
for fam in transfuser ltf; do
    for seed in 0 1 2; do
        degs=(latency)
        [ "$seed" != 0 ] && degs=(lateral_noise latency)
        for deg in "${degs[@]}"; do
            for level in 1 2 3; do
                bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu jobs/navsim_v1_pdms.sbatch \
                    "CONFIG=configs/experiments/popA_${fam}_navtest.yaml" "STAGES=select,score,report" \
                    "OVERRIDES=policy.seed=$seed degradation.name=$deg degradation.level=$level"
                count=$((count + 1))
            done
        done
    done
done

# ---- BridgeSim --------------------------------------------------------------------------------
for fam in transfuser ltf; do
    for seed in 1 2; do
        for deg in lateral_noise latency; do
            for level in 1 2 3; do
                bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} gpu jobs/bridgesim.sbatch ACTION=evaluate \
                    CONFIG=configs/experiments/popA_bridgesim_navhard.yaml \
                    "OVERRIDES=policy.family=$fam policy.seed=$seed degradation.name=$deg degradation.level=$level" \
                    -- --time=06:00:00
                count=$((count + 1))
            done
        done
    done
done

# ---- NAVSIM v2: caches, then one checked run each ----------------------------------------------
cache_nt="$(submit_capture cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
    CONFIG=configs/experiments/cache_navtest_v2.yaml STAGES=select,cache -- --time=24:00:00)"
cache_nh="$(submit_capture cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
    CONFIG=configs/experiments/cache_navhard_v2.yaml STAGES=select,cache -- --time=24:00:00)"
submit_capture cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
    CONFIG=configs/experiments/smoke_transfuser_navtest_v2.yaml STAGES=select,score,report \
    -- "--dependency=afterok:$cache_nt" >/dev/null
submit_capture cpu jobs/navsim_v1_pdms.sbatch NAVSIM_VERSION=v2 \
    CONFIG=configs/experiments/popA_transfuser_navhard_v2.yaml STAGES=select,score,report \
    "OVERRIDES=degradation.name=none degradation.level=0" -- "--dependency=afterok:$cache_nh" >/dev/null
count=$((count + 4))
echo "[review] $count jobs submitted (v2 caches: $cache_nt, $cache_nh)"
