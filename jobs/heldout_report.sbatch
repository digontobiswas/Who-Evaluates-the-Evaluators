#!/bin/bash
#SBATCH --job-name=wete-heldout-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Writes the report (per_scenario.csv, summary.json, results/ export) of every held-out BridgeSim
# run from the scenarios already scored. No simulation: CPU only, a few minutes.
#
#     bash scripts/submit.sh cpu jobs/heldout_report.sbatch
#
# The runs are those of scripts/submit_heldout.sh: TransFuser, LTF and DiffusionDrive seed 0
# (clean + latency, lateral_noise, tail_lateral L1-3) and Population B (TF/LTF seeds 1-2, Ego MLP
# seeds 0-2). A run with an unscored expected scenario is reported as incomplete, not hidden.

set -euo pipefail
: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_use_bridgesim
cd "$WETE_REPO"

report() {
    # report <config> <overrides...>
    local cfg="$1"; shift
    echo "[job] report $cfg $*"
    python -m wete.cli.bridgesim_run --config "configs/experiments/$cfg.yaml" --action report "$@"
}

for spec in "none 0" "latency 1" "latency 2" "latency 3" "lateral_noise 1" "lateral_noise 2" \
            "lateral_noise 3" "tail_lateral 1" "tail_lateral 2" "tail_lateral 3"; do
    set -- $spec
    for family in transfuser ltf; do
        report popA_bridgesim_heldout "policy.family=$family" "degradation.name=$1" "degradation.level=$2"
    done
    report popA_bridgesim_heldout_diffusiondrive "degradation.name=$1" "degradation.level=$2"
done
for id in transfuser_seed1 transfuser_seed2 ltf_seed1 ltf_seed2 ego_status_mlp_seed0 ego_status_mlp_seed1 ego_status_mlp_seed2; do
    report popB_bridgesim_heldout "policy.family=${id%_seed*}" "policy.seed=${id##*_seed}"
done
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
