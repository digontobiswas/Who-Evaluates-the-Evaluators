#!/bin/bash
# Re-analysis after the 1 Nov check of the final results. CPU only, no new experiment runs.
#
#     bash scripts/submit_reanalysis.sh [--dry-run]
#
#   final_v2       rerun with pairwise Population B agreement: every pair of evaluators is compared
#                  on the policies BOTH scored (before, one intersection over all five evaluators
#                  reduced every pair to LTF + Ego MLP, because NAVSIM v2 two-stage cannot run
#                  TransFuser and v2 has no DiffusionDrive)
#   shared_v2      the shared-scenario comparison on final_v2, same fix
#   final_seed0    sensitivity: split as in 'final', but degradations and benign perturbations are
#                  validated on the seed-0 bases only (the G2 protocol); seeds 1-2 are replication.
#                  Reported next to 'final' (all seeds), never instead of it.
# Afterwards push results/ (git add results && git commit -m "re-analysis" && git push).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi

submit() {
    # submit <job> <vars...>: prints the job id
    local out
    out="$(bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu "$@")"
    echo "$out" >&2
    if [ ${#DRY[@]} -gt 0 ]; then echo 0; else echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p'; fi
}

a="$(submit jobs/analysis.sbatch NAME=final_v2 META=configs/meta.all.json SPLIT=1 \
    SELECTION_BASES=transfuser,ltf BRIDGESIM_SCORE=DS)"
submit jobs/shared.sbatch NAME=final_v2 META=configs/meta.all.json TAG=v2 -- "--dependency=afterok:$a" >/dev/null
submit jobs/analysis.sbatch NAME=final_seed0 META=configs/meta.bridgesim.json SPLIT=1 \
    SELECTION_BASES=transfuser_seed0,ltf_seed0 BRIDGESIM_SCORE=DS >/dev/null
echo "[reanalysis] 3 CPU jobs queued"
