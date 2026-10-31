#!/bin/bash
# Final analyses, queued to start only after every job you currently have queued or running has
# ended. Submit this last, after all experiment jobs (including the NAVSIM v2 seeds).
#
#     bash scripts/submit_final.sh [--dry-run]
#
#   final         main analysis: selection/evaluation split, degradations validated on TransFuser and
#                 LTF (DiffusionDrive held out), NAVSIM v1 + open-loop L2 + BridgeSim (DS)
#   final_v2      the same with NAVSIM v2 one-stage and two-stage added (meta.all.json)
#   final_all     sensitivity: no split (all scenarios)
#   final_epdms   sensitivity: BridgeSim EPDMS instead of DS
#   shared        shared-scenario comparison on 'final' (v1 evaluators) and on 'final_v2' (TAG=v2)
#   tables        results/tables/*.csv rebuilt from every summary.json
# Afterwards push results/ (git add results && git commit -m "final results" && git push).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi

pending="$(squeue -h -u "$USER" -o %i | paste -sd: -)"
dep=()
[ -n "$pending" ] && dep=("--dependency=afterany:$pending")
echo "[final] waiting for $(echo "$pending" | tr ':' '\n' | grep -c . || true) jobs"

submit() {
    # submit <job> <vars...>: prints the job id
    local out
    out="$(bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu "$@")"
    echo "$out" >&2
    if [ ${#DRY[@]} -gt 0 ]; then echo 0; else echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p'; fi
}

common=(SPLIT=1 SELECTION_BASES=transfuser,ltf BRIDGESIM_SCORE=DS)
a1="$(submit jobs/analysis.sbatch NAME=final META=configs/meta.bridgesim.json "${common[@]}" -- ${dep[@]+"${dep[@]}"})"
a2="$(submit jobs/analysis.sbatch NAME=final_v2 META=configs/meta.all.json "${common[@]}" -- ${dep[@]+"${dep[@]}"})"
submit jobs/analysis.sbatch NAME=final_all META=configs/meta.bridgesim.json SELECTION_BASES=transfuser,ltf -- ${dep[@]+"${dep[@]}"} >/dev/null
submit jobs/analysis.sbatch NAME=final_epdms META=configs/meta.bridgesim.json SPLIT=1 SELECTION_BASES=transfuser,ltf BRIDGESIM_SCORE=EPDMS_no_ep -- ${dep[@]+"${dep[@]}"} >/dev/null
submit jobs/shared.sbatch NAME=final META=configs/meta.bridgesim.json -- "--dependency=afterok:$a1" >/dev/null
submit jobs/shared.sbatch NAME=final_v2 META=configs/meta.all.json TAG=v2 -- "--dependency=afterok:$a2" >/dev/null
submit jobs/export_tables.sbatch -- ${dep[@]+"${dep[@]}"} >/dev/null
echo "[final] 7 jobs queued (analyses $a1, $a2, ...)"
