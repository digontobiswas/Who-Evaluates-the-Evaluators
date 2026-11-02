#!/bin/bash
# After the held-out GPU runs: write their reports (one scene that BridgeSim cannot simulate is
# excluded for every run, configs/experiments/stage1_bridgesim_heldout.yaml), then rerun the
# held-out analyses, the shared-scene comparison and the tables. CPU only.
#
#     bash scripts/submit_heldout_report.sh [--dry-run]
#
# Afterwards push results/ (git add results && git commit -m "held-out results" && git push).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi

submit() {
    local out
    out="$(bash "$SCRIPT_DIR/submit.sh" ${DRY[@]+"${DRY[@]}"} cpu "$@")"
    echo "$out" >&2
    if [ ${#DRY[@]} -gt 0 ]; then echo 0; else echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p'; fi
}

r="$(submit jobs/heldout_report.sbatch)"
dep="--dependency=afterok:$r"
common=(HELDOUT=1 SELECTION_BASES=transfuser,ltf BRIDGESIM_SCORE=DS)
a1="$(submit jobs/analysis.sbatch NAME=heldout META=configs/meta.heldout.json "${common[@]}" -- "$dep")"
submit jobs/analysis.sbatch NAME=heldout_v2 META=configs/meta.heldout_all.json "${common[@]}" -- "$dep" >/dev/null
submit jobs/shared.sbatch NAME=heldout META=configs/meta.heldout_shared.json TAG=ext -- "--dependency=afterok:$a1" >/dev/null
submit jobs/export_tables.sbatch -- "$dep" >/dev/null
echo "[heldout-report] 5 CPU jobs queued (report $r)"
