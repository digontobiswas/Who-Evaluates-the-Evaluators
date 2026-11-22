#!/bin/bash
# After every job of scripts/submit_revision.sh has ended: collect all runs, rerun the main
# held-out analysis with the projected evaluators added (matched bases), then the reviewer
# analyses (wete.cli.revision) and the tables. CPU only.
#
#     bash scripts/submit_revision_analysis.sh [--dry-run]
#
# Afterwards push results/ (git add results && git commit -m "revision results" && git push).

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

a="$(submit jobs/analysis.sbatch NAME=revision META=configs/meta.revision.json HELDOUT=1 \
    SELECTION_BASES=transfuser,ltf MEASURE_BASES=transfuser_seed0,ltf_seed0,diffusiondrive_seed0 BRIDGESIM_SCORE=DS)"
submit jobs/revision.sbatch NAME=revision -- "--dependency=afterok:$a" >/dev/null
submit jobs/export_tables.sbatch >/dev/null
echo "[revision-analysis] 3 CPU jobs queued"
