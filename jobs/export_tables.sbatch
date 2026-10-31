#!/bin/bash
#SBATCH --job-name=wete-tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Rebuild results/tables/*.csv from every summary.json under results/ (CPU, seconds).
#     bash scripts/submit.sh cpu jobs/export_tables.sbatch

set -euo pipefail
: "${WETE_REPO:?Submit with scripts/submit.sh so that WETE_REPO is set.}"
# shellcheck source=scripts/env.sh
source "$WETE_REPO/scripts/env.sh"
wete_load_modules
wete_activate
export PYTHONPATH="$WETE_REPO/src"
cd "$WETE_REPO"
python -m wete.cli.export_tables
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
