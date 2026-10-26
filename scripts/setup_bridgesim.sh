#!/bin/bash
# Build the BridgeSim environment. Run on the LOGIN node (it needs the internet), after
# scripts/setup_env.sh has built the NAVSIM environment:
#
#     bash scripts/setup_bridgesim.sh
#
# What it does, in order:
#   1. Creates a second venv ($WETE_BRIDGESIM_VENV) from the same relocatable Python 3.9.
#   2. Installs requirements/bridgesim-cp39-manylinux2014.lock.txt: the NAVSIM pins unchanged
#      plus MetaDrive's dependencies, wheels only (four pure-Python packages build from source).
#   3. Checks out BridgeSim at the commit pinned in configs/third_party.env. BridgeSim and its
#      vendored MetaDrive stay source checkouts on PYTHONPATH; nothing is compiled.
#   4. Downloads MetaDrive's assets (3D models, textures) into the checkout, and registers the
#      pedestrian and traffic-cone model paths with Panda3D (BridgeSim Q&A).
#   5. Verifies the imports the probe job needs.
#
# Safe to re-run: finished steps are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
# shellcheck source=configs/third_party.env
source "$WETE_REPO/configs/third_party.env"

LOCK_FILE="$WETE_REPO/requirements/bridgesim-cp39-manylinux2014.lock.txt"
BRIDGESIM_DIR="$WETE_THIRD_PARTY/bridgesim"

log() { echo "[setup-bridgesim] $*"; }
die() {
    echo "[setup-bridgesim] ERROR: $*" >&2
    exit 1
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "WARNING: running inside a SLURM job. Compute nodes have no internet; downloads will fail."
fi

# ---------------------------------------------------------------------------------------------
# 1. Interpreter: the one the NAVSIM venv was built from
# ---------------------------------------------------------------------------------------------
[ -f "$WETE_VENV/pyvenv.cfg" ] || die "NAVSIM venv not found at $WETE_VENV. Run scripts/setup_env.sh first."
BASE_DIR="$(sed -n 's/^home *= *//p' "$WETE_VENV/pyvenv.cfg")"
BASE_PYTHON="$BASE_DIR/python3.9"
[ -x "$BASE_PYTHON" ] || die "Base interpreter $BASE_PYTHON (from $WETE_VENV/pyvenv.cfg) not found."
[ -d "$WETE_THIRD_PARTY/nuplan-devkit/nuplan" ] || die "nuPlan devkit checkout missing. Run scripts/setup_env.sh first."

if [ ! -f "$WETE_BRIDGESIM_VENV/bin/activate" ]; then
    log "Creating venv at $WETE_BRIDGESIM_VENV"
    "$BASE_PYTHON" -m venv "$WETE_BRIDGESIM_VENV"
fi
wete_activate "$WETE_BRIDGESIM_VENV"
python -m pip install --quiet --upgrade "pip>=24.2,<26"

# ---------------------------------------------------------------------------------------------
# 2. Pinned wheels
# ---------------------------------------------------------------------------------------------
STAMP="$WETE_BRIDGESIM_VENV/.wete_lock_sha256"
LOCK_SHA="$(sha256sum "$LOCK_FILE" | cut -d' ' -f1)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$LOCK_SHA" ]; then
    log "Lock file already installed."
else
    log "Installing $(grep -c '==' "$LOCK_FILE") pinned packages (most come from the pip cache of the NAVSIM setup)"
    python -m pip install \
        --only-binary=:all: \
        --no-binary=antlr4-python3-runtime,control,progressbar,pyrr \
        -r "$LOCK_FILE"
    echo "$LOCK_SHA" >"$STAMP"
fi

# ---------------------------------------------------------------------------------------------
# 3. BridgeSim at the pinned commit
# ---------------------------------------------------------------------------------------------
head_of() { (cd "$1" && git rev-parse HEAD); }
if [ -d "$BRIDGESIM_DIR/.git" ] && [ "$(head_of "$BRIDGESIM_DIR")" = "$BRIDGESIM_COMMIT" ]; then
    log "BridgeSim already at $BRIDGESIM_COMMIT"
else
    if [ ! -d "$BRIDGESIM_DIR/.git" ]; then
        log "Cloning $BRIDGESIM_URL"
        git clone --quiet "$BRIDGESIM_URL" "$BRIDGESIM_DIR"
    fi
    (cd "$BRIDGESIM_DIR" && git fetch --quiet origin && git checkout --quiet "$BRIDGESIM_COMMIT")
    [ "$(head_of "$BRIDGESIM_DIR")" = "$BRIDGESIM_COMMIT" ] || die "BridgeSim checkout is not at $BRIDGESIM_COMMIT"
    log "BridgeSim at $BRIDGESIM_COMMIT"
fi

wete_use_bridgesim

# ---------------------------------------------------------------------------------------------
# 4. MetaDrive assets and Panda3D model paths
# ---------------------------------------------------------------------------------------------
ASSETS="$BRIDGESIM_DIR/metadrive/metadrive/assets"
if [ -f "$ASSETS/version.txt" ]; then
    log "MetaDrive assets present (version $(head -1 "$ASSETS/version.txt"))"
else
    log "Downloading MetaDrive assets"
    python -m metadrive.pull_asset
fi
[ -f "$ASSETS/version.txt" ] || die "MetaDrive assets missing after download."

PRC="$(python -c 'import panda3d, os; print(os.path.join(os.path.dirname(panda3d.__file__), "etc", "Config.prc"))')"
[ -f "$PRC" ] || die "Panda3D Config.prc not found at $PRC"
for model in pedestrian traffic_cone; do
    line="model-path $ASSETS/models/$model"
    if ! grep -qxF "$line" "$PRC"; then
        echo "$line" >>"$PRC"
        log "Registered $line"
    fi
done

# ---------------------------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------------------------
log "Verifying imports"
python - <<'PY'
import sys

import numpy, panda3d, torch  # noqa: F401,E401
import metadrive
from metadrive.envs.scenario_env import ScenarioEnv  # noqa: F401
from bridgesim.evaluation.models.transfuser_adapter import TransfuserAdapter  # noqa: F401
from bridgesim.evaluation.models.ltf_adapter import LTFAdapter  # noqa: F401
from bridgesim.evaluation.core.base_evaluator import BaseEvaluator  # noqa: F401
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: F401
import wete.cli.bridgesim_probe  # noqa: F401

print(f"[setup-bridgesim] python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}")
from panda3d.core import PandaSystem
print(f"[setup-bridgesim] panda3d {PandaSystem.getVersionString()}, metadrive from {metadrive.__file__}")
PY
python "$BRIDGESIM_DIR/converters/openscene/convert_openscene_with_filter.py" --help >/dev/null
log "Done. BridgeSim environment: $WETE_BRIDGESIM_VENV"
log "Next: bash scripts/submit.sh gpu jobs/bridgesim_probe.sbatch"
