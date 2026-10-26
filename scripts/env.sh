# shellcheck shell=bash
# Common environment for every script and job. Source it, do not execute it:
#
#     source scripts/env.sh
#
# Reads configs/hpc/cluster.env and derives all other locations from WETE_ROOT. This file is
# the only place where directory layout is decided; Python reads the same variables through
# configs/default.yaml.

_wete_fail() {
    echo "[wete] ERROR: $*" >&2
    return 1
}

# Repository root. Inside an sbatch job the script runs from a spool copy, so the submit
# wrapper passes WETE_REPO explicitly; elsewhere it is derived from this file's location.
if [ -z "${WETE_REPO:-}" ]; then
    WETE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export WETE_REPO

WETE_CLUSTER_ENV="${WETE_CLUSTER_ENV:-$WETE_REPO/configs/hpc/cluster.env}"
if [ ! -f "$WETE_CLUSTER_ENV" ]; then
    _wete_fail "$WETE_CLUSTER_ENV not found. Copy configs/hpc/cluster.env.example to it and fill it in." || return 1
fi
if grep -q $'\r' "$WETE_CLUSTER_ENV"; then
    _wete_fail "$WETE_CLUSTER_ENV has Windows line endings. Fix with: sed -i 's/\\r\$//' $WETE_CLUSTER_ENV" || return 1
fi
set -a
# shellcheck disable=SC1090
. "$WETE_CLUSTER_ENV"
set +a

if [ -z "${WETE_ROOT:-}" ]; then
    _wete_fail "WETE_ROOT is empty in $WETE_CLUSTER_ENV." || return 1
fi

export WETE_ROOT
export WETE_VENV_ROOT="${WETE_VENV_ROOT:-$WETE_ROOT/venvs}"
export WETE_VENV="$WETE_VENV_ROOT/navsim"
export WETE_BRIDGESIM_VENV="$WETE_VENV_ROOT/bridgesim"
export WETE_TOOLS="$WETE_ROOT/tools"
export WETE_THIRD_PARTY="$WETE_ROOT/third_party"
export WETE_DATA_ROOT="$WETE_ROOT/data"
export WETE_CKPT_ROOT="$WETE_ROOT/checkpoints"
export WETE_EXP_ROOT="$WETE_ROOT/exp"
export WETE_CACHE_ROOT="$WETE_ROOT/cache"
export WETE_SLURM_LOGS="$WETE_ROOT/slurm_logs"

# Variables the NAVSIM devkit reads (docs/install.md of the devkit).
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="$WETE_DATA_ROOT/openscene"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_EXP_ROOT="$WETE_EXP_ROOT/navsim"

# Keep every cache off $HOME, whose quota is small.
export PIP_CACHE_DIR="$WETE_CACHE_ROOT/pip"
export HF_HOME="$WETE_CACHE_ROOT/huggingface"
export TORCH_HOME="$WETE_CACHE_ROOT/torch"
export MPLCONFIGDIR="$WETE_CACHE_ROOT/matplotlib"
export XDG_CACHE_HOME="$WETE_CACHE_ROOT/xdg"
export PYTHONDONTWRITEBYTECODE=1

# Load optional modules listed in cluster.env.
wete_load_modules() {
    [ -n "${WETE_MODULES:-}" ] || return 0
    if ! command -v module >/dev/null 2>&1; then
        _wete_fail "WETE_MODULES is set but the 'module' command is not available." || return 1
    fi
    local mod
    for mod in $WETE_MODULES; do
        module load "$mod" || { _wete_fail "module load $mod failed." || return 1; }
    done
}

# Activate a virtual environment: the NAVSIM one by default, or the directory given.
wete_activate() {
    local venv="${1:-$WETE_VENV}"
    if [ ! -f "$venv/bin/activate" ]; then
        _wete_fail "Virtual environment not found at $venv. Run scripts/setup_env.sh (NAVSIM) or scripts/setup_bridgesim.sh (BridgeSim)." || return 1
    fi
    # The activate script reads variables that may be unset; tolerate that under `set -u`.
    local restore_nounset=0
    case $- in *u*) restore_nounset=1; set +u ;; esac
    # shellcheck disable=SC1091
    . "$venv/bin/activate"
    if [ "$restore_nounset" = 1 ]; then set -u; fi
    # The relocatable CPython has no system certificate store configured; point OpenSSL at
    # the certifi bundle so that plain urllib downloads verify correctly too.
    local cert
    cert="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [ -n "$cert" ]; then
        export SSL_CERT_FILE="$cert"
        export REQUESTS_CA_BUNDLE="$cert"
    fi
}

# Put one NAVSIM devkit (v1 or v2), the nuPlan devkit and this repository on PYTHONPATH.
# The two devkits share the package name "navsim", so they are never installed with pip;
# a job selects exactly one here.
wete_use_navsim() {
    local version="${1:-}"
    case "$version" in
        v1 | v2) ;;
        *) _wete_fail "wete_use_navsim expects v1 or v2, got '$version'." || return 1 ;;
    esac
    export NAVSIM_DEVKIT_ROOT="$WETE_THIRD_PARTY/navsim_$version"
    if [ ! -d "$NAVSIM_DEVKIT_ROOT/navsim" ]; then
        _wete_fail "NAVSIM $version devkit not found at $NAVSIM_DEVKIT_ROOT. Run: bash scripts/setup_env.sh" || return 1
    fi
    export PYTHONPATH="$WETE_REPO/src:$NAVSIM_DEVKIT_ROOT:$WETE_THIRD_PARTY/nuplan-devkit"
}

# Activate the BridgeSim environment and put BridgeSim, its vendored MetaDrive, the nuPlan devkit
# and this repository on PYTHONPATH. No NAVSIM devkit: BridgeSim ships its own copy of the agents.
wete_use_bridgesim() {
    wete_activate "$WETE_BRIDGESIM_VENV" || return 1
    export BRIDGESIM_ROOT="$WETE_THIRD_PARTY/bridgesim"
    if [ ! -d "$BRIDGESIM_ROOT/bridgesim" ]; then
        _wete_fail "BridgeSim not found at $BRIDGESIM_ROOT. Run: bash scripts/setup_bridgesim.sh" || return 1
    fi
    export PYTHONPATH="$WETE_REPO/src:$BRIDGESIM_ROOT:$BRIDGESIM_ROOT/metadrive:$WETE_THIRD_PARTY/nuplan-devkit"
}
