#!/bin/bash
# Build the NAVSIM virtual environment. Run on the LOGIN node (it needs the internet):
#
#     bash scripts/setup_env.sh
#
# What it does, in order:
#   1. Gets a Python 3.9 interpreter. The cluster's own `python` is 2.7 and its module
#      Python 3 has no working ssl module, so pip cannot reach PyPI with it. We unpack a
#      pinned relocatable CPython 3.9 (ssl, lzma and sqlite built in; needs glibc >= 2.17)
#      and use nothing from the system Python.
#   2. Creates an ordinary venv with `python -m venv`. No conda.
#   3. Installs the pinned wheel set in requirements/navsim-cp39-manylinux2014.lock.txt with
#      source builds forbidden, so nothing is compiled on the cluster.
#   4. Checks out the NAVSIM v1.1, NAVSIM v2 and nuPlan devkits at pinned commits.
#   5. Verifies the imports the first job needs and writes an environment report.
#
# Safe to re-run: finished steps are detected and skipped.
#
# Options:
#   --python-tarball FILE   Use an already-downloaded CPython archive (same file as
#                           PYTHON_STANDALONE_URL in configs/third_party.env) instead of
#                           downloading it. For a login node that cannot reach GitHub releases.
#   --wheelhouse DIR        Install only from wheels in DIR (pip --no-index --find-links).
#                           For a login node that cannot reach PyPI.
#   --recreate              Delete the venv first and rebuild it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
# shellcheck source=configs/third_party.env
source "$WETE_REPO/configs/third_party.env"

LOCK_FILE="$WETE_REPO/requirements/navsim-cp39-manylinux2014.lock.txt"
PYTHON_TARBALL=""
WHEELHOUSE=""
RECREATE=0

log() { echo "[setup] $*"; }
die() {
    echo "[setup] ERROR: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --python-tarball)
            PYTHON_TARBALL="${2:?--python-tarball needs a file}"
            shift 2
            ;;
        --wheelhouse)
            WHEELHOUSE="${2:?--wheelhouse needs a directory}"
            shift 2
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        -h | --help)
            sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
done

if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "WARNING: running inside a SLURM job. Compute nodes have no internet; downloads will fail."
fi

mkdir -p "$WETE_ROOT" "$WETE_VENV_ROOT" "$WETE_TOOLS" "$WETE_THIRD_PARTY" "$WETE_DATA_ROOT" \
    "$WETE_CKPT_ROOT" "$WETE_EXP_ROOT" "$WETE_CACHE_ROOT" "$WETE_SLURM_LOGS" "$PIP_CACHE_DIR"

# ---------------------------------------------------------------------------------------------
# 1. Python 3.9 interpreter
# ---------------------------------------------------------------------------------------------
python_is_usable() {
    # Exactly 3.9, and with the modules the system builds on this cluster are known to lack.
    "$1" - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info[:2] == (3, 9), sys.version
import ssl, lzma, bz2, sqlite3, ctypes, zlib, venv, ensurepip  # noqa: F401,E401
PY
}

download() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 30 -o "$dest.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=60 -O "$dest.part" "$url"
    else
        die "Neither curl nor wget is available."
    fi
    mv "$dest.part" "$dest"
}

if [ -n "${WETE_PYTHON:-}" ]; then
    BASE_PYTHON="$WETE_PYTHON"
    python_is_usable "$BASE_PYTHON" || die "WETE_PYTHON=$BASE_PYTHON is not a Python 3.9 with ssl, lzma, bz2, sqlite3 and venv."
    log "Using interpreter from cluster.env: $BASE_PYTHON"
else
    PYTHON_HOME="$WETE_TOOLS/cpython-$PYTHON_STANDALONE_VERSION"
    BASE_PYTHON="$PYTHON_HOME/bin/python3.9"
    if [ -x "$BASE_PYTHON" ] && python_is_usable "$BASE_PYTHON"; then
        log "Relocatable CPython already unpacked: $BASE_PYTHON"
    else
        archive="$WETE_TOOLS/cpython-$PYTHON_STANDALONE_VERSION.tar.gz"
        if [ -n "$PYTHON_TARBALL" ]; then
            [ -f "$PYTHON_TARBALL" ] || die "--python-tarball file not found: $PYTHON_TARBALL"
            cp "$PYTHON_TARBALL" "$archive"
        elif [ ! -f "$archive" ]; then
            log "Downloading CPython $PYTHON_STANDALONE_VERSION (about 28 MB)"
            download "$PYTHON_STANDALONE_URL" "$archive" || die \
                "Download failed. On a machine with internet, fetch $PYTHON_STANDALONE_URL , copy it here, and re-run with --python-tarball FILE."
        fi
        echo "$PYTHON_STANDALONE_SHA256  $archive" | sha256sum -c - >/dev/null \
            || die "Checksum mismatch for $archive. Delete it and re-run."
        mkdir -p "$PYTHON_HOME"
        tar -xzf "$archive" -C "$PYTHON_HOME" --strip-components=1
        python_is_usable "$BASE_PYTHON" || die \
            "Unpacked CPython does not run here. Check glibc with: ldd --version (2.17 or newer is required)."
        log "Unpacked CPython: $("$BASE_PYTHON" --version 2>&1), $("$BASE_PYTHON" -c 'import ssl; print(ssl.OPENSSL_VERSION)')"
    fi
fi

# ---------------------------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------------------------
if [ "$RECREATE" = 1 ] && [ -d "$WETE_VENV" ]; then
    [ -f "$WETE_VENV/pyvenv.cfg" ] || die "$WETE_VENV does not look like a venv; refusing to delete it."
    log "Removing existing venv $WETE_VENV"
    rm -rf "$WETE_VENV"
fi
if [ ! -f "$WETE_VENV/bin/activate" ]; then
    log "Creating venv at $WETE_VENV"
    "$BASE_PYTHON" -m venv "$WETE_VENV"
fi
wete_activate

PIP_SOURCE_ARGS=()
if [ -n "$WHEELHOUSE" ]; then
    [ -d "$WHEELHOUSE" ] || die "--wheelhouse directory not found: $WHEELHOUSE"
    PIP_SOURCE_ARGS=(--no-index --find-links "$WHEELHOUSE")
    log "Offline install from $WHEELHOUSE"
else
    python -m pip install --quiet --upgrade "pip>=24.2,<26"
fi
log "pip: $(python -m pip --version)"

# ---------------------------------------------------------------------------------------------
# 3. Pinned wheels. Source builds are forbidden except for two pure-Python packages that
#    publish no wheel, so a missing wheel fails here instead of invoking a compiler.
# ---------------------------------------------------------------------------------------------
STAMP="$WETE_VENV/.wete_lock_sha256"
LOCK_SHA="$(sha256sum "$LOCK_FILE" | cut -d' ' -f1)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$LOCK_SHA" ]; then
    log "Lock file already installed."
else
    log "Installing $(grep -c '==' "$LOCK_FILE") pinned packages (several GB, mostly torch and its CUDA libraries)"
    # ${array[@]+...} keeps an empty array from tripping `set -u` on bash older than 4.4.
    python -m pip install \
        --only-binary=:all: \
        --no-binary=antlr4-python3-runtime,control \
        ${PIP_SOURCE_ARGS[@]+"${PIP_SOURCE_ARGS[@]}"} \
        -r "$LOCK_FILE"
    echo "$LOCK_SHA" >"$STAMP"
fi

# ---------------------------------------------------------------------------------------------
# 4. Third-party devkits at pinned commits (source checkouts on PYTHONPATH, not pip installs)
# ---------------------------------------------------------------------------------------------
head_of() { (cd "$1" && git rev-parse HEAD); }

clone_pinned() {
    local name="$1" url="$2" ref="$3" commit="$4"
    [ -n "$name" ] || die "clone_pinned: empty name"
    local dir="$WETE_THIRD_PARTY/$name"

    if [ -d "$dir/.git" ] && [ "$(head_of "$dir")" = "$commit" ]; then
        log "$name already at $commit"
        return 0
    fi
    if [ ! -d "$dir/.git" ]; then
        log "Cloning $name ($ref)"
        if ! git -c advice.detachedHead=false clone --quiet --branch "$ref" --depth 1 "$url" "$dir"; then
            rm -rf "$dir"
            git clone --quiet "$url" "$dir"
        fi
    fi
    if [ "$(head_of "$dir")" != "$commit" ]; then
        # The branch has moved past the pin: fetch full history and check the pin out.
        (
            cd "$dir"
            git fetch --quiet --unshallow origin 2>/dev/null || git fetch --quiet origin
            git checkout --quiet "$commit"
        ) || die "Could not check out $commit in $dir"
    fi
    [ "$(head_of "$dir")" = "$commit" ] || die "$name is at $(head_of "$dir"), expected $commit"
    log "$name at $commit"
}

command -v git >/dev/null 2>&1 || die "git is not available."
clone_pinned navsim_v1 "$NAVSIM_V1_URL" "$NAVSIM_V1_REF" "$NAVSIM_V1_COMMIT"
clone_pinned navsim_v2 "$NAVSIM_V2_URL" "$NAVSIM_V2_REF" "$NAVSIM_V2_COMMIT"
clone_pinned nuplan-devkit "$NUPLAN_DEVKIT_URL" "$NUPLAN_DEVKIT_REF" "$NUPLAN_DEVKIT_COMMIT"

# ---------------------------------------------------------------------------------------------
# 5. Verify and report
# ---------------------------------------------------------------------------------------------
wete_use_navsim v1
log "Verifying imports"
python - <<'PY'
import lzma, ssl, sys  # noqa: F401,E401

import hydra, numpy, pandas, pytorch_lightning, ray, timm, torch, torchvision  # noqa: F401,E401
import nuplan, navsim  # noqa: F401,E401
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent  # noqa: F401
from navsim.evaluate.pdm_score import pdm_score  # noqa: F401
import wete  # noqa: F401
from wete.cli import navsim_v1  # noqa: F401
from wete import analysis, collect, reliability, stats  # noqa: F401,E401
from wete.navsim import degraded_agent  # noqa: F401  (imports the v1.1 devkit's AbstractAgent)

print(f"[setup] python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")
print(f"[setup] torch {torch.__version__} (built for CUDA {torch.version.cuda}), torchvision {torchvision.__version__}")
print(f"[setup] numpy {numpy.__version__}, ray {ray.__version__}, timm {timm.__version__}, hydra {hydra.__version__}")
print(f"[setup] navsim from {navsim.__file__}")
PY

REPORT_DIR="$WETE_ROOT/env_report"
mkdir -p "$REPORT_DIR"
python -m pip freeze >"$REPORT_DIR/navsim_pip_freeze.txt"
{
    echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname)"
    echo "python: $(python --version 2>&1) at $(command -v python)"
    echo "base_python: $BASE_PYTHON"
    echo "venv: $WETE_VENV"
    echo "lock_sha256: $LOCK_SHA"
    echo "navsim_v1: $(head_of "$WETE_THIRD_PARTY/navsim_v1")"
    echo "navsim_v2: $(head_of "$WETE_THIRD_PARTY/navsim_v2")"
    echo "nuplan_devkit: $(head_of "$WETE_THIRD_PARTY/nuplan-devkit")"
} >"$REPORT_DIR/navsim_env.txt"

log "Done. Environment report: $REPORT_DIR"
log "Next: bash scripts/fetch_assets.sh --accept-licenses"
