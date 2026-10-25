#!/bin/bash
# Download NAVSIM data and one pretrained checkpoint. Run on the LOGIN node, after
# scripts/setup_env.sh, ideally inside `screen` because the full sensor set takes hours:
#
#     bash scripts/fetch_assets.sh --accept-licenses                       # smoke-test set
#     bash scripts/fetch_assets.sh --accept-licenses --what sensors --sensor-shards all
#
# What it can fetch (--what, comma-separated, default: all five):
#   maps        nuPlan maps v1.1 archive (needed for metric caching)
#   logs        OpenScene "test" split annotations (documented size about 1 GB)
#   sensors     OpenScene "test" split cameras + LiDAR, in 32 shards each
#               (documented total 217 GB; navtest is a filter over this split)
#   checkpoint  one NAVSIM baseline checkpoint from Hugging Face
#   backbone    the ImageNet ResNet weights that timm downloads when the TransFuser agent is
#               constructed; cached now because compute nodes have no internet
#
# Options:
#   --accept-licenses        Required for maps/logs/sensors. Confirms you have read the nuPlan
#                            dataset licence and the OpenScene licence (CC BY-NC-SA 4.0).
#   --sensor-shards SPEC     Which of the 32 shards to fetch: "0", "0-3", "0,5,9" or "all".
#                            Default "0", enough for the smoke test.
#   --checkpoint FAMILY/SEED One of transfuser, ltf, ego_status_mlp and a seed 0-2.
#                            Default transfuser/0.
#   --keep-archives          Do not delete archives after extraction.
#
# Re-running resumes: finished archives are recorded and skipped, partial downloads continue.
# Archives are extracted once under $OPENSCENE_DATA_ROOT/_extracted and the directory layout
# the devkit expects is created with symlinks, so nothing is copied twice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

OPENSCENE_URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
MAPS_URL="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
BASELINES_URL="https://huggingface.co/autonomousvision/navsim_baselines/resolve/main"
NUPLAN_LICENSE_URL="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE"
NUM_SHARDS=32

# File sizes in bytes as listed by the Hugging Face API for autonomousvision/navsim_baselines.
declare -A CHECKPOINT_BYTES=(
    ["transfuser/0"]=672447984 ["transfuser/1"]=672447984 ["transfuser/2"]=672447984
    ["ltf/0"]=673235500 ["ltf/1"]=673235500 ["ltf/2"]=673235500
    ["ego_status_mlp/0"]=6518594 ["ego_status_mlp/1"]=6518530 ["ego_status_mlp/2"]=6518530
)

WHAT="maps,logs,sensors,checkpoint,backbone"
SHARD_SPEC="0"
CHECKPOINT="transfuser/0"
KEEP_ARCHIVES=0
ACCEPT_LICENSES=0

log() { echo "[fetch] $*"; }
die() {
    echo "[fetch] ERROR: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --what)
            WHAT="${2:?--what needs a list}"
            shift 2
            ;;
        --sensor-shards)
            SHARD_SPEC="${2:?--sensor-shards needs a spec}"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="${2:?--checkpoint needs FAMILY/SEED}"
            shift 2
            ;;
        --keep-archives)
            KEEP_ARCHIVES=1
            shift
            ;;
        --accept-licenses)
            ACCEPT_LICENSES=1
            shift
            ;;
        -h | --help)
            sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
done

wants() {
    case ",$WHAT," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

for item in ${WHAT//,/ }; do
    case "$item" in
        maps | logs | sensors | checkpoint | backbone) ;;
        *) die "Unknown item in --what: $item" ;;
    esac
done

if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "WARNING: running inside a SLURM job. Compute nodes have no internet; downloads will fail."
fi

DOWNLOADS="$OPENSCENE_DATA_ROOT/_downloads"
EXTRACTED="$OPENSCENE_DATA_ROOT/_extracted"
STATE="$OPENSCENE_DATA_ROOT/_state"
mkdir -p "$DOWNLOADS" "$EXTRACTED" "$STATE"

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
fetch() {
    # Resumable download to $2. A ".part" suffix marks an unfinished file.
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget --continue --tries=5 --timeout=60 --progress=dot:giga -O "$dest.part" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --connect-timeout 30 -C - -o "$dest.part" "$url"
    else
        die "Neither wget nor curl is available."
    fi
    mv "$dest.part" "$dest"
}

fetch_and_extract() {
    # Download one archive and unpack it under $EXTRACTED, once.
    local name="$1" url="$2"
    local marker="$STATE/$name.done"
    if [ -f "$marker" ]; then
        log "$name: already extracted"
        return 0
    fi
    log "$name: downloading"
    fetch "$url" "$DOWNLOADS/$name"
    log "$name: extracting"
    tar -xzf "$DOWNLOADS/$name" -C "$EXTRACTED"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$marker"
    if [ "$KEEP_ARCHIVES" = 0 ]; then
        rm -f "$DOWNLOADS/$name"
    fi
}

link_split() {
    # Expose $EXTRACTED/openscene-v1.1/<kind>[/test] as <dest>. The archives may or may not
    # contain a per-split "test" level; both layouts are handled.
    local kind="$1" dest="$2"
    local src="$EXTRACTED/openscene-v1.1/$kind"
    [ -d "$src" ] || die "Expected directory not found after extraction: $src"
    if [ -d "$src/test" ]; then
        src="$src/test"
    fi
    mkdir -p "$(dirname "$dest")"
    if [ -L "$dest" ]; then
        rm "$dest"
    elif [ -e "$dest" ]; then
        die "$dest exists and is not a symlink; move it away first."
    fi
    ln -s "$src" "$dest"
    log "$dest -> $src"
}

parse_shards() {
    # Expand "all", "3", "0-3" or "0,5,9" into a validated list of shard indices.
    local spec="$1" part lo hi i
    if [ "$spec" = "all" ]; then
        seq 0 $((NUM_SHARDS - 1))
        return 0
    fi
    for part in ${spec//,/ }; do
        if [[ "$part" =~ ^[0-9]+$ ]]; then
            lo="$part"
            hi="$part"
        elif [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo="${BASH_REMATCH[1]}"
            hi="${BASH_REMATCH[2]}"
        else
            die "Bad shard spec '$part'. Use e.g. 0, 0-3, 0,5,9 or all."
        fi
        if [ "$lo" -gt "$hi" ] || [ "$hi" -ge "$NUM_SHARDS" ]; then
            die "Shard range '$part' is outside 0-$((NUM_SHARDS - 1))."
        fi
        for ((i = lo; i <= hi; i++)); do
            echo "$i"
        done
    done
}

require_license_ack() {
    [ "$ACCEPT_LICENSES" = 1 ] && return 0
    cat >&2 <<EOF
[fetch] The data you are about to download is licensed for non-commercial use.
[fetch]   nuPlan dataset licence: $NUPLAN_LICENSE_URL
[fetch]   OpenScene (redistribution of nuPlan) on Hugging Face: tagged CC BY-NC-SA 4.0
[fetch]     https://huggingface.co/datasets/OpenDriveLab/OpenScene
[fetch] Read them, then re-run with --accept-licenses.
EOF
    exit 2
}

# ---------------------------------------------------------------------------------------------
# plan and disk check
# ---------------------------------------------------------------------------------------------
SHARDS=()
if wants sensors; then
    # Command substitution, so that a bad spec stops the script rather than a subshell.
    SHARD_LIST="$(parse_shards "$SHARD_SPEC")" || exit 1
    for shard in $SHARD_LIST; do
        SHARDS+=("$shard")
    done
    [ "${#SHARDS[@]}" -gt 0 ] || die "No sensor shards selected."
fi

if wants maps || wants logs || wants sensors; then
    require_license_ack
fi

FREE_KB="$(df -Pk "$OPENSCENE_DATA_ROOT" | awk 'NR==2 {print $4}')"
log "Free space reported by df for $OPENSCENE_DATA_ROOT: $((FREE_KB / 1024 / 1024)) GB (quota may be lower; check with myquota)"
if wants sensors; then
    # 217 GB documented for 32 shard pairs; an archive and its extracted copy coexist briefly.
    log "Sensor shards selected: ${#SHARDS[@]} of $NUM_SHARDS, roughly $((${#SHARDS[@]} * 217 / NUM_SHARDS)) GB extracted (estimate from the documented total)"
fi

# ---------------------------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------------------------
if wants maps; then
    if [ -f "$STATE/maps.done" ]; then
        log "maps: already extracted"
    else
        log "maps: downloading"
        fetch "$MAPS_URL" "$DOWNLOADS/nuplan-maps-v1.1.zip"
        log "maps: extracting"
        mkdir -p "$EXTRACTED/maps"
        if command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$DOWNLOADS/nuplan-maps-v1.1.zip" -d "$EXTRACTED/maps"
        else
            wete_activate
            python -m zipfile -e "$DOWNLOADS/nuplan-maps-v1.1.zip" "$EXTRACTED/maps"
        fi
        date -u +%Y-%m-%dT%H:%M:%SZ >"$STATE/maps.done"
        [ "$KEEP_ARCHIVES" = 1 ] || rm -f "$DOWNLOADS/nuplan-maps-v1.1.zip"
    fi
    # The archive unpacks to a directory named after the map version the devkit asks for.
    MAPS_SRC="$EXTRACTED/maps/$NUPLAN_MAP_VERSION"
    [ -d "$MAPS_SRC" ] || die "Expected $MAPS_SRC after unpacking the maps; found: $(ls "$EXTRACTED/maps")"
    if [ -L "$NUPLAN_MAPS_ROOT" ]; then
        rm "$NUPLAN_MAPS_ROOT"
    elif [ -e "$NUPLAN_MAPS_ROOT" ]; then
        die "$NUPLAN_MAPS_ROOT exists and is not a symlink; move it away first."
    fi
    ln -s "$MAPS_SRC" "$NUPLAN_MAPS_ROOT"
    log "$NUPLAN_MAPS_ROOT -> $MAPS_SRC"
fi

# ---------------------------------------------------------------------------------------------
# logs (annotations) of the OpenScene test split
# ---------------------------------------------------------------------------------------------
if wants logs; then
    fetch_and_extract "openscene_metadata_test.tgz" "$OPENSCENE_URL/openscene_metadata_test.tgz"
    link_split "meta_datas" "$OPENSCENE_DATA_ROOT/navsim_logs/test"
    NUM_LOGS="$(find -L "$OPENSCENE_DATA_ROOT/navsim_logs/test" -maxdepth 1 -name '*.pkl' | wc -l)"
    [ "$NUM_LOGS" -gt 0 ] || die "No .pkl log files under $OPENSCENE_DATA_ROOT/navsim_logs/test; archive layout is not what the devkit documents."
    log "logs: $NUM_LOGS log files"
fi

# ---------------------------------------------------------------------------------------------
# sensors of the OpenScene test split
# ---------------------------------------------------------------------------------------------
if wants sensors; then
    for shard in "${SHARDS[@]}"; do
        fetch_and_extract "openscene_sensor_test_camera_${shard}.tgz" \
            "$OPENSCENE_URL/openscene_sensor_test_camera/openscene_sensor_test_camera_${shard}.tgz"
        fetch_and_extract "openscene_sensor_test_lidar_${shard}.tgz" \
            "$OPENSCENE_URL/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${shard}.tgz"
    done
    link_split "sensor_blobs" "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
    DONE_CAMERA="$(find "$STATE" -maxdepth 1 -name 'openscene_sensor_test_camera_*.done' | wc -l)"
    DONE_LIDAR="$(find "$STATE" -maxdepth 1 -name 'openscene_sensor_test_lidar_*.done' | wc -l)"
    log "sensors: $DONE_CAMERA/$NUM_SHARDS camera shards and $DONE_LIDAR/$NUM_SHARDS LiDAR shards on disk"
    if [ "$DONE_CAMERA" -lt "$NUM_SHARDS" ] || [ "$DONE_LIDAR" -lt "$NUM_SHARDS" ]; then
        log "sensors: enough for the smoke test only. The full navtest run needs --sensor-shards all."
    fi
fi

# ---------------------------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------------------------
if wants checkpoint; then
    EXPECTED_BYTES="${CHECKPOINT_BYTES[$CHECKPOINT]:-}"
    [ -n "$EXPECTED_BYTES" ] || die "Unknown checkpoint '$CHECKPOINT'. Known: ${!CHECKPOINT_BYTES[*]}"
    FAMILY="${CHECKPOINT%/*}"
    SEED="${CHECKPOINT#*/}"
    CKPT_DIR="$WETE_CKPT_ROOT/navsim_baselines/$FAMILY"
    CKPT_FILE="$CKPT_DIR/${FAMILY}_seed_${SEED}.ckpt"
    mkdir -p "$CKPT_DIR"
    log "checkpoint: $CKPT_FILE"
    fetch "$BASELINES_URL/$FAMILY/${FAMILY}_seed_${SEED}.ckpt" "$CKPT_FILE"
    ACTUAL_BYTES="$(stat -c %s "$CKPT_FILE")"
    [ "$ACTUAL_BYTES" = "$EXPECTED_BYTES" ] || die \
        "$CKPT_FILE is $ACTUAL_BYTES bytes, expected $EXPECTED_BYTES. Delete it and re-run."
    SUMS="$WETE_CKPT_ROOT/navsim_baselines/SHA256SUMS"
    if ! grep -q " $FAMILY/${FAMILY}_seed_${SEED}.ckpt\$" "$SUMS" 2>/dev/null; then
        (cd "$WETE_CKPT_ROOT/navsim_baselines" && sha256sum "$FAMILY/${FAMILY}_seed_${SEED}.ckpt" >>"$SUMS")
    fi
    log "checkpoint: size verified, sha256 recorded in $SUMS"
fi

# ---------------------------------------------------------------------------------------------
# backbone weights for offline model construction
# ---------------------------------------------------------------------------------------------
if wants backbone; then
    wete_activate
    wete_use_navsim v1
    log "backbone: caching the timm weights the TransFuser backbone requests into $HF_HOME"
    python - <<'PY'
import timm
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

architecture = TransfuserConfig().image_architecture
timm.create_model(architecture, pretrained=True, features_only=True)
print(f"[fetch] backbone: downloaded timm weights for '{architecture}'")
PY
    # Prove that the same call succeeds with the network forbidden, as it must on a compute node.
    HF_HUB_OFFLINE=1 python - <<'PY'
import timm
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

architecture = TransfuserConfig().image_architecture
timm.create_model(architecture, pretrained=True, features_only=True)
print(f"[fetch] backbone: offline load of '{architecture}' works")
PY
fi

log "Done."
log "Smoke test:  bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/stage1_pdms_transfuser_smoke.yaml"
