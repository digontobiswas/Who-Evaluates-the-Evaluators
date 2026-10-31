#!/bin/bash
# One-time setup for DiffusionDrive (run on a LOGIN node: it downloads).
#
#     bash scripts/setup_diffusiondrive.sh
#
# 1. Clones the official DiffusionDrive code at the pinned commit (configs/third_party.env) into
#    $WETE_THIRD_PARTY/diffusiondrive and links its agent package into the NAVSIM v1.1 devkit as
#    navsim/agents/diffusiondrive. Nothing of the devkit itself is changed: the link is an extra,
#    untracked directory, and DiffusionDrive's own navsim copy equals v1.1 apart from line endings.
# 2. Installs diffusers 0.35.2 (the only missing dependency; DDIMScheduler) into the NAVSIM and the
#    BridgeSim venvs with --no-deps, so no pinned package changes. Checked against the pins:
#    huggingface-hub 1.8.0, torch 2.0.1, Python 3.9.
# 3. Downloads the checkpoint, the 20 trajectory anchors and the authors' per-token navtest scores
#    into $WETE_CKPT_ROOT/diffusiondrive.
# 4. Builds the model offline in both venvs and loads the checkpoint (the same calls the jobs make).
# Re-running skips what is already there.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"
# shellcheck source=configs/third_party.env
source "$WETE_REPO/configs/third_party.env"

log() { echo "[setup-dd] $*"; }
DD="$WETE_THIRD_PARTY/diffusiondrive"
CK="$WETE_CKPT_ROOT/diffusiondrive"
RELEASE="https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file"
HF="https://huggingface.co/hustvl/DiffusionDrive/resolve/main"

# ---- 1. code -----------------------------------------------------------------------------------
if [ ! -d "$DD/.git" ]; then
    log "cloning $DIFFUSIONDRIVE_URL"
    git clone "$DIFFUSIONDRIVE_URL" "$DD"
fi
(cd "$DD" && git fetch -q origin && git checkout -q "$DIFFUSIONDRIVE_COMMIT")
log "code at $(cd "$DD" && git rev-parse HEAD)"
LINK="$WETE_THIRD_PARTY/navsim_v1/navsim/agents/diffusiondrive"
if [ -L "$LINK" ] || [ -e "$LINK" ]; then
    log "agent link exists: $LINK -> $(readlink "$LINK")"
else
    ln -s "$DD/navsim/agents/diffusiondrive" "$LINK"
    log "linked $LINK -> $DD/navsim/agents/diffusiondrive"
fi

# ---- 2. diffusers --------------------------------------------------------------------------------
for venv in "$WETE_VENV" "$WETE_BRIDGESIM_VENV"; do
    if "$venv/bin/python" -c "import diffusers" 2>/dev/null; then
        log "diffusers present in $venv"
    else
        log "installing diffusers 0.35.2 (--no-deps) into $venv"
        "$venv/bin/python" -m pip install --quiet --no-deps "diffusers==0.35.2"
    fi
done

# ---- 3. weights ----------------------------------------------------------------------------------
mkdir -p "$CK"
fetch() {
    # fetch <url> <file>
    if [ -s "$CK/$2" ]; then log "have $2"; return; fi
    log "downloading $2"
    curl -fL --retry 3 -o "$CK/$2.part" "$1" && mv "$CK/$2.part" "$CK/$2"
}
fetch "$HF/diffusiondrive_navsim_88p1_PDMS" diffusiondrive_navsim_88p1_PDMS
fetch "$RELEASE/kmeans_navsim_traj_20.npy" kmeans_navsim_traj_20.npy
fetch "$RELEASE/diffusiondrive_88p1_PDMS.csv" diffusiondrive_88p1_PDMS.csv
ls -l "$CK"

# ---- 4. offline build checks ---------------------------------------------------------------------
log "check 1/2: NAVSIM v1.1 devkit + DiffusionDrive agent, offline"
wete_activate
wete_use_navsim v1
HF_HUB_OFFLINE=1 CK="$CK" python - <<'PY'
import os
import torch
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from wete.navsim.degraded_agent import DegradedAgent

ck = os.environ["CK"]
cfg = TransfuserConfig()
cfg.plan_anchor_path = f"{ck}/kmeans_navsim_traj_20.npy"
agent = TransfuserAgent(cfg, lr=6e-4, checkpoint_path=f"{ck}/diffusiondrive_navsim_88p1_PDMS")
agent.initialize()
wrapped = DegradedAgent(agent, "none", 0, used_cameras=["cam_l0", "cam_f0", "cam_r0"])
n = sum(p.numel() for p in agent.parameters())
print(f"[setup-dd] NAVSIM: DiffusionDrive built and loaded, {n / 1e6:.1f} M parameters; wrapper ok")
PY

log "check 2/2: BridgeSim adapter, offline"
wete_use_bridgesim
HF_HUB_OFFLINE=1 CK="$CK" python - <<'PY'
import os
from bridgesim.evaluation.models.diffusiondrive_adapter import DiffusionDriveAdapter

ck = os.environ["CK"]
adapter = DiffusionDriveAdapter(checkpoint_path=f"{ck}/diffusiondrive_navsim_88p1_PDMS",
                                plan_anchor_path=f"{ck}/kmeans_navsim_traj_20.npy")
adapter.device = "cpu"   # the adapter defaults to cuda; login nodes have no GPU (jobs run on GPU)
adapter.load_model()     # prints the number of missing / unexpected checkpoint keys, if any
print("[setup-dd] BridgeSim: DiffusionDrive adapter loaded")
PY
log "Done. Submit with: bash scripts/submit_diffusiondrive.sh --dry-run"
