#!/usr/bin/env zsh
# Drop-in replacement for a HUGSIM AD launch shell (set base.ltf_path / uniad_path
# in the HUGSIM base config to this file). HUGSIM calls: zsh relay.sh <cuda_id> <output>
#
# Required environment variables:
#   WETE_BASE_AD_SHELL  the policy's real launch shell (e.g. HUGSIM's LTF client shell)
#   WETE_SPEC           degradation spec, e.g. "camera_dropout@2" or "none"
#   WETE_SCENARIO       scene id (used for common random numbers)
# Optional:
#   WETE_VALUE          calibrated parameter value for this level
#   WETE_PYTHON         python executable with wete + wete_hugsim importable
set -euo pipefail
CUDA_ID=$1
OUTER=$2
INNER=$OUTER/wete_inner
mkdir -p $INNER
PY=${WETE_PYTHON:-python}

# create inner FIFOs before the policy starts, in case it does not create them
[[ -p $INNER/obs_pipe ]]  || mkfifo $INNER/obs_pipe
[[ -p $INNER/plan_pipe ]] || mkfifo $INNER/plan_pipe

zsh $WETE_BASE_AD_SHELL $CUDA_ID $INNER > $OUTER/wete_policy.log 2>&1 &
POLICY_PID=$!

VALUE_ARG=()
[[ -n ${WETE_VALUE:-} ]] && VALUE_ARG=(--value $WETE_VALUE)
$PY -m wete_hugsim.relay --outer $OUTER --inner $INNER --spec $WETE_SPEC \
    --scenario $WETE_SCENARIO $VALUE_ARG > $OUTER/wete_relay.log 2>&1
wait $POLICY_PID
