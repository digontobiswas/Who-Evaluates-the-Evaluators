#!/bin/bash
# Collect the facts about the cluster that the project must not guess. Run on the LOGIN node:
#
#     bash scripts/hpc_probe.sh 2>&1 | tee hpc_probe_login.txt
#
# Read-only: it submits nothing, installs nothing and changes no file. Paste the output back.
# The compute-node half is jobs/probe_node.sbatch.

section() {
    echo
    echo "===== $* ====="
}

try() {
    echo "\$ $*"
    "$@" 2>&1 || echo "(exit code $?)"
}

http_status() {
    # Print only the HTTP status line; 10 s limit so a blocked host does not hang the probe.
    local url="$1" out code
    echo "\$ curl -sI --max-time 10 $url | head -1"
    out="$(curl -sI --max-time 10 "$url" 2>&1)"
    code=$?
    echo "$out" | head -1
    [ "$code" = 0 ] || echo "(curl exit code $code)"
}

section "identity and OS"
try hostname
try whoami
try date
try uname -r
try cat /etc/os-release
echo "\$ ldd --version | head -1"
ldd --version 2>&1 | head -1
echo "HOME=$HOME"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "PWD=$PWD"

section "SLURM partitions"
try sinfo --version
# partition, availability, time limit, node count, CPUs per node, memory, generic resources
try sinfo -o "%.14P %.6a %.12l %.6D %.5c %.9m %.25G %N"
try sinfo -s
try scontrol show partition

section "SLURM limits for this user"
try sacctmgr -nP show assoc "user=$USER" format=cluster,account,partition,qos,maxjobs,maxsubmit,maxwall,grptres
try sacctmgr -nP show qos format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu
try squeue -u "$USER"
echo "\$ squeue -h -o %P | sort | uniq -c   (jobs per partition right now)"
squeue -h -o "%P" 2>/dev/null | sort | uniq -c

section "storage and quota"
try myquota
try df -h "$HOME"
if [ -n "${SCRATCH:-}" ]; then
    try df -h "$SCRATCH"
    try lfs quota -hu "$USER" "$SCRATCH"
fi
try lfs quota -hu "$USER" "$HOME"
if [ -d "$HOME/nuscenes-trainval" ]; then
    echo "nuScenes found at $HOME/nuscenes-trainval:"
    try ls "$HOME/nuscenes-trainval"
fi

section "software on the login node"
for tool in python python3 python3.9 python3.10 git curl wget rsync unzip tar zsh screen tmux gcc cmake nvidia-smi; do
    path="$(command -v "$tool" 2>/dev/null || true)"
    if [ -n "$path" ]; then
        echo "$tool -> $path"
    else
        echo "$tool -> (not found)"
    fi
done
try python3 --version
try git --version
echo "\$ curl --version | head -1"
curl --version 2>&1 | head -1
try python3 -c "import ssl; print('ssl ok:', ssl.OPENSSL_VERSION)"
try python3 -c "import lzma, bz2, sqlite3, ctypes; print('lzma bz2 sqlite3 ctypes ok')"

section "modules"
if command -v module >/dev/null 2>&1; then
    echo "\$ module avail 2>&1 | grep -i -E 'python|conda|cuda|cudnn|gcc|cmake|git|mesa|egl'"
    module avail 2>&1 | grep -i -E "python|conda|cuda|cudnn|gcc|cmake|git|mesa|egl" || echo "(no match)"
    try module list
else
    echo "(module command not available in this shell)"
fi

section "outbound network from the login node"
http_status https://github.com
http_status https://github.com/astral-sh/python-build-standalone/releases/download/20251031/SHA256SUMS
http_status https://pypi.org/simple/pip/
http_status https://files.pythonhosted.org
http_status https://huggingface.co/api/models/autonomousvision/navsim_baselines
http_status https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE
echo "proxy variables: http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"

section "done"
echo "Next: submit jobs/probe_node.sbatch to the GPU partition named above (see README)."
