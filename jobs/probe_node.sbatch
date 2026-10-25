#!/bin/bash
#SBATCH --job-name=wete-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# Collect the facts about a COMPUTE node that the project must not guess. Partition, GPU
# request and time are given on the command line because they are exactly what we are trying
# to learn, for example (from a directory on scratch):
#
#     sbatch --partition=<GPU_PARTITION> --gres=<GRES> --time=00:05:00 \
#            --output=wete-probe-%j.out /path/to/repo/jobs/probe_node.sbatch
#
# Read-only. Paste the .out file back.

section() {
    echo
    echo "===== $* ====="
}

try() {
    echo "\$ $*"
    "$@" 2>&1 || echo "(exit code $?)"
}

section "node"
try hostname
try date
try uname -r
echo "\$ ldd --version | head -1"
ldd --version 2>&1 | head -1
try nproc
try free -g
echo "\$ lscpu | grep -E 'Model name|Socket|Core|Thread|^CPU\\(s\\)'"
lscpu 2>/dev/null | grep -E "Model name|Socket|Core|Thread|^CPU\(s\)"
echo "\$ grep -o -w -E 'avx|avx2|avx512f' /proc/cpuinfo | sort -u"
grep -o -w -E "avx|avx2|avx512f" /proc/cpuinfo | sort -u | tr '\n' ' '
echo

section "SLURM allocation"
env | grep -E "^SLURM_(JOB_ID|JOB_PARTITION|CPUS_PER_TASK|JOB_CPUS_PER_NODE|MEM_PER_NODE|JOB_GPUS|GPUS|SUBMIT_DIR|JOB_NODELIST)=" | sort
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
try scontrol show job "${SLURM_JOB_ID:-0}"

section "GPU"
try nvidia-smi
try nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv

section "local storage and limits"
echo "TMPDIR=${TMPDIR:-<unset>}"
try df -h /tmp /dev/shm
echo "\$ ulimit -a"
ulimit -a

section "outbound network from the compute node (expected: none)"
for url in https://github.com https://pypi.org/simple/pip/ https://huggingface.co; do
    echo "\$ curl -sI --max-time 5 $url | head -1"
    out="$(curl -sI --max-time 5 "$url" 2>&1)"
    code=$?
    echo "$out" | head -1
    [ "$code" = 0 ] || echo "(curl exit code $code)"
done

section "graphics libraries (only matters for a rendering-based simulator fallback)"
echo "\$ find /usr/lib64 -maxdepth 1 -iname 'libEGL*' (and libGL, libOSMesa, libnvidia-egl, libGLX_nvidia)"
found="$(find /usr/lib64 -maxdepth 1 \( -iname 'libEGL*' -o -iname 'libGL.*' -o -iname 'libOSMesa*' \
    -o -iname 'libnvidia-egl*' -o -iname 'libGLX_nvidia*' \) 2>/dev/null | sort)"
echo "${found:-(none found)}"

section "software"
for tool in python3 git zsh gcc nvcc; do
    path="$(command -v "$tool" 2>/dev/null || true)"
    echo "$tool -> ${path:-(not found)}"
done
if command -v module >/dev/null 2>&1; then
    try module list
fi

section "done"
