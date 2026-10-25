#!/bin/bash
# Submit a job script with the partition, time and core count taken from
# configs/hpc/cluster.env, so that no job file contains a guessed cluster setting.
#
#     bash scripts/submit.sh [--dry-run] <cpu|gpu> <job.sbatch> [KEY=VALUE ...] [-- <extra sbatch options>]
#
# Examples:
#     bash scripts/submit.sh cpu jobs/navsim_v1_pdms.sbatch CONFIG=configs/experiments/stage1_pdms_transfuser_smoke.yaml
#     bash scripts/submit.sh --dry-run cpu jobs/navsim_v1_pdms.sbatch CONFIG=... -- --time=00:30:00
#
# KEY=VALUE pairs are exported into the job's environment. The wrapper prints the exact
# sbatch command it runs. It changes directory to $WETE_ROOT/slurm_logs first, because this
# cluster requires jobs to be submitted from scratch, and writes the job's output there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/env.sh"

die() {
    echo "[submit] ERROR: $*" >&2
    exit 1
}

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

KIND="${1:-}"
JOB="${2:-}"
[ -n "$KIND" ] && [ -n "$JOB" ] || die "Usage: submit.sh [--dry-run] <cpu|gpu> <job.sbatch> [KEY=VALUE ...] [-- sbatch options]"
shift 2

case "$JOB" in
    /*) JOB_PATH="$JOB" ;;
    *) JOB_PATH="$WETE_REPO/$JOB" ;;
esac
[ -f "$JOB_PATH" ] || die "Job script not found: $JOB_PATH"

need() {
    # need VAR "how to find the value"
    local name="$1"
    [ -n "${!name:-}" ] || die "$name is empty in $WETE_CLUSTER_ENV. $2"
}

SBATCH_ARGS=()
case "$KIND" in
    cpu)
        need WETE_CPU_PARTITION "Take the partition name from: sinfo"
        need WETE_CPU_CPUS_PER_TASK "Choose a core count the partition allows."
        need WETE_CPU_TIME "Choose a wall time within the partition limit shown by: sinfo"
        SBATCH_ARGS+=("--partition=$WETE_CPU_PARTITION" "--cpus-per-task=$WETE_CPU_CPUS_PER_TASK" "--time=$WETE_CPU_TIME")
        [ -z "${WETE_CPU_MEM:-}" ] || SBATCH_ARGS+=("--mem=$WETE_CPU_MEM")
        ;;
    gpu)
        need WETE_GPU_PARTITION "Take the partition name from: sinfo"
        need WETE_GPU_GRES "Take the generic resource string from: sinfo -o '%P %G'"
        need WETE_GPU_CPUS_PER_TASK "Choose a core count the partition allows."
        need WETE_GPU_TIME "Choose a wall time within the partition limit shown by: sinfo"
        SBATCH_ARGS+=("--partition=$WETE_GPU_PARTITION" "--gres=$WETE_GPU_GRES" "--cpus-per-task=$WETE_GPU_CPUS_PER_TASK" "--time=$WETE_GPU_TIME")
        [ -z "${WETE_GPU_MEM:-}" ] || SBATCH_ARGS+=("--mem=$WETE_GPU_MEM")
        ;;
    *) die "First argument must be cpu or gpu, got '$KIND'." ;;
esac
[ -z "${WETE_ACCOUNT:-}" ] || SBATCH_ARGS+=("--account=$WETE_ACCOUNT")
[ -z "${WETE_QOS:-}" ] || SBATCH_ARGS+=("--qos=$WETE_QOS")
SBATCH_ARGS+=("--output=$WETE_SLURM_LOGS/%x-%j.out")

# KEY=VALUE pairs for the job, then anything after "--" for sbatch itself (later options win).
JOB_VARS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --)
            shift
            SBATCH_ARGS+=("$@")
            break
            ;;
        *=*)
            JOB_VARS+=("$1")
            shift
            ;;
        *) die "Expected KEY=VALUE or --, got '$1'." ;;
    esac
done

mkdir -p "$WETE_SLURM_LOGS"
cd "$WETE_SLURM_LOGS"

# The job's variables travel in the environment (sbatch exports it by default). Passing them
# through --export would break on values that contain commas, such as STAGES=cache,score.
echo "[submit] cd $WETE_SLURM_LOGS"
printf '[submit] env WETE_REPO=%q' "$WETE_REPO"
for pair in ${JOB_VARS[@]+"${JOB_VARS[@]}"}; do
    printf ' %q' "$pair"
done
printf ' sbatch'
for arg in "${SBATCH_ARGS[@]}"; do
    printf ' %q' "$arg"
done
printf ' %q\n' "$JOB_PATH"

if [ "$DRY_RUN" = 1 ]; then
    echo "[submit] dry run: nothing submitted."
    exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch not found. Run this on the cluster login node."
env "WETE_REPO=$WETE_REPO" ${JOB_VARS[@]+"${JOB_VARS[@]}"} sbatch "${SBATCH_ARGS[@]}" "$JOB_PATH"
echo "[submit] output will be written to $WETE_SLURM_LOGS/<job-name>-<job-id>.out"
echo "[submit] check the queue with: squeue -u $USER"
