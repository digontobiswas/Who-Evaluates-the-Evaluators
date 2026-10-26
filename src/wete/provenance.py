"""Record where a number came from: code versions, inputs, machine."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def git_commit(directory: Path, fallback_env: Optional[str] = None) -> Optional[str]:
    """Commit hash of a checkout, with ``-dirty`` appended if it has local changes.

    Uses ``cwd`` rather than ``git -C``: the git shipped with CentOS 7 predates ``-C``.
    On PARAM Shakti git works on the login nodes but not inside jobs, so ``scripts/submit.sh``
    records the commits at submission time; ``fallback_env`` names the variable holding that
    value, used (with a ``submit:`` prefix) when git fails here.
    """
    fallback = os.environ.get(fallback_env) if fallback_env else None
    fallback = f"submit:{fallback}" if fallback else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(directory),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(directory),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback
    return f"{commit}-dirty" if status else commit


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def machine() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
