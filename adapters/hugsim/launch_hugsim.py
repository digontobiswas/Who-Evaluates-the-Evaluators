"""Run one HUGSIM closed-loop episode with the wete relay in front of the policy.

HUGSIM's closed_loop.py picks the policy launch shell from cfg.base.<ad>_path
(ad in {uniad, vad, ltf}) and writes to cfg.base.output_dir + ad. We write a
copy of the base config with <ad>_path -> relay.sh and output_dir -> --out,
then call closed_loop.py unchanged. The real policy shell is passed to the
relay through WETE_BASE_AD_SHELL (set by scripts/make_jobs.py).

Output: <out>/<ad>/<scene>_<mode>/{eval.json, wete_trace.jsonl, ...}
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hugsim-root", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--ad", default="ltf", choices=["uniad", "vad", "ltf"])
    ap.add_argument("--ad-cuda", default="0")
    ap.add_argument("--relay", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for v in ("WETE_BASE_AD_SHELL", "WETE_SPEC", "WETE_SCENARIO"):
        if not os.environ.get(v):
            sys.exit(f"missing environment variable {v}")
    base = OmegaConf.load(a.base)
    base[f"{a.ad}_path"] = str(Path(a.relay).resolve())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base["output_dir"] = str(out) + os.sep          # closed_loop.py appends the ad name
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out) as f:
        OmegaConf.save(base, f.name)
        base_copy = f.name
    cmd = [sys.executable, "closed_loop.py", "--scenario_path", a.scenario, "--base_path", base_copy,
           "--camera_path", a.camera, "--kinematic_path", a.kinematic, "--ad", a.ad,
           "--ad_cuda", a.ad_cuda]
    return subprocess.call(cmd, cwd=a.hugsim_root)


if __name__ == "__main__":
    sys.exit(main())
