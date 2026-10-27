"""Run BridgeSim's unified_evaluator unmodified, with the model adapter wrapped for Population A.

    python -m wete.cli.bridgesim_degraded_eval --degradation camera_dropout --level 2 \
        [--value 0.3] <every unified_evaluator.py argument>

Our three options are taken off the command line; everything else goes to BridgeSim's own
``main()``. The only change to BridgeSim's behaviour is that ``create_model_adapter`` returns the
real adapter inside :class:`wete.closed_loop.degraded_adapter.DegradedAdapter`. With
``--degradation none`` the wrapper passes everything through unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from wete import degradations as D
from wete.closed_loop import degraded_adapter


def parse_ours(argv: List[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--degradation", default="none")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--value", type=float, default=None)
    ours, rest = parser.parse_known_args(argv)
    spec = D.DegradationSpec(ours.degradation, ours.level, ours.value)
    if spec.degradation != "none":
        D.level_param(spec.degradation, spec.level)  # validates name and level
    if spec.degradation not in degraded_adapter.SUPPORTED:
        raise SystemExit(f"[wete] {spec.degradation} is not available in BridgeSim")
    return spec, rest


def main(argv: Optional[List[str]] = None) -> int:
    spec, rest = parse_ours(list(sys.argv[1:] if argv is None else argv))

    from bridgesim.evaluation import unified_evaluator as U  # needs the BridgeSim environment

    original = U.create_model_adapter

    def create_wrapped(args):
        base = original(args)
        return degraded_adapter.wrap(
            base,
            spec,
            scenario_id=Path(args.scenario_path).name,
            sim_dt=float(args.sim_dt),
            replan_interval_s=float(args.replan_rate) * float(args.sim_dt),
            plan_log=Path(args.output_dir) / "wete_plans.jsonl",
        )

    U.create_model_adapter = create_wrapped
    sys.argv = ["unified_evaluator.py", *rest]
    print(f"[wete] BridgeSim with degradation {spec.policy_suffix} (param {spec.param})", flush=True)
    U.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
