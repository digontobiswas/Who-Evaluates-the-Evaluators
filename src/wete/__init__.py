"""Who Evaluates the Evaluators? Audit harness for driving-policy evaluators.

Layout
------
config, provenance, cli/, navsim/  cluster jobs: config loading, NAVSIM devkit driver, reports
navsim/degraded_agent.py           Population A wrapper for NAVSIM agents (imports the devkit)
degradations                       graded degradation operators (Population A), framework-agnostic
wrappers                           stateful degrader for closed-loop evaluators
stats                              ranking statistics, bootstrap CIs, monotonicity test, MMRV
reliability                        per-scenario reliability labels, predictor, weighted score, CV
analysis                           score table -> report, LaTeX tables, figures
collect                            native evaluator outputs -> long-format score table
toysim                             toy world with PLANTED blind spots, for pipeline checks only
"""

__version__ = "0.1.0"
