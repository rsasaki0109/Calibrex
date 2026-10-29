# Direct metric-depth calibration protocol

Compare a bounded robust metric estimator against unchanged MI on the frozen
[analytic scene](known_geometry_joint.md), all five conditions and ± initial seeds
plus separate reference-start controls (10 perturbed / 5 drift trials). This follows
failed MI selection validation; it is adaptive development, not unseen evaluation.
Do not change scene, seeds, point populations or full-pose accuracy gates.

Metric objective: transformed LiDAR optical z minus bilinearly sampled camera z,
with radial images converted to optical z before interpolation. Use all original
fitting points, never the held-out lattice. No reference extrinsics enter fitting.
Four valid positive depth neighbors are required. Each unsupported point retains
a constant 1 m residual rather than disappearing; this prevents free deletion
but does not constitute a smooth/physical visibility model. Retain this limitation.
No occlusion filtering or residual-based population pruning.

Use SciPy TRF least_squares, two-point finite differences, soft-L1 scale .1 m,
rotation bounds ±2 degrees / translation ±1 m in the same initial-pose coordinates,
x_scale=(1,1,1,.1,.1,.1), all three tolerances 1e-8, max_nfev=100. Count EVERY residual
call, including numerical-Jacobian calls, with a hard 800-call ceiling; abort and
retain logs on exceeding it. The effective 100-function-evaluation cap is also
reported and may stop earlier. MI retains its 800-objective-call ceiling. These
objectives and per-call costs differ; this is not an isolated optimizer comparison.
Report actual calls and wall time. Support counting/holdout diagnostics are separate
from optimization calls and included in elapsed metric runtime only as implemented.

Require optimizer success and >=90% of initial supported points (>=64) in EACH
frame, plus the unchanged <0.5-degree/<0.20-m reference-scored pose gates. Retain
all failed/boundary cases; convergence is not evidence of pose correctness. No
reference-start rescue or threshold tuning. Evaluate the frozen held-out metric
validator only after fitting and never use it to select parameters or accept a
pose as ground truth. Its known false-accept behavior remains part of the evidence.

Freeze source/settings before scoring. Recompute MI and verify exact baseline
reproduction. Unit tests verify metric units, subpixel interpolation and fixed
missing-support population. A positive outcome remains limited to one analytic
scene, and depth bias/occlusion/true-start drift must be reported before deciding
whether further geometry tests or a real-data comparison is warranted.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_joint_metric.py docs/assets/joint_metric_20261029/plan.json \
  docs/assets/joint_metric_20261029/evaluation.json
```
