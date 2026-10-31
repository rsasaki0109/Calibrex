# Fixed-budget joint multistart diagnostic

Freeze before running the unchanged [known-geometry cases](known_geometry_joint.md).
Compare the production single-start solver (800 calls maximum) against five fixed
starts with 160 calls maximum each. Both use exactly the same MI, histogram ranges,
step sizes, stopping tolerance and bounds in coordinates relative to the original
initial pose: rotation ±2 degrees per axis, translation ±1 m per axis. Multistart
uses an experimental global-coordinate pattern-search adapter; it does not shift
bounds around each restart. Actual evaluations/time are reported; equal ceilings
do not mean equal used computation. No unused budget redistribution.

Starts are zero plus tetrahedral rotation offsets (degrees): (.6,.6,.6),
(.6,-.6,-.6), (-.6,.6,-.6), (-.6,-.6,.6); all translation offsets zero. They depend
only on the provided initial pose. Select highest MI across all five outputs,
using the earlier start on ties, regardless of convergence/accuracy. A selected
unfinished run is a failed recovery even if its pose meets the accuracy gates.
Do not choose the best ground-truth pose or omit failed starts.

Use all five conditions × two perturbed seeds plus reference start: 10 recovery
trials and five separately reported drift controls, 15 baseline fits and 75
restart fits. Same <0.5-degree/<0.20-m gates, including convergence. Retain all
traces, counts, times, selected index, and outputs. Recomputed baseline must match
the previous study. No real-data rerun or default promotion unless this control
improves recovery without unacceptable drift; no tuning based on the results.
Unit controls verify budget, global bounds, best-value retention and flat-objective
termination. This compares an allocation strategy under a fixed maximum budget,
not a proof about all multistart or global optimization methods.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/joint_multistart_control.py docs/assets/joint_multistart_20261029/plan.json \
  docs/assets/joint_multistart_20261029/evaluation.json
```
