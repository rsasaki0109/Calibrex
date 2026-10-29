# Learned stereo calibration comparison

## Frozen protocol

Compare pinned UniMatch SceneFlow with SGBM on the same six spent central frames
and twelve seeds as the preceding ZNCC comparison. This is development evidence,
not held-out validation or SOTA. Model provenance, training exposure and license
limitations are in [runtime documentation](unimatch_runtime.md).

[Plan](../assets/learned_stereo_20261029/plan.json) fixes source hashes and cases.
Use the existing single-frame Borer six-DoF D2D solver and unchanged defaults.
Recovery requires convergence, rotation error <0.5 degrees and translation error
<0.20 m, evaluated against the reference only after optimization. Report all
24 fits, including numerical convergence without accuracy. No tuning or filtering
of cases after scoring. Runtime failures stop the run with logs retained; no
smaller model/resolution retry and no dropped denominator.

UniMatch consumes RGB, SGBM grayscale, from identical image pairs. Each method
uses identical camera re-rectification. Native resolution, four CPU threads,
300-second inference budget per frame and checkpoint/configuration are unchanged
from the smoke. Rerun all six inferences, including the smoke frame. Convert
rectified disparity with (P1[0,3]-P2[0,3])/d, check finite positive disparity,
left/right remap support and projected right-image bounds, retain 0.1–80 m, and
restore depth to original camera rays. No confidence threshold or extra learned
mask. LiDAR/reference extrinsics do not enter inference. Synthetic metric-depth,
missing-support and padding tests must pass before execution.

Recompute SGBM rather than relying on historical scores. Report full-pose gates,
rotation/translation separately, depth coverage and paired projected-point
residuals on the exact common valid pixel set. The comparison denominator is
single-frame re-rectified SGBM (previously 2/12), not three-frame joint D2D (5/12).

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/compare_learned_stereo.py \
  docs/assets/learned_stereo_20261029/plan.json \
  /workspace/work/learned-stereo-comparison-20261029 \
  docs/assets/learned_stereo_20261029/evaluation.json
```
