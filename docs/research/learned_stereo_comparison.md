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

## Results

Protocol/code frozen at `2f73c756`. All six inferences and all 24 calibration
fits completed. Recomputed SGBM outputs exactly match the earlier comparison.

| Case | SGBM deg | SGBM m | Pass | UniMatch deg | UniMatch m | Pass |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 2011_09_26_0005-positive | 0.9060 | 0.2257 | False | 0.7960 | 0.2138 | False |
| 2011_09_26_0005-negative | 0.7027 | 0.2050 | False | 0.6824 | 0.1724 | False |
| 2011_09_28_0002-positive | 0.2715 | 0.0395 | True | 0.5520 | 0.1567 | False |
| 2011_09_28_0002-negative | 0.4826 | 0.3052 | False | 1.1305 | 0.2125 | False |
| 2011_09_26_0009-positive | 1.7845 | 0.2872 | False | 1.2462 | 0.2512 | False |
| 2011_09_26_0009-negative | 0.9000 | 0.5563 | False | 0.4377 | 0.1772 | True |
| 2011_09_26_0011-positive | 0.3269 | 0.1487 | True | 0.3098 | 0.0901 | True |
| 2011_09_26_0011-negative | 0.3706 | 0.5859 | False | 0.4435 | 0.4651 | False |
| 2011_09_26_0013-positive | 0.7705 | 0.2670 | False | 1.2336 | 0.3865 | False |
| 2011_09_26_0013-negative | 1.3022 | 0.3424 | False | 0.9576 | 0.2137 | False |
| 2011_09_26_0015-positive | 0.7243 | 0.3288 | False | 0.9565 | 0.2979 | False |
| 2011_09_26_0015-negative | 0.6583 | 0.2019 | False | 0.7170 | 0.1406 | False |

Common-point residuals use the same z-buffered LiDAR/pixel population within
each frame; they are diagnostics, not independent depth ground truth.

| Drive | Common points | SGBM median z m | UniMatch median z m | SGBM abs p90 m | UniMatch abs p90 m | SGBM pixels | UniMatch pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011_09_26_0005 | 16720 | -0.0844 | -0.2048 | 2.8967 | 2.4186 | 361931 | 439253 |
| 2011_09_28_0002 | 15623 | -0.0616 | -0.0861 | 3.5040 | 4.2118 | 322547 | 362904 |
| 2011_09_26_0009 | 14667 | -0.0697 | -0.0803 | 4.0017 | 4.3257 | 341257 | 384801 |
| 2011_09_26_0011 | 15427 | -0.1036 | -0.1008 | 3.4285 | 3.3632 | 300244 | 331763 |
| 2011_09_26_0013 | 14240 | -0.0514 | -0.0824 | 2.4555 | 2.0739 | 323879 | 420660 |
| 2011_09_26_0015 | 14887 | -0.0601 | -0.0404 | 3.4339 | 3.0581 | 296940 | 357545 |

### Decision and limits

Full-pose recovery is **2/12 for both**. All 24 fits report convergence, so each
method has ten converged-but-inaccurate outcomes. UniMatch gains 0009-negative
but loses 0002-positive; 0011-positive passes with both. Translation improves in
10/12, with median error 0.2771 → 0.2131 m. Rotation improves in only 6/12 and its
median worsens 0.7135 → 0.7565 degrees. No default promotion or gate relaxation.
The twelve seeds are paired development cases, not twelve independent scenes.

UniMatch produces more valid depth in every frame. On common points, median
residual magnitude improves in 2/6, absolute p90 in 4/6; the signed median remains
negative in all six for both methods. These mixed results do not establish a
unique error cause. Denser learned stereo alone has not solved full-pose recovery.
CPU forward times are 13.77–15.50 s per frame, excluding model loading and
preparation. No inference failed or required a retry.

[Full results](../assets/learned_stereo_20261029/evaluation.json) include runtime,
input/output hashes, all solver outputs and paired diagnostics;
[JSON schema](../assets/learned_stereo_20261029/evaluation.schema.json) preserves
the comparison envelope. The SGBM method objects match the previous evaluation
exactly in all twelve cases. Eight relevant unit tests, repository Ruff checks
and strict documentation build pass.

The next useful test is a matched **three-frame joint** comparison, using the
already archived frames and the same frozen model, to see whether the observed
translation benefit survives aggregation and whether rotation improves. Recompute
both frontends for all frames: historical un-re-rectified joint D2D 5/12 is not
a matched control. Pre-register this before running, retain both seed directions,
and avoid more matcher/confidence-threshold tuning on these results. Even a
positive result would remain spent-data evidence and require independent validation.
