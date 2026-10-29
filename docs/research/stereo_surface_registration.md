# Stereo surface registration with existing tangent-plane ICP

## Frozen experiment

Use the existing Chen–Medioni discrete tangent-plane ICP implementation rather
than extending the image-support penalty. Correspondences are refreshed 3D
nearest neighbors with mutual checks, a 0.5 m gate and 80% point-plane trimming.
Target normals come from local PCA; rank/condition checks and a spatial 20% source
holdout are retained. Multi-start and effective-curvature diagnostics are disabled
for bounded runtime. The holdout is a spatial subset of the same frame, not
independent sensor truth or a separate recording.

Before real-data execution, five controlled three-plane tests recover <0.05 deg
and <5 mm pose error and <5 mm holdout plane RMSE: complete surfaces, target holes,
partial source overlap with distant unmatched points, their combination, and a
foreground patch replacing a background patch only in the target. These are
geometric point-cloud controls, not a realistic stereo noise/synchronization
simulation. They validate recovery with missing/conflicting geometry, beyond
continuity alone.

Freeze code/options/protocol before real execution. Use the **middle archived
frame** for each of six spent recordings and both original initial perturbations:
12 ICP fits and 12 newly matched single-frame D2D controls. Both receive the same
raw frame depth, cloud, intrinsics and initial pose. Do not compare success rates
as though the historical three-frame D2D run used the same observations.

Backproject valid 0.1–80 m stereo depth into original-left camera coordinates.
Select LiDAR points in the initial camera field of view and depth range, without
using reference calibration. For ICP only, deterministically retain the first
point in 0.2 m voxels then uniformly cap at 6,000 points per cloud. D2D retains
its existing preprocessing/defaults. No target-normal filling, support-field
penalty or cross-frame point association is introduced.

ICP retains its 50-iteration default and 1e-7 update tolerance. Report all raw
candidates, status, source/target sizes, train/holdout plane residuals, tangent
rank/condition and signed probes. Treat `converged` as numerical acceptance only;
score the unchanged <0.5 deg and <0.20 m gates after both fits. Non-converged ICP
returns the original pose while preserving the raw candidate when available.
Represent nonfinite diagnostic evidence explicitly as strings, not JSON NaN.

No defaults are promoted, no fresh evaluation inputs read, and no accuracy-driven
reruns or parameter changes are allowed. Compare all twelve cases, including
failed geometry or support, without counting a rejected candidate as recovery.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_stereo_surface_registration.py \
  docs/assets/stereo_surface_registration_20261029/plan.json /tmp/surface-icp.json
```


## Results

Code, settings and protocol were frozen at `464279c5`. Five controlled geometry
scenarios and the five existing tangent-ICP tests passed. All **24 real fits**
(12 ICP plus 12 newly matched D2D controls) completed without outcome-based
retries or changes to thresholds.

- Surface ICP recovery: **0/12**. Seven converge, but all seven fail joint
  reference-accuracy gates. Five hit the iteration limit and return the exact
  initializer, with raw candidates retained.
- Matched **single-frame** D2D recovery: **3/12**. This is not the historical
  three-frame D2D result of 5/12; do not replace or reinterpret that verdict.
- All ICP fits have tangent rank six. Spatial holdout point-plane RMSE spans
  **0.0317–0.0712 m**, yet none meets full-pose accuracy criteria. Local rank and
  same-frame held-out residuals are not sufficient accuracy evidence.

Raw translation differences below are metres to the dataset reference; rotations
are degrees. Rejected ICP candidates are shown for diagnosis only.

| Recording / seed | ICP status | ICP raw rotation | ICP raw translation | Holdout plane RMSE | D2D rotation | D2D translation | D2D recovered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0005 positive | max_iterations | 1.2815 | 0.3921 | 0.0626 | 1.1977 | 0.3189 | False |
| 0005 negative | max_iterations | 0.1973 | 0.4431 | 0.0651 | 0.5732 | 0.1946 | False |
| 0002 positive | converged | 0.7461 | 0.3220 | 0.0317 | 0.3910 | 0.2365 | False |
| 0002 negative | converged | 0.9472 | 0.3287 | 0.0360 | 0.0214 | 0.1658 | True |
| 0009 positive | max_iterations | 0.9588 | 0.2269 | 0.0524 | 1.1283 | 0.1820 | False |
| 0009 negative | max_iterations | 0.5832 | 0.1624 | 0.0538 | 0.2968 | 0.2473 | False |
| 0011 positive | converged | 1.1247 | 0.2782 | 0.0357 | 0.3341 | 0.1641 | True |
| 0011 negative | max_iterations | 1.8009 | 0.7149 | 0.0359 | 0.3382 | 0.5852 | False |
| 0013 positive | converged | 0.6415 | 0.1514 | 0.0414 | 0.9923 | 0.3163 | False |
| 0013 negative | converged | 0.7947 | 0.1753 | 0.0428 | 1.1291 | 0.4390 | False |
| 0015 positive | converged | 0.3267 | 0.2740 | 0.0531 | 0.3668 | 0.1958 | True |
| 0015 negative | converged | 1.4670 | 0.5643 | 0.0712 | 0.4221 | 0.2184 | False |

### What this changes

The controlled tests now exercise actual pose recovery with missing, conflicting
and partially overlapping surfaces, not just objective continuity. However, those
idealized clouds do not reproduce noisy stereo reconstruction, correlated surface
bias, time alignment or realistic cross-sensor occlusion. Their success does not
predict real calibration success.

The real test replaces projective matching with existing mutual 3D matching and
robust tangent-plane trimming, yet fails to produce a qualifying pose. That is
evidence against adopting this specific adapter/settings, not against ICP in all
settings or against the original Chen–Medioni method. It also shows why a spatial
holdout drawn from the same potentially biased stereo target is not an independent
check of camera/LiDAR calibration.

### Decision

Keep the adapter experimental; no solver or frontend defaults change. Do not
relax convergence, fit acceptance thresholds to these twelve outcomes, or report
the seven converged poses as successful calibrations. Preserve all 24 outputs.

The next substantive comparison should challenge the shared stereo reconstruction
assumption with an independently specified depth/correspondence source, rather
than another nearest-neighbor or penalty variant over the same depth arrays.
Any new source must have reproducible version/settings, clear input/calibration
requirements and the same complete failure-aware comparison. No fresh evaluation
or SOTA claim follows from these development experiments.

[Evaluation](../assets/stereo_surface_registration_20261029/evaluation.json),
[schema](../assets/stereo_surface_registration_20261029/evaluation.schema.json),
[execution log](../assets/stereo_surface_registration_20261029/execution.log),
[test log](../assets/stereo_surface_registration_20261029/tests.log) and frozen
settings preserve reference/input identities, source/target counts, normal/rank
statistics, correspondence histories, holdout evidence and signed perturbation
probes. Nonfinite evidence, if present, is represented explicitly by strings.
