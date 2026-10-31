# Joint pose fitting with pixel-scaled inverse-depth residuals

## Frozen development protocol

Implement a new experimental six-DoF estimator in the ROS-independent core.
Residual is `fx * baseline * (1 / stereo_z - 1 / transformed_lidar_z)` in pixel
units, divided by 0.5 px. It is an **equivalent-disparity/inverse-depth residual**,
not the exact raw horizontal disparity for non-horizontal stereo baselines.
This formulation naturally gives smaller metre-error weight at large depth,
without a reference-derived distance cutoff. It replaces noisy point-plane
normals with bilinear inverse-depth sampling and jointly fits rotation/translation.

Freeze code, options and the 12-case plan before testing on real inputs. Use all
six spent recordings (0005,0002,0009,0011,0013,0015), both original perturbations,
and the exact depth/cloud/intrinsic inputs used by each archived joint D2D run.
The first four drives retain historical depth conversion; the last two use its
corrected version. Each paired comparison uses identical inputs; results across
these groups do not isolate frontend differences. Do not silently regenerate
historical depth or use D2D output as the new method's starting point.

- Select an initial nearest-visible-point population, with four finite depth
  neighbors, local equivalent-disparity range <=2 px and initial residual <=8 px.
  Cap at 3,000 points/frame by deterministic point-index priority; require 128.
- Hold these LiDAR point identities fixed throughout the solve. Sample inverse
  depth bilinearly at the current pose. Missing support receives a fixed 16-unit
  residual rather than disappearing from the objective. This penalty and validity
  boundaries are discontinuous and can still produce local traps.
- Optimize left-multiplied rotation-vector increments and additive translation
  with soft-L1 least squares, 150 evaluations, per-axis bounds ±3 deg and ±0.5 m,
  and `ftol=xtol=gtol=1e-8`. Publish the exact options and optimizer diagnostics.
- Reject unsuccessful termination, proximity to a bound, final support below 128
  or 90% of its initial population, and robustly weighted Jacobian condition
  number >1e6. Parameters are scaled by the search bounds for conditioning.
  This is a local numerical check, not global observability or accuracy assurance.
- Preserve every raw candidate. Rejected fits return the initializer exactly.
  Score reference rotation/translation only after the solver finishes. Compare
  accepted-stage recovery against the unchanged 0.5 deg / 0.20 m joint gates.

Synthetic tests recover all six components from independent known planes,
reject an unobservable single plane and reject absent depth. No production
registration/default is changed, no fresh evaluation data is used, and no
post-result adjustment or favorable-case rerun is planned. The implementation is
a development candidate, not a SOTA claim.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_disparity_pose.py \
  docs/assets/disparity_pose_20261029/plan.json /tmp/disparity-pose.json
```


## Results

Code/options/plan were frozen at `517c4eb6`. All twelve fits completed without
parameter changes or retries. The synthetic three-plane test recovered the pose,
but the real-data comparison **does not support adopting the new estimator**.

- Archived matched-input D2D recovery: **5/12**.
- New estimator accepted-stage recovery: **1/12**; returned-pipeline recovery
  **1/12** as well (original perturbations do not satisfy the rotation gate).
- New estimator accepts **4/12** on numerical checks; only **1 of those 4** meets
  the independent reference accuracy gates. The other three are false accepts
  under these development scoring criteria.
- The other **8/12** are rejected for `lost_support`, with exact initializer
  fallback. Each initial frame has 3,000 selected points. Final retained counts
  range from 2,332 to 2,895; failure is the frozen 90% retention rule, not an
  inability to reach the 128-point minimum. No retention threshold is relaxed.

All raw candidates remain visible below. Distances are to the dataset reference,
not independent ground truth. A candidate's rotation or translation improvement
alone does not imply a successful full calibration.

| Recording / seed | New status | New raw rotation (deg) | New raw translation (m) | New accepted recovery | D2D rotation (deg) | D2D translation (m) | D2D recovery |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 0005 positive | lost_support | 0.5227 | 0.1304 | False | 0.8406 | 0.2773 | False |
| 0005 negative | lost_support | 0.6737 | 0.0833 | False | 0.2416 | 0.1288 | True |
| 0002 positive | lost_support | 0.5172 | 0.1500 | False | 0.1652 | 0.1064 | True |
| 0002 negative | converged | 0.8516 | 0.0565 | False | 0.1168 | 0.1821 | True |
| 0009 positive | lost_support | 0.6708 | 0.1669 | False | 0.8124 | 0.2569 | False |
| 0009 negative | converged | 0.4533 | 0.0794 | True | 0.3340 | 0.2915 | False |
| 0011 positive | lost_support | 0.4752 | 0.2193 | False | 0.3001 | 0.1358 | True |
| 0011 negative | converged | 0.5961 | 0.0796 | False | 0.1601 | 0.2751 | False |
| 0013 positive | lost_support | 0.5243 | 0.0368 | False | 0.6570 | 0.3021 | False |
| 0013 negative | converged | 1.6521 | 0.1700 | False | 0.9426 | 0.5408 | False |
| 0015 positive | lost_support | 0.9181 | 0.2231 | False | 0.2688 | 0.2019 | False |
| 0015 negative | lost_support | 0.9806 | 0.1117 | False | 0.2416 | 0.1663 | True |

Every inner optimizer terminates successfully in 18–30 reported function
evaluations, but that does not establish either accurate calibration or sound
visibility modeling. Bilinear interpolation is continuous only within valid
neighborhoods. The fixed missing-data penalty and neighborhood-validity gates
introduce discontinuities when projections cross holes or boundaries. A numerical
Jacobian near those transitions can be misleading, and its condition number is
not an independent acceptance check. These are known modeling limitations, not
isolated causal conclusions from this experiment.

### Decision

Keep this estimator experimental and unregistered. Do not replace D2D, relax
retention, increase a success count by accepting rejected candidates, or claim
SOTA. This experiment tested a substantive model change—joint six-DoF fitting in
pixel-scaled inverse depth—rather than another frontend threshold tweak, and
its negative result is retained.

The next estimator revision should explicitly model visibility and depth-boundary
transitions with an objective whose derivatives remain meaningful, and verify
those behaviors with controlled occlusion/hole tests before another real-data
run. Full-pose acceptance also needs independent validation: the current local
numerical criteria accepted three inaccurate calibrations. Further changes must
use a new frozen protocol and preserve this 1/12 outcome.

[Evaluation](../assets/disparity_pose_20261029/evaluation.json),
[schema](../assets/disparity_pose_20261029/evaluation.schema.json) and
[execution log](../assets/disparity_pose_20261029/execution.log) preserve all input
hashes, baseline comparisons, rejected raw candidates, retained counts, costs,
scaled parameters and Jacobian singular values. No original inputs or historical
verdicts were changed; no fresh evaluation data was used.
