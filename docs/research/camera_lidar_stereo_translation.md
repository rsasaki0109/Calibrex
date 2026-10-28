# Camera–LiDAR: metric stereo translation refinement

This spent-data experiment tests a different geometric objective for the
translation regression seen in the [four-drive joint study](camera_lidar_expanded_joint.md).
It does not tune MI thresholds or revisit frame selection. Freeze the same eight
joint D2D outputs, three observations per case and existing success criteria
before fitting. The [plan](../assets/stereo_translation_20261029/plan.json) pins
source hashes, exact baseline identities and the full case inventory.

## Method and assumptions

`stereo_translation_refiner.py` is an opt-in, ROS-independent experimental solver;
it is not registered as a production default. Hold D2D's rotation fixed exactly
and fit only translation. Back-project metric stereo **optical-axis z** into 3D,
form local surface normals using neighboring points, and minimize point-to-plane
residuals between projected LiDAR returns and those surfaces. This compares
geometry in the camera frame, not camera z with LiDAR range.

The frozen [options](../assets/stereo_translation_20261029/options.json) specify:

- Center and four-neighbor positive finite depth, with a 5% local depth-jump mask.
- Nearest-point z buffering; at most 2,000 uniformly indexed matches per frame.
- Stereo disparity uncertainty 0.5 px, propagated via `z²/(fx*baseline)` into
  normal-direction uncertainty, combined with a 0.02 m floor.
- A depth association gate of `max(1 m, 3*sigma_depth)` and soft-L1 fitting.
- At least 64 matches in **every** frame, weighted-design condition number at most
  1,000, translation bounds ±0.5 m about D2D, and at most 15 outer iterations.
- Accept only a converged step below 0.1 mm with final support/rank checks and no
  bound hit. Otherwise return the D2D transform unchanged and retain the raw
  candidate, rejection reason and trace.

These are modeling choices frozen before real-data fitting, not independently
validated uncertainty or covariance estimates. Stereo depth can be biased and
surface normals are noisy; held-out calibration evidence is absent. The rank
check detects degenerate fitted geometry, not global calibration observability.
Fixed erroneous rotation can still be absorbed into translation; retaining
rotation isolates this failure mode rather than proving it absent.

Synthetic tests recover translation under a nonidentity fixed rotation from
three independent planes, reject a single-plane ambiguity, reject missing depth,
and retain the initializer when the bounded solution hits a limit. The real-data
fit receives only D2D's output, clouds, stereo depth, intrinsics and stereo
baseline. The scorer reads the reference **after** the fitting subprocess exits.

## Frozen scoring

All eight cases from 0005, 0002, 0009 and 0011 remain scheduled. Use four affinity
CPUs, one OpenBLAS thread, and a 180-second timeout per case. No outcome-dependent
reruns or parameter changes. Primary diagnostics are translation disagreement
change against D2D, counts improved/worsened/unchanged, and accepted/rejected fits.
Retain raw candidate errors even when the published experimental output falls
back. Separately report accepted-refinement recovery and whole-pipeline recovery
including the explicit unchanged-D2D fallback; fallback success is not success
of the new stage.

The strict joint event remains `<0.5 deg / <0.20 m`. Rotation stays unchanged, so
this stage cannot repair the two D2D initializers already failing rotation.
Do not compare an eight-case assisted-input result to a monocular SOTA result.
Four spent drives, two seed directions, uncertain stereo/motion models and
outstanding 55-drive / learned-baseline / independent-family / fresh-audit
requirements remain limitations.

## Execution result

Source and plan were committed at `9d01486a` before fitting. All eight subprocesses
completed, but **all eight hit the 15-iteration limit**. None satisfied the frozen
outer convergence condition, so the returned transforms are exactly equal
to the D2D initializers. No parameter was changed after inspecting the results.

- Accepted refinements: **0/8**; accepted-stage recoveries: **0/8**.
- Returned pipeline recoveries: **4/8**, unchanged from D2D because of fallback.
- Unaccepted raw candidates: translation improves in **5/8**, worsens in **3/8**.

| Recording / seed | D2D translation | Unaccepted candidate translation | Last outer step |
| --- | ---: | ---: | ---: |
| 0005 positive | 0.2773 m | 0.1556 m | 1.10 mm |
| 0005 negative | 0.1288 m | 0.1647 m | 2.06 mm |
| 0002 positive | 0.1064 m | 0.1282 m | 3.84 mm |
| 0002 negative | 0.1821 m | 0.0886 m | 22.51 mm |
| 0009 positive | 0.2569 m | 0.1383 m | 1.37 mm |
| 0009 negative | 0.2915 m | 0.1747 m | 1.39 mm |
| 0011 positive | 0.1358 m | 0.3397 m | 6.84 mm |
| 0011 negative | 0.2751 m | 0.1478 m | 11.75 mm |

Every outer iteration retained 2,000 matches in each frame, and every inner
least-squares solve succeeded. Last recorded weighted-design condition numbers range
approximately 3.79–6.33. Thus these runs did not stop for too few correspondences,
rank failure, a boundary hit or process failure; the outer translation updates
did not settle within the required 0.1 mm. This does not establish that simply
allowing more iterations would yield accurate calibration.

Correspondences and the sampled match set change at each outer iteration.
Their per-iteration costs are therefore not evaluations of one fixed set, and
must not be used as a monotonic accuracy claim. The current traces establish
non-convergence but do not isolate whether correspondence switching, stereo
bias, normal noise or fixed rotation error dominates. In particular, the raw
0011 positive candidate worsens from 0.136 to 0.340 m: apparent improvement in
other candidates does not justify adopting the stage or relaxing its criteria.

The [complete evaluation](../assets/stereo_translation_20261029/evaluation.json),
[summary](../assets/stereo_translation_20261029/summary.json) and
[summary schema](../assets/stereo_translation_20261029/summary.schema.json) retain
all candidates, rejection reasons and returned outputs. Per-case typed reports
include input/source hashes, fixed rotation, options, support counts, singular
values, inner status and outer steps. The publisher verifies the exact inventory,
source identities, unchanged rotation and exact fallback transforms. No fresh
sensor payload was read or downloaded.

## Decision and reproduction

Do not promote this experimental module or claim a calibration improvement.
The useful next diagnostic is association stability and stereo-surface validity,
with the existing traces and input inventory preserved; threshold relaxation is
not evidence that the model is sound. Neither the depth uncertainty formula nor
the fixed-rotation assumption has been validated on held-out recordings.

From the repository root with the existing local bundles:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python tools/evaluate_stereo_translation_refinement.py \
  docs/assets/stereo_translation_20261029/plan.json \
  /path/to/new-experiment-directory
```

A single fit can be replayed with `tools/run_stereo_translation_refinement.py
BUNDLE OUTPUT`. That executable never opens `seed.json`; reference scoring is
performed by the separate evaluator after fitting. The experiment is explicitly
assisted-input development, not a fresh audit or a new production default.
