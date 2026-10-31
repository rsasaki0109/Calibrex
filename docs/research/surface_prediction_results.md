# Disjoint spatial prediction passes the controlled surface-quality screen

Frozen `d72a83a1`, evaluated 2026-10-30. See
[protocol](surface_prediction_protocol.md),
[plan](../assets/surface_prediction_20261030/plan.json),
[all54 paired rows](../assets/surface_prediction_20261030/evaluation.json),
[schema](../assets/surface_prediction_20261030/evaluation.schema.json), and
[exact-plane validation](../assets/surface_prediction_20261030/evaluation.validation.json).

The additional predictive model test passes the declared quality/support screen
in **18/18 noisy frame conditions**, versus10/18 for the fixed21px baseline.
All nine clean conditions also pass, versus2/9 baseline. This supports the
measurement-only surface construction on these controlled geometries; it does
not yet establish correspondence accuracy, ICP recovery or real-data performance.

| Condition | Method | Median joint surface precision | Median domain retained | Frames passing all gates |
| --- | --- | ---: | ---: | ---: |
| Clean | Baseline21 | 74.65% | 33.34% | 2/9 |
| Clean | Predictive21 | 98.04% | 25.47% | 9/9 |
| Seed730 | Baseline21 | 80.94% | 30.65% | 5/9 |
| Seed730 | Predictive21 | 99.92% | 23.44% | 9/9 |
| Seed731 | Baseline21 | 81.69% | 30.81% | 5/9 |
| Seed731 | Predictive21 | 99.87% | 23.86% | 9/9 |

Noisy selected precision ranges99.52–99.99% (seed730) and99.63–99.98% (731).
Retention ranges14.13–30.81% and14.46–31.07%. Every noisy frame also retains at
least eight image cells and the declared normal-direction spectrum ratio. All
27 baseline rows exactly reproduce the archived21px result; paired measurement
hashes match. The candidate changes only the selection mask, preserving the full
21px estimated depth/normal/confidence at retained pixels.

The exact-plane test accepts98.88% of10000 independent noise draws against the
nominal99% predictive gate, passing the frozen98–99.7% sanity interval. It uses
prediction covariance with shared coefficient uncertainty, not an independence
assumption for residuals produced by the same fitted plane. The two checkerboard
pixel sets have disjoint measurement noise under this synthetic model.

## Limits and interpretation

The test fits221 pixels and predicts220 held-out pixels. It then selects locations
whose original full441-pixel estimate is retained; final estimates reuse held-out
pixels. This is model validation followed by refitting, not an independent final
uncertainty audit. Checkerboard spatial samples are also not independent recordings.
Real stereo errors may be correlated and may not have the assumed .5px variance.

Joint precision compares smoothed XYZ with exact analytic XYZ within .2m and
unsigned normal error within15degrees. This is not proof that nearest-neighbor
cross-modal pairings refer to the same physical surface, nor full six-DoF
observability. Normal spectra are not covariance. Controlled layouts are spent
development data, not fresh evaluation or an independent dataset family.

Observed angular coverage for selected noisy surfaces ranges94.40–98.57% for
seed730 and93.65–96.91% for731. Clean coverage ranges91.26–97.00%. Therefore the
noise-only angular bound must not be advertised as guaranteed95% coverage on every
curved frame. Clean precision is95.52–99.07%, lower than noisy selected precision
in some frames: selection distributions differ; this is not evidence that adding
noise improves the underlying surface measurement.

Five focused tests, repository Ruff and strict documentation build pass. Tests
cover exact-plane prediction, curvature rejection, missing data, unchanged selected
estimates, empirical Gaussian predictive calibration and affine reconstruction.
Validation checks schema,54-case completion, all frozen hashes, input identity
and exact archived-baseline reproduction. No real-data read, calibration rerun,
core/default change or SOTA claim occurred.

Next make the construction camera-parametric, verifying exact parity on these
frozen controls. The pilot currently uses analytic-camera constants fx=fy240 and
fx*baseline120; applying it verbatim to KITTI would be incorrect. Bind real camera
intrinsics, stereo geometry and assumed noise explicitly in a new plan, account
for re-rectification/resampling limitations, then assess the same18 spent real
frames without fitting thresholds to reference normals. Only after that input
quality check should a matched ICP comparison be frozen. Retain all rejected
points/counts and failed frames; this synthetic pass is not adoption permission.
