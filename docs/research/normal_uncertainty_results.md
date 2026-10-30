# Conservative angular confidence rejects all local surface selections

Frozen `ce9d5e9e`, evaluated 2026-10-30. See
[protocol](normal_uncertainty_protocol.md),
[plan](../assets/normal_uncertainty_20261030/plan.json),
[27 conditions](../assets/normal_uncertainty_20261030/evaluation.json),
[schema](../assets/normal_uncertainty_20261030/evaluation.schema.json) and
[Monte Carlo validation](../assets/normal_uncertainty_20261030/validation.json).

The angular confidence gate retains **zero points in all27 conditions**, including
clean controls. Therefore every noisy condition fails retention and spatial support;
precision/coverage for the empty selected population are null, not perfect accuracy.
Do not apply this gate to real data or rerun registration with empty inputs.

| Population | Angular bound median | Observed angular coverage |
| --- | ---: | ---: |
| All valid, clean | 90 deg in every frame | 91.60–93.67% |
| All valid, seed730 | 90 deg in every frame | 90.68–92.77% |
| All valid, seed731 | 90 deg in every frame | 90.78–92.73% |
| Prior gate, clean | 90 deg in every frame | 100% |
| Prior gate, seed730 | 90 deg in every frame | 99.14–99.58% |
| Prior gate, seed731 | 90 deg in every frame | 98.90–99.36% |
| Prior + angular <=15 deg | No retained points | Undefined |

High coverage with a90-degree bound is uninformative for precise registration.
Despite many broad bounds, all-valid coverage is below nominal95%: curved surfaces
and boundaries violate the affine-disparity model even in clean measurements.
The .5px model is deliberately assumed on clean controls, as frozen. Ground-truth
normals are analytic and only used for evaluation; candidate normals, covariance
and selection use noisy disparity and intrinsics only.

## What the uncertainty calculation does establish

Independent .5px noise on25 pixels gives coefficient covariance
`diag(.25/50,.25/50,.25/25)` for centered `[dx,dy,1]` regression. The known mapping
from disparity slopes to an unnormalized normal transforms this covariance. A
95% coefficient ellipsoid is enclosed by a Euclidean ball using the largest
normal covariance eigenvalue; its angular cone is conservative under an exact
plane model. This construction is not a calibrated real-stereo likelihood.

The20000-sample exact tilted-plane check has **99.465% coverage**, median cone
radius35.68 degrees and median actual angular error9.19 degrees. It passes the
>=95% sanity check, but also demonstrates substantial conservatism. Consequently,
zero retained points does **not** prove the measurements intrinsically contain no
normals accurate to15 degrees. It proves that this particular bound cannot certify
usable normals under the fixed gate. Distinguish a loose bound from actual error.

All27 generated depth hashes match the preceding reliability experiment. Prior
selection remains identical; no thresholds or windows were changed after outcomes.
Four focused tests, repository Ruff and strict documentation build pass. Validation
checks schema, all source hashes, case counts, paired depths, invalid-patch handling
and exact-plane coverage. No calibration, real-data read, default change or SOTA.

Next assess a tighter angular uncertainty construction under the same independently
specified exact-plane model before changing the window size or another threshold.
Check coverage and retained information together. Then test affine-model error on
curved/boundary surfaces separately; a covariance derived from noise alone cannot
certify model bias. If usable support still fails, move to larger coherent surface
estimation rather than retuning acceptance on spent real frames. ICP and real-data
transfer remain blocked by the present failed quality screen.
