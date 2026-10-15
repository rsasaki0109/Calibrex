# Batched probabilistic Camera–LiDAR pose evaluation

| Scope | Previous (s) | Current (s) | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 39.837464 | 9.799824 | 4.065x |
| Artifact-backed API, including schema validation | 32.170695 | 1.940954 | 16.575x |

| Repeat | Variant | Complete seconds |
| --- | --- | ---: |
| 0 | prior | 27.677353 |
| 0 | current | 9.776534 |
| 1 | current | 9.823113 |
| 1 | prior | 51.997574 |

Two alternating fresh-process pairs, one BLAS thread, no discarded warmup. Whole
time includes Python imports, artifact parsing, invariant capture, every pose
candidate, disjoint holdout, acceptance, full trace, saving, schema validation
and frozen-source verification. The unreserved host produced large variation in
the scalar baseline; pair ratios are about 2.83x and 5.29x. This four-worker
sample establishes no portable timing guarantee. Ratios apply to probabilistic
refinement only, not raw-sensor feature extraction or the full calibration stack.

Nine original KITTI raw clouds from `2011_09_30_drive_0018_sync` provide 1,024
deterministically selected points per frame, 9,216 total. A declared generated
camera pose projects them through pinhole, MEI and double-sphere models, with
fixed 0.35 px noise and anisotropic covariance. These are generated execution
controls, not independently measured image correspondences or calibration
accuracy evidence. Seven frames (7,168 correspondences) train; two (2,048) hold
out. Both implementations converge and accept the candidate after exactly 325
evaluations. No frame, point, candidate, iteration or holdout is removed by the
optimization. Raw KITTI coordinates and generated correspondence files remain
local; the archived builder and raw input hashes reproduce the fixture.

All complete result fields match exactly, including every trace objective,
candidate/selected transforms, acceptance and holdout. Only creation timestamps
and source commit provenance differ. The declared quaternion normalization and
Hamilton arithmetic order, camera projection powers, reductions and confidence
gates remain unchanged. Scalar fallback semantics are retained above the bounded
capture budget. Captures belong to one call and never enter a shared cache.

Prior runtime `2c82d9ef625673d9c236320af75b60b7dab6765f`; candidate runtime
`4e6b766cc86f679c73fcf912e2183a909a87f2a9`. Source capsules/manifests, complete outputs,
launcher, source-freezer helper, input hashes, schemas, publisher and 71-test
receipt are bound by SHA-256. The original scalar fallback body is also checked
against the prior evaluator. The failed schema-kind validation attempt remains
in `failed-validation-v1`, outside successful timing medians. Audit with:

```bash
python -m tools.audit_probabilistic_refinement_benchmark docs/assets/probabilistic_pose_batching
```
