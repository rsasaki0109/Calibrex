# Batched probabilistic Camera–LiDAR pose evaluation

| Scope | Previous (s) | Current (s) | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 29.919726 | 10.352140 | 2.890x |
| Artifact-backed API, including schema validation | 21.964982 | 2.005127 | 10.954x |

| Repeat | Variant | Complete seconds |
| --- | --- | ---: |
| 0 | prior | 30.445932 |
| 0 | current | 10.453279 |
| 1 | current | 10.251000 |
| 1 | prior | 29.393520 |

Two alternating fresh-process pairs, one BLAS thread, no discarded warmup. Whole
time includes Python imports, artifact parsing, invariant capture, every pose
candidate, disjoint holdout, acceptance, full trace, saving, schema validation
and frozen-source verification. The host was unreserved; the two complete pair ratios are 2.913x and 2.867x. This four-worker
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
gates remain unchanged. Tiny frames retain scalar evaluation. Scalar fallback semantics are retained above the bounded
capture budget. Captures belong to one call and never enter a shared cache.

Prior runtime `2c82d9ef625673d9c236320af75b60b7dab6765f`; candidate runtime
`0ea59903716ab2e269abf8f7d9e8f93444137fb7`. Source capsules/manifests, complete outputs,
launcher, source-freezer helper, input hashes, schemas, publisher and 73-test
receipt are bound by SHA-256. The original scalar fallback body is also checked
against the prior evaluator. The earlier dense-only candidate cohort is retained in `earlier-dense-v2` and is not pooled with this final-runtime comparison. The failed schema-kind validation attempt remains
in `earlier-dense-v2/failed-validation-v1`, outside successful timing medians. Audit with:

```bash
python -m tools.audit_probabilistic_refinement_benchmark docs/assets/probabilistic_pose_batching
```
