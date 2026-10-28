# Camera–LiDAR assisted-input development matrix

This plan freezes a 12-case execution before the matrix is run. It is spent-data
development, not fresh evaluation, and covers only 2 of the required 55 raw
drives. The previous single-frame result is not used to choose successful frames.

The [machine-readable plan](../assets/assisted_matrix_20261028/plan.json) and
[schema](../assets/assisted_matrix_20261028/plan.schema.json) pin source hashes,
archive identities, frame inventory and perturbations:

- KITTI 0005 (154 frames): indices 38, 76, 114.
- KITTI 0002 (376 frames): indices 93, 187, 281.
- Selection is `floor((N-1)*q/4)` for `q=1,2,3`, without image inspection.
- Each receives xyz Euler `[1,-1,0.5]` deg / translation `[0.1,-0.05,0.05]` m
  and their sign-negated versions. Rotation is left-multiplied onto the reference;
  translation is added in camera coordinates, as in the earlier diagnostic.

Each case uses the unchanged stereo/OXTS frontend and default bounded six-DoF
D2D solver from the [aligned stereo execution](camera_lidar_aligned_stereo_d2d.md).
Koide uses its unchanged official NID BFGS optimizer on the identical float32
cloud, with the same seed. Koide's preprocessing is bypassed for the shared
motion model. D2D additionally consumes stereo depth. This is an assisted-input
comparison with unequal sensor use, not monocular superiority evidence.

Cases run sequentially. Native execution is limited to four affinity CPUs with
one OpenBLAS thread; Koide retains its four-CPU/4-GB container and default internal
threading. Both get 180 seconds per case. The native timer includes its process;
Koide's inherited timeout also includes Xvfb startup. No speed-comparison claim
is planned. There are no outcome-dependent parameter changes or retries.

Report both rotation geodesic disagreement and translation norm against the
reference, plus the strict `<0.5 deg / <0.20 m` recovery event. D2D must also
converge, and both methods must complete with a finite rigid transform. Failed
preparation, timeout, invalid output or interrupted case remains in the fixed
12-case denominator. Missing cases are never dropped. Report per-drive outcomes
and descriptive median/p95 errors with valid-output counts; do not treat these
correlated frame/seed cases as independent recordings for confidence intervals.

A successful plan execution is not a SOTA verdict. Even complete success here
would leave 55-drive development coverage, independent-family evidence,
monocular depth-provider exposure, learned baselines and fresh preregistration
outstanding. No fresh evaluation sensor payload is authorized by this plan.

## Execution result

The plan and driver were committed at `6a1ff93d` before execution. All 12 scheduled
cases ran for both methods. There were no preparation failures, timeouts, invalid
rigid outputs or missing cases. All native cases returned `converged`; all Koide
containers exited 0. Those statuses do not establish accuracy.

| Method | Scoped recoveries | Failures | Median rotation / p95 | Median translation / p95 |
| --- | ---: | ---: | --- | --- |
| Stereo D2D | 5/12 | 7/12 | 0.448 / 1.071 deg | 0.187 / 0.336 m |
| Koide | 0/12 | 12/12 | 1.694 / 3.287 deg | 0.127 / 0.216 m |

| Recording | D2D recoveries | Koide recoveries |
| --- | ---: | ---: |
| 2011-09-26 0005 | 1/6 | 0/6 |
| 2011-09-28 0002 | 4/6 | 0/6 |

D2D improved rotation disagreement in **12/12**, but improved translation in
only **2/12**. Translation worsened in the other ten. Koide improved rotation
in 5/12 and translation in 5/12. These comparisons use each case's own initial
transform, including the small difference between positive/negative Euler norms.
A median below both thresholds does not make a failing case pass: the joint
per-case D2D recovery count remains five. D2D still fails the rotation threshold
in 6/12 and the translation threshold in 5/12, with four cases failing both;
rotation improvement must not be confused with sufficient rotation accuracy.

The [complete matrix](../assets/assisted_matrix_20261028/matrix.json),
[summary](../assets/assisted_matrix_20261028/summary.json) and
[summary schema](../assets/assisted_matrix_20261028/summary.schema.json) preserve
every case. Per-case archives contain seed/reference transforms, optimizer
outputs, full native candidate traces, frontend hashes, logs and container exit
states. The publisher checks exact inventory equality, the precommitted plan and
driver hashes, and native cloud/calibration input hashes against the frontend.
It does not omit failures from the scheduled denominator. No sensor arrays are
committed. The original Docker entrypoint was restored and byte-compared after
the batch; the runtime identity and final state are archived.

There is no monocular superiority claim: D2D uses stereo information unavailable
to the Koide objective, and both consume a custom OXTS-assisted cloud. Neither
method was tuned in this matrix. Koide's single-scan assisted-input result must
not be substituted for its official multi-scan workflow's general performance.
This result also does not update the separate rotation-verification audit.

## Interpretation and next experiment

Native translation stability is poor, and half the cases still miss the rotation
threshold despite improving from initialization. Do not raise the 0.20 m threshold, choose only
successful frames, or optimize another setting on this matrix. A substantive
next comparison is **joint estimation across the three time-separated frames**
per recording, using the same frontend, seeds and thresholds, against the
single-frame results retained here. This tests whether shared geometry across
views constrains translation; improvement is not assumed. Both optimizers can
consume multiple observations, so keep their common frame inventory matched.

That experiment remains spent-data development. Even if it works, required
55-drive coverage, monocular/learned competitors, independent-family evidence and
fresh preregistration remain outstanding. A larger initial-perturbation capture
range is also untested by these two sign directions.

Reproduce the frozen batch from the repository root, with the existing runtime
container whose `/data` bind mount is the output directory's parent:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python tools/run_camera_lidar_development_matrix.py \
  docs/assets/assisted_matrix_20261028/plan.json \
  /path/to/camera-lidar-recovery-20261027 \
  /path/to/existing-data-mount/new-matrix-directory
```

A new output directory is required, preventing accidental overwrite. Replaying
a development batch must retain its original result and be labeled separately.
