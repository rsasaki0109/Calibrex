# Camera–LiDAR: first shared-bundle six-DoF comparison

The external comparison now has two executed optimizers receiving the same
integrated image/cloud and initial six-DoF transform. This is a development
experiment on three already-spent KITTI 0005 frames, **not an end-to-end benchmark
or a SOTA audit**. The native arm is the existing Pandey intensity-to-intensity
implementation; it is not the D2D release candidate or the recent rotation checker.

## Result

| Arm | Rotation reference disagreement | Translation reference disagreement | Strict `<0.5 deg / <0.20 m` event |
| --- | ---: | ---: | --- |
| Shared initial transform | 1.5029 deg | 0.12247 m | false |
| Official Koide NID BFGS | 0.6452 deg | 0.18014 m | false |
| Existing native Pandey MI | 1.2808 deg | 0.12190 m | false |

Both processes completed. Koide retained Wolfe line-search warnings. Native
Pandey stopped after four iterations because monotonic backtracking found no
improving step; its `converged` status does not mean accurate calibration.
Neither output satisfies the scoped recovery event. A single shared bundle
cannot establish comparative success rates, independent evidence, or superiority.

The [report](../assets/shared_koide_bundle_20261028/comparison.json) and
[schema](../assets/shared_koide_bundle_20261028/comparison.schema.json) retain
transforms, logs, source hashes, input hashes, solver options and limitations.
The earlier unadapted smoke remains published separately; it is not replaced.

## Correct time origin, without inventing better timing

The KITTI converter explicitly models each point's time relative to the scan's
forward-facing middle timestamp. Negative offsets are valid in that convention.
Koide's pinned source documents a frame timestamp at the first point and point
times relative to that first point (`include/vlcal/common/time_keeper.hpp`). Its
negative-time fallback subtracts the minimum from offsets and also subtracts it
from the header (`src/vlcal/common/time_keeper.cpp`), which does not preserve
`header + offset` for this input convention. We do not modify external source.

The new `tools/koide_kitti_time_adapter.py` instead applies:

```text
shift_ns = floor(min(point_offsets_s) * 1e9)
new_header_ns = old_header_ns + shift_ns
new_offset_s = old_offset_s - shift_ns * 1e-9
```

It stably orders points by time, preserving xyz and intensity, and writes float64
relative times. It accepts only the converter's explicit five-float KITTI layout,
rejects missing/nonfinite/degenerate timing, and copies other messages unchanged.
The original bag remains untouched. The three header shifts were approximately
-51.635, -51.651 and -51.671 ms; measured arithmetic residual in acquisition time
was zero for every point. The final adapter reproduced every consumed message
byte-for-byte in a separate replay. Tests also cover positive offsets and rejected
invalid times. This validates the conversion, **not** the underlying azimuth model.

Official preprocessing and calibration then exited 0, with no negative-time,
missing-time or frame-rewind warnings. The integrated cloud contains 370,836
points. The mapping between the integrated cloud's reference pose and the camera
capture time still needs verification; the diagnostic reference disagreements
above must not be treated as a verified physical extrinsic accuracy measurement.

## What was shared and what was not

Both optimizers consumed the exact same PNG/PLY bundle produced by official Koide
preprocessing and the same reference-derived diagnostic seed used in the earlier
smoke. The native driver checks that its seed equals the inverse of Koide's
`init_T_lidar_camera`. All points are supplied; no outcome-dependent subsampling
or parameter tuning was performed.

`tools/koide_shared_bundle_pandey.py` invokes the existing
`PandeyMutualInformationSolver` with unchanged optimization defaults. Only
`min_train_observations=1` and `holdout_ratio=0` change to allow this single bundle;
there is no held-out validation. Full options and iteration evidence are archived.
Koide uses its unchanged NID BFGS defaults. The shared Koide preprocessing makes
the native arm a hybrid pipeline, useful for checking optimizer behavior but not
an independent end-to-end Calibrex result.

The native optimizer call took 5.22 s, excluding loading and preprocessing. Koide
was not separately timed from preprocessing/display startup. Native execution
used one OpenBLAS thread and `OMP_NUM_THREADS=4`; the Koide container had a 4-CPU,
4-GB limit and a 180-second timeout. These are not speed-comparison measurements.

To avoid duplicating another multi-GB Docker `vfs` layer, the stopped smoke
container was reused with a temporary startup wrapper invoking the archived
`koide-job.sh`. The official calibration source and binaries were not changed.
The first wrapper attempt failed while sourcing ROS with shell `nounset`; that
setup failure is retained. The corrected wrapper removed `nounset` and completed
from `2026-10-28T13:47:31.194855364Z` to `13:47:38.379157278Z`, exit 0, no OOM.
The original entrypoint was restored afterward. Networking remained as in the
previous package-installation container; this was not a network-isolated run.

## Reproduction and next milestone

From the repository root, using the same existing spent smoke source:

```sh
PYTHONPATH=src:. python tools/koide_kitti_time_adapter.py \
  /path/to/original-smoke-bag /path/to/paired-input/spent-0005-three-frames \
  --report /path/to/time-adapter.json
```

Run the archived Koide job with `paired-input` under `/data`, the previous
`calib.json` holding the fixed seed, and Xvfb. Then:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 180s \
  python tools/koide_shared_bundle_pandey.py \
  /path/to/paired-output /path/to/seed.json /path/to/paired-native.json
```

Next, establish camera/cloud reference-time alignment and build the independent
end-to-end comparison on representative spent intervals. The D2D release
candidate needs an independently generated image-depth input; this mono-only bag
cannot supply one, and projecting LiDAR with the reference extrinsic to fabricate
it would contaminate the task. Select and freeze that provider and its exposure
before running D2D. Required 55-drive development coverage and learned external
baselines remain outstanding. No fresh evaluation sensor payload was read.

Do not tune either optimizer on this one result. The next deliverable is broader,
physically aligned comparison evidence, not another two-drive objective tweak.
