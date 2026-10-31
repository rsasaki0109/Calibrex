# Camera–LiDAR: motion-aligned stereo D2D execution

The comparison now has a shared point cloud expressed at the camera capture time
and a depth image produced without the target camera–LiDAR reference transform.
The existing six-DoF D2D solver and official Koide optimizer both execute on that
cloud from the same seed. **Neither recovers the reference within the scoped
thresholds.** This is one spent-data development case, not an audit or proof of
comparative superiority.

## Result and input conditions

| Arm | Rotation reference disagreement | Translation reference disagreement | `<0.5 deg / <0.20 m` |
| --- | ---: | ---: | --- |
| Initial transform | 1.5029 deg | 0.12247 m | false |
| Stereo-assisted native D2D | 1.1097 deg | 0.27386 m | false |
| Koide NID BFGS | 1.3735 deg | 0.19471 m | false |

D2D stopped with `converged` after 193 evaluations (1.80 s for its solver call,
excluding the frontend). Rotation improved, but translation got worse. Koide
exited 0; neither completion status establishes successful recovery. No settings
were tuned after these results.

The [schema-valid report](../assets/aligned_stereo_d2d_20261028/comparison.json)
retains input/source identities, options, raw outputs, complete D2D trace and
execution logs. [Frontend evidence](../assets/aligned_stereo_d2d_20261028/frontend.json)
records the depth coverage and motion model. The initial transform is numerically
checked against the inverse of Koide's native initialization field.

This case is **stereo-assisted**, not an equal-information monocular comparison:
D2D uses the right image and calibrated stereo baseline; Koide uses the left
image and point intensity. Both use the same OXTS motion and known IMU–LiDAR
extrinsic for point compensation. This is a separate assisted-input track.
The D2D options are the existing `BorerSixDofOptions` defaults, not the exact
`integrated-r5` release protocol. The recent rotation checker is not used.

## Why the earlier shared bundle was insufficient

Pinned Koide source `02a0dc039f5509708f384be4ff3228e0ae09352d` reads the first
image (`src/preprocess_ros2.cpp::get_image`) and places the first cloud into the
map with identity begin/end poses
(`src/vlcal/preprocess/dynamic_point_cloud_integrator.cpp::insert_points`).
The first scan therefore receives no within-scan motion correction. The source
KITTI 0005 OXTS stream already reports roughly 3.5 m/s at that time. Correcting
only the point-time origin does not make this cloud an instantaneous scan at
the image timestamp. The previous shared-bundle figures remain diagnostics;
they are not upgraded to physical accuracy measurements.

`tools/kitti_aligned_stereo_bundle.py` instead transforms every point through the
interpolated OXTS pose at its modeled acquisition time, then into the LiDAR frame
at the left image timestamp. It uses linear position interpolation and quaternion
SLERP, with the known IMU–LiDAR lever arm and rotation. Times are seconds relative
to an integer-nanosecond origin, avoiding epoch-sized float subtraction.
Extrapolation is rejected.

Frame `0000000001` is used to provide pose coverage on both sides of the scan;
the first scan extends before the first OXTS sample. This fixed-index choice was
made before calibration results. The aligned cloud has **123,683 points**.
Modeled compensation changes positions by a median **0.07614 m**, maximum
**0.53594 m**. These are model corrections, not measured scene displacement.
Tests recover fixed world points under simultaneous rig rotation, translation
and a nonzero IMU–LiDAR lever arm, and reject out-of-range query times.

No camera–LiDAR reference transform enters this frontend. The reference is used
only to create the common diagnostic perturbation and to score the results.
OXTS pose error, the known auxiliary extrinsic and the azimuth timing model
remain assumptions; this execution does not independently validate them.

## Independent stereo depth

The right-camera images for frames 0 and 1 and the timestamp file were fetched
from the same already-spent KITTI 0005 archive using pinned size/ETag, ZIP CRC
validation and output hashes. Only frame 1 enters the final run. The camera
intrinsics and **0.532725 m** baseline come from `P_rect_02/P_rect_03`.
No LiDAR projection or target camera–LiDAR reference is used to create depth.

The provider is CPU OpenCV StereoSGBM 4.14.0: 128 disparities, block size 5,
uniqueness ratio 10, speckle filtering, and the full fixed settings recorded in
the frontend report. Positive disparities produce optical-axis depth
`fx * baseline / disparity`, restricted to 0.1–80 m; other pixels are invalid.
**313,968 / 465,750 pixels (67.4%)** have valid depth. This is support coverage,
not a depth-accuracy measurement or evidence about moving objects.

Both optimizers receive the same float32 PLY positions. D2D uses the generated
depth; Koide uses the equalized left image and rank-equalized intensities.
Intensity ties are ordered stably in the custom frontend; this is not a claim
of byte-identical official preprocessing. Official Koide preprocessing is
bypassed for this shared motion model; its optimizer source/binary is unchanged.
The existing runtime container was reused with the archived startup wrapper,
then its original entrypoint restored. It exited 0 without OOM, from
`2026-10-28T14:23:36.738117192Z` to `14:23:40.817656278Z`. That container interval
includes startup and is not comparable to the D2D solver-only timing.

## Reproduction and next work

The builder accepts the original bag, KITTI raw directory, calibration directory,
right image, right-camera timestamps and a new output directory:

```sh
PYTHONPATH=src:. python tools/kitti_aligned_stereo_bundle.py \
  /path/to/original.db3 /path/to/raw-drive /path/to/calibration \
  /path/to/right/0000000001.png /path/to/right/timestamps.txt \
  /path/to/aligned-bundle --frame-index 1

PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 180s \
  python tools/run_aligned_stereo_d2d.py \
  /path/to/aligned-bundle /path/to/seed.json /path/to/d2d.json
```

Run D2D before the archived Koide job modifies `calib.json`; the archived
`calib-frontend.json` preserves its original input bytes. The source hashes and
that input hash were verified against the actual executions. Data arrays remain
in the workspace; only their hashes and generated reports are committed.

The next milestone is a frozen multi-interval, multi-drive development matrix,
with separate assisted-input and monocular tracks, failures retained, and a
shared resource policy. Stop repeating single-frame variants. Monocular D2D
still requires a frozen independent image-depth provider and its training
exposure record; stereo does not satisfy that missing prerequisite. Full 55-drive
development coverage, learned external baselines and fresh-data preregistration
remain outstanding. No fresh evaluation sensor payload was read.


The [fixed 12-case follow-up](camera_lidar_assisted_matrix.md) is now complete:
D2D recovers 5/12, Koide 0/12, with no computational failures. Native rotation
improves in all cases but translation worsens in ten. The next structural test
is joint multi-frame estimation, not tuning on individual failures.
