# Camera–LiDAR from a checkerboard (development)

`calibrex.evaluation.board_camera_lidar` measures `T_camera_lidar` from a checkerboard held
still at several poses in front of both sensors. It is the path for the camera-LiDAR
**translation**, which the targetless `camera-lidar` check does not judge (its edge objective
is flat within ±5 cm; see [camera-LiDAR check](camera_lidar_check.md)). This page records the
method and its first run on real data. It is development, not an audit: there is no
reference extrinsic for the data used here, and no claim is made.

## Method

Per board pose:

1. **Camera.** OpenCV finds the inner corners (`findChessboardCornersSB`, with
   `findChessboardCorners` and sub-pixel refinement as a fallback). The corners are
   undistorted with the camera model (`radtan`, `equidistant` or none) and the board pose
   comes from PnP (`IPPE`, refined by Levenberg-Marquardt). The board plane in the camera
   frame is its normal and offset.
2. **LiDAR.** The scan is mapped into the board frame the camera saw, through the current
   extrinsic. Points within a margin of the squares and of the board plane are candidates;
   RANSAC (2 cm) finds the plane, and least squares refines it to a fixed point, keeping
   the points within 3 robust sigmas of it (at least 2 cm): a Livox board's points spread
   over several centimetres at a distance. A plane turned more than 25 deg from the
   camera's board is rejected.

All poses together go to the Zhang–Pless plane solver of
`calibrex.solvers.planar_board_lidar_camera_solver` (rotation from the normals, then
translation from the offsets, with Huber reweighting). Each pose is weighted by
`(1 m / distance)^2`: a focal-length error scales the PnP distance, so the camera's
plane-offset error grows with the distance. The solve runs three times, the first
from a rough prior (only the axis permutation between the LiDAR and the optical frame is
needed) with a generous crop, then with the solution and a tighter crop.

**Observability.** A plane fixes the translation only along its normal. The information
`sum w n n^T` over the camera-frame normals has an eigen-direction with less than 1% of the
largest eigenvalue when the boards were only turned one way. The translation along such a
direction stays at the prior's value, so the next crop does not drift off the board. A
translation axis is reported as not observed when such a direction has 30% or more of it.
Every axis also gets a jackknife std: one pose left out at a time. An axis counts as
observed when its std is within 2 cm, or 0.3 deg for rotations, and its direction is
observed. Synthetic boards tilted both left/right and up/down recover all six axes to 1 cm
and 0.2 deg, and boards turned only left and right leave the vertical translation
unobserved (`tests/unit/test_board_camera_lidar.py`).

## First real data: the ACSC sample

The sample of ACSC (Cui et al., *ACSC: Automatic Calibration for Non-repetitive Scanning
Solid-State LiDAR and Camera Systems*; repository `HViktorTsoi/ACSC`, GPL-3.0; its data
are not redistributed here) holds 21 static board poses.

| Item | Value |
| --- | --- |
| LiDAR | Livox Horizon, one accumulated cloud per pose (1.2 M points) |
| Camera | 2048 x 1536, the sample's intrinsics and radial-tangential distortion (its 2 px skew dropped) |
| Board | 7 x 5 inner corners, 8 cm squares |
| Poses | 21, at 3.3–12.0 m, turned left and right only |
| Archive | `acsc_sample.zip`, sha256 `38459093f5cbac4b7acbd7331f42bf2aba76f2596af90ab1ae4f972104a58ca3` |
| Prior | the axis permutation only (identity rotation between the LiDAR and optical axes, zero translation) |

The camera found the board in all 21 images (reprojection 0.04–0.32 px), and the LiDAR plane
in all 21 (0.5–1.9 cm plane RMS).

Reproduce with the extracted sample (needs `opencv-python-headless`):

```bash
python tools/board_camera_lidar_acsc.py ACSC/sample --output acsc_board.json
```

| Axis (camera frame) | Value | Jackknife std | Observed |
| --- | --- | --- | --- |
| roll (about x) | 2.22 deg from the prior | 0.35 deg | no (std over 0.3 deg) |
| pitch (about y) | −0.25 deg from the prior | 0.299 deg | yes, just within the bound |
| yaw (about z) | −0.62 deg from the prior | 0.41 deg | no (std over 0.3 deg) |
| x | 13.3 cm | 1.6 cm | yes |
| y | −0.7 cm (the prior's 0, kept) | — | **no**: the board normals have almost no vertical component |
| z | 15.2 cm | 0.8 cm | yes |

What this shows:

- **Turned only left and right, the boards cannot fix the vertical translation.** The board
  normals' singular values are 4.20, 1.83 and 0.067; the weak direction is the camera's
  y axis. With the y translation free, it drifted by tens of centimetres between solves and
  pulled the next crop off the board; holding it at the prior and reporting it as not
  observed is what keeps the other axes stable.
- **The plane offsets disagree more for far boards.** Residuals are within 4 cm at
  3–4.4 m, up to 7 cm at 6–7 m and up to 9 cm at 9–12 m. A small focal-length or
  distortion error in the sample's intrinsics would do this; the distance weighting limits
  its effect, and boards within about 6 m are better.
- **Rotation is limited by the LiDAR plane normals.** A Livox plane fit on a 0.64 x 0.48 m
  board disagrees with the camera's by 0.3–3.1 deg per pose, so with 21 poses two rotation
  axes stay above the 0.3 deg bound and the third is just within it.
- **A fixed RANSAC band made the plane depend on the point order.** The board's Livox
  points spread over 4–6 cm at 6–7 m. With the first version's 2 cm band, a different
  slice was kept for each RANSAC sample: reordering a pose's points moved its plane by up
  to 3 deg and 10 cm. The fixed-point refinement removed that (the plane now moves by
  less than 0.001 deg and 0.1 mm under reordering), and the table is from it.

**For a capture:** tilt the board up and down as well as left and right, keep it within
about 6 m, hold it still, and use a dozen or more poses.

## The same poses through `calibrex estimate --board`

`calibrex.evaluation.board_camera_lidar_bag` finds the poses in a bag: images searched four
times a second, a pose wherever the board's centre stays within 1 cm and its normal within
0.5 deg for a second or more, and the LiDAR scans stamped inside that, less 0.2 s at either
end, accumulated. The 21 ACSC poses were packed into a rosbag2 for it: each pose as four
identical PNG images over 1.2 s (`CompressedImage`, with a `CameraInfo` holding the sample's
intrinsics) and its cloud split into ten `PointCloud2` scans inside the hold, then one second
without data. `calibrex estimate BAG --board 7x5x0.08 --pairs camera-lidar` (no `--tf`)
found 21 still poses and gave the result above to the last digit: the same axes observed,
x 13.27 cm and z 15.22 cm with the same stds. Without a prior that connects the two frames,
the pair is `partial` and no frame is exported; with a rough `--tf` prior, the axes the
boards do not observe are filled from it and marked NOT MEASURED, as for the other pairs.

## Pre-registered: KITTI's calibration recordings

KITTI's raw data hold one calibration recording per day, the vehicle standing still in
front of a dozen checkerboards of different sizes, all in view at once. Their reference is
KITTI's own calibration of each day (`calib_velo_to_cam`), itself made from these recordings
with a target-based method. The audit is pre-registered in
[`kitti_board_camera_lidar_preregistration.yaml`](kitti_board_camera_lidar_preregistration.yaml)
before any image, point cloud or OXTS record of the five recordings is read (only the
archives' file lists were):

- **Claim 1 (accuracy):** on all five days the calibration converges and lands within
  0.5 deg and 0.20 m of KITTI's calibration.
- **Claim 2 (honest stds):** on every day it converges (at least three), no axis it reports
  as observed fails `calibrex check --board`'s rule against KITTI's calibration.

For these recordings `find_boards` returns every board of the still left grey image, each
board's square size comes from KITTI's stereo pair (the sizes are not documented), and
radial-tangential undistortion now iterates to convergence (five iterations left up to
4 px in the corners of a lens like KITTI's). The harness,
`tools/kitti_board_camera_lidar_evaluate.py`, starts from the axis permutation only and reads
KITTI's camera-LiDAR calibration only to score the estimate.

### Result: both claims refuted, no estimate on any day

The harness ran once (2026-11-03, from a checkout whose `src/` and `tools/` equal the frozen
commit; records in [`docs/assets/kitti_board_camera_lidar/`](../assets/kitti_board_camera_lidar/run.json)).
It stopped before the calibration on four days and had too few boards on the fifth:

| Day (recording) | Still interval | Boards found in the chosen left image | Outcome |
| --- | --- | --- | --- |
| 2011_09_26 (0119) | 3.4 s | 30 (the cap) | no right image within 5 ms |
| 2011_09_28 (0225) | 3.8 s | 30 (the cap) | no right image within 5 ms |
| 2011_09_29 (0108) | 3.9 s | 17 | no right image within 5 ms |
| 2011_09_30 (0072) | 2.3 s | 17 | no right image within 5 ms |
| 2011_10_03 (0058) | 7.1 s | 30 (the cap); 16 in the right image | 2 boards with a stereo square size; 4 needed |

Claim 1 (accuracy on all five days) is refuted with no day converged; claim 2 is refuted
because it needs at least three converged days. The audit says nothing about the plane solve
on KITTI's Velodyne: no day reached it.

What failed, read from the records and, after the run, from the cameras' timestamp files:

- **The stereo time bound.** The harness took the right image within 5 ms of the chosen left
  one. In these recordings the two grey cameras run at about 5 Hz, and the nearest right image
  was 8 ms away on 2011_09_26 and 92-111 ms away (half a frame) on 2011_09_28, 2011_09_29 and
  2011_09_30; only 2011_10_03 was within 5 ms. With the vehicle standing still, any right
  image of the still interval would have served; the 5 ms bound was a wrong design choice,
  not a property of the data.
- **Board fragments.** `find_boards` with `CALIB_CB_LARGER` sometimes returned a 4 x 3 or
  3 x 7 corner of a board; painting over that piece left the rest to be found again, so frames
  reached the 30-board cap, and choosing the frame with the most detections favoured the most
  fragmented one. On 2011_10_03, 28 of the 30 left detections were such pieces and none
  triangulated with the right image. The two that did were 7 x 5 inner-corner boards whose
  stereo square sizes were 0.099 m and 0.100 m (0.21 and 0.02 px stereo error) at 5.6 and
  7.6 m.

The five recordings are now spent. Fixing either point (pairing the stereo images over the
still interval; rejecting grids smaller than a declared pattern, or detecting the declared
pattern only) would be development on them, and a claim would need recordings no method here
has read.

### Development after the audit (spent recordings, not an evaluation)

The harness's `--still-stereo` variant fixes the two points: the right image is the nearest
inside the still interval, and the frame is the one with the most boards that triangulate.
It ran once on the five spent recordings
([records](../assets/kitti_board_camera_lidar/development/run.json)). Every day converged,
from 5 to 10 boards (7 x 5, 11 x 7 and 15 x 5 inner corners, all with 0.099-0.100 m squares
by stereo), and none was within the acceptance gates:

| Day | Boards used | Rotation error | Position error | Axes reported observed |
| --- | --- | --- | --- | --- |
| 2011_09_26 | 9 | 1.56 deg | 0.22 m | x |
| 2011_09_28 | 10 | 0.89 deg | 0.07 m | none |
| 2011_09_29 | 10 | 1.40 deg | 0.03 m | x, z |
| 2011_09_30 | 5 | 1.40 deg | 0.18 m | none |
| 2011_10_03 | 9 | 1.76 deg | 0.28 m | none |

- **The stds were honest.** Leave-one-board-out stds of 0.3-2.1 deg and up to 0.36 m left
  almost every axis unobserved, and no axis reported as observed failed the check rule
  against KITTI's calibration: the method did not give a wrong value as a measurement.
- **Some LiDAR planes are not the board's.** Per board, the LiDAR plane differed from the
  camera's by up to 9 deg and 0.5 m, where the ACSC boards held in free space agreed within
  3.1 deg and 0.1 m. Most of KITTI's boards hang on walls, doors and stands, so the crop
  around a board also holds the wall or floor around it, and the RANSAC plane can be that
  surface; many boards also share the wall's normal, which leaves little variety in the
  normals.

A board mounted on a wall is therefore a poor target for this method as it stands: the
LiDAR plane would need the board's own edges (or a board standing free of the wall), not a
plane fit around it. That would be development on these spent recordings again; it is not
pursued further here.

**A second development attempt made it worse, and is not merged.** Keeping only each
board's inner-corner area after the first pass (where the board hides what is behind it)
fixed a synthetic scene with a denser wall 12 cm behind every board, where the margin crop
took the wall (z 14 cm off), and left the ACSC result about where it was (x 11.3 instead of
13.3 cm). On the five spent KITTI recordings it made every day worse: 1.9-4.6 deg and
0.06-0.81 m from KITTI's calibration, with boards up to 24 deg and 1.1 m off
([records](../assets/kitti_board_camera_lidar/development_inner_crop/summary.json); code on
the unmerged branch `feat/board-inner-crop`, commit `87bff4f7`). The stds grew with the
errors, and still no axis reported as observed failed the check rule. The larger error is
earlier: from the axis permutation, the first pass's generous crop already takes wrong
planes on these scenes, and a narrower crop around a misplaced board loses it. A first
pass robust to that (the rotation from the normals alone, or several starts) would be the
next development step; with no unseen board recordings with a reference at hand, it is
not taken here.

## Next

A real capture tilted up and down as well, to observe the vertical translation and all
three rotations. `calibrex check --board` judges a deployed camera-LiDAR transform with the
same method ([tutorial](../tutorials/calibrex_check.md#camera-lidar-with-a-checkerboard-board));
on real data it has run only on the ACSC sample above.
