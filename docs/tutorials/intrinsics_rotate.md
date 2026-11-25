# Camera intrinsics without a target

`calibrex intrinsics rotate` estimates a camera's intrinsics from a recording of the camera
turned in place: no checkerboard. When a camera only rotates, where a scene point appears
in one frame fixes where it appears in every other frame through the intrinsics and the
rotation alone, so panning, tilting and rolling the camera over a textured, static scene
determines the focal lengths, the principal point and the lens distortion. The result is a
Kalibr camchain entry (`pinhole` / `radtan`: `k1 k2 p1 p2`, or `equidistant`: `k1..k4`
with `--distortion equidistant` for fisheye lenses), like
[`calibrex intrinsics guide`](intrinsics_guide.md) writes.

## Run it

From a ROS bag, a video, or a live camera (recorded for `--seconds`, default 20):

```bash
calibrex intrinsics rotate --bag my_bag --topic /camera/image_raw
calibrex intrinsics rotate --video turning.mp4
calibrex intrinsics rotate --camera 0 --resolution 1280x720 --seconds 20
```

For a ROS 1 `.bag` with raw `sensor_msgs/Image` messages or a ROS 2 bag, use
`--start-seconds 12 --seconds 10` to use the interval from 12 to 22 seconds after
the topic's first image stamp.

With a live camera a window guides the turning (below) and the calibration starts as soon
as the turns are enough; `--no-window` records for `--seconds` without it.

For a fisheye lens add `--distortion equidistant`. On a simulated 110 deg fisheye
(480 x 360, f = 235 px, Kannala-Brandt k1..k4) turned for 8 s, the focal lengths came
within 0.2 %, the principal point within 0.5 px, and the fitted model put pixels in the
image corners within 0.5 deg of their true directions. The guide window then reads the
turns through an ideal fisheye of 0.45 times the image width.

It needs OpenCV (`pip install 'calibrex[opencv]'`; the live window needs `opencv-python`,
not the headless build). `--output` (default
`intrinsics_camchain.yaml`) and `--camera-name` (default `cam0`) name the camchain; `--json`
prints the result as JSON.

### Stereo

A stereo rig turned in place gives both cameras' intrinsics and their relative rotation.
Pass the right camera as a second topic, a second video, or a side-by-side stream:

```bash
calibrex intrinsics rotate --bag my_bag --topic /left/image_raw --right-topic /right/image_raw
calibrex intrinsics rotate --video left.mp4 --right-video right.mp4
calibrex intrinsics rotate --video stereo.mp4 --side-by-side --baseline-m 0.12
calibrex intrinsics rotate --camera 0 --side-by-side --seconds 20
```

The camchain gets `cam0` (left) and `cam1` (right). Both cameras turn together, so between
any two moments the right camera's rotation is the left camera's seen in the right
camera's axes; aligning the two rotation sequences (keyframes matched by time, left
orientations interpolated) gives the relative rotation. Bag topics are paired by their
header stamps, so the cameras must share a clock. Two separate live cameras cannot be
recorded in step here: record them to a bag or to videos.

The baseline is harder. Its **length** is not observable from rotation, so `T_cn_cnm1` is
written only with `--baseline-m`. Its **direction** comes from features matched between
simultaneous left and right images, and a rotation error tilts it by about the rotation
error over the parallax angle (in a narrow field a rotation about the camera's y axis and a
translation along x look alike). It is reported only when the matches' median parallax is
at least 2 px and ten times the rotation's error in pixels; otherwise the command says so
and leaves `T_cn_cnm1` out. On a simulated 10 cm rig turned in a room 16 m across, the
intrinsics came within 0.4 % (focal) and 4 px (principal point), the relative rotation
within 0.7 deg (0.2 deg in a room 40 m across), and the parallax (under 1 px) was too small
for a direction. With the true rotation and objects 2-3.5 m away the direction came within
3 deg, but a rotation 0.5 deg off is already too coarse for it. Use a checkerboard when the
full stereo extrinsics matter.

Turning a rig about its middle also offsets each lens by half the baseline from the rotation
centre, which lengthens the focal lengths by about 0.8 (b/2)/D: face a distant scene.

## Rolling shutter

Most webcams and phone cameras read the frame out row by row, so a camera turned briskly
sees each row a little later, turned a little further. Ignoring that bends the fit:
`--rolling-shutter` also fits the readout time (first row to last), turning each row's
view on by the keyframe's angular velocity times the row's delay. The camchain then
carries `readout_s`, which Calibrex's camera pairs (camera-imu, camera-focal) read.

On a simulated camera (320 x 240, f = 280 px) read out in 30 ms and turned up to about
50 deg/s, the fit gave a readout of 31.1 ± 0.2 ms, the focal length within 0.05 % and the
principal point within 0.7 px; without `--rolling-shutter` the focal length came out 1.5 %
long and the principal point 16 px off. The fit runs the global-shutter fit first, then
refines with the readout (about twice the time). On RobotNativeEngine's simulated camera
read out in 30 ms (in 24 bands) and turned more slowly, it estimated 25.9 ± 0.5 ms and
moved the principal point from 0.9 px to 0.5 px off; there the readout cost little.

## How to record

- **Turn the camera about its lens.** This is what matters most. A camera on a tripod or a
  pan-tilt head is ideal; by hand, hold it close to the lens and rotate it, not your arm.
- **Face a static, textured scene, as far away as you can.** Outdoors, a street or a
  building façade tens of metres away; indoors, a cluttered room. Nothing should move.
- **Pan, tilt and roll**, each by 20 deg or more, for 10-20 s. Turning about one axis only
  leaves the focal length undetermined; the command warns when fewer than two axes turned
  20 deg.
- Turn smoothly: features are tracked frame to frame and dropped when tracking them back
  misses by more than 0.3 px.

## The guide window

`--camera` shows the live image with what to do next, and `--video --guide` shows the same
over a recording (`--no-window --record annotated.avi` writes it to a video instead):

- **Point the camera at a static scene with more texture**: fewer than 60 features are
  tracked.
- **Slower**: the features moved more than 6 % of the image width in one frame, or half
  of them were lost.
- **Pan**, then **Tilt**, then **Roll**, with how far the camera has turned about that axis
  and the 25 deg it should reach. A bar per axis fills as you turn and turns green.
- **Done**: all three axes turned 25 deg over at least 20 keyframes. The live camera then
  stops and calibrates; `s` calibrates earlier, `q` quits.

The guide turns the image motion between keyframes into a rotation with a guessed focal
length (0.8 times the image width, about 64 deg across), so its angles are approximate: a
guess 20 % off reads pan and tilt 20 % off, while roll is exact. The 25 deg target leaves
room above the 20 deg the calibration needs. On a simulated camera (280 px focal length,
k1 = -0.12) a person sweeping each axis as told reached done in about 80 frames (8 s at 10 fps),
pan and tilt read within 10 % of the truth, roll within 1.5 deg, and the calibration of
those frames gave the focal length within 0.3 %.

The guide cannot tell whether you turn about the lens: the lens distortion bends the image
motion as much as a few centimetres of translation, so it only reminds you.

## What it reports

The camchain, the intrinsics with their stds, the transfer error (how far a tracked
feature lands from where the fitted rotation and intrinsics put it, in pixels: its RMS and
its median), the number of keyframes, how far the camera turned about its x (tilt), y (pan)
and z (roll) axes, and warnings.

The command warns when the median transfer error is above 1 px: the scene moved or the
features are tracked poorly. It judges the median, not the RMS, because the RMS follows the
tail of poorly tracked features: on RobotNativeEngine's simulated room, rendered one ray
per pixel so its edges alias, a camera that only turned gave a median of 0.48 px and an
RMS of 1.13 px. A translation of the camera does not raise either (see below).

The stds come from the fit's Jacobian, scaled for each tracked point entering several
keyframe pairs. They do not include the bias below.

## The limit: translation

A hand-held camera also translates as it turns, and the intrinsics take up the parallax of
that translation **without raising the residual**, so the data alone cannot show it. On a
simulated camera (480 x 360, barrel distortion k1 = -0.12, a textured room 8 m across):

| Lens offset from the rotation centre | Scene distance | Focal-length error |
| --- | --- | --- |
| 0 (turned about the lens) | 4 m | -0.08 % (principal point within 0.2 px, k1 -0.1202) |
| 5 cm | 4 m | +0.9 % |
| 5 cm | 2 m | +1.9 % |
| 15 cm | 4 m | +2.8 % |
| 15 cm | 10 m | +1.1 % |

The focal length comes out too long by about 0.8 times the offset over the distance. For
0.5 % (enough to start a camera-LiDAR or camera-IMU calibration), turn the camera about its
lens, or keep the offset under about 1/150 of the scene distance. When the focal length
must be known to 0.1 %, use a checkerboard ([`calibrex intrinsics guide`](intrinsics_guide.md)).

## Real-data results (TUM-VI)

The [TUM-VI dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
provides these 512 × 512 global-shutter fisheye recordings. The audit estimates
each camera independently with `--distortion equidistant`. It uses OptiTrack rig
poses to choose a 10 s window with rotation about at least two axes and the
smallest measured rig displacement. Only the image stream enters the fit. The
[published Kalibr camchain](../assets/tumvi_camera_focal/camchain.yaml) supplies
the reference values for scoring.

The [pre-registration](../benchmarks/tumvi_intrinsics_rotate_preregistration.yaml)
was committed as `3112f7df` after development on `room1`. It requires both
absolute focal errors to be at most 5% and both absolute principal-point errors
to be at most 5 px. At least five of the ten evaluation cases must pass.
Inconclusive estimates contribute zero passes. The distortion coefficients are
reported as signed differences. The earlier camera-focal and camera-IMU checks
had used the evaluation recordings.

In the table, focal errors are percentages of the reference focal lengths.
Principal-point and coefficient errors are estimate minus reference.
The development verdicts apply the evaluation gates retrospectively.

| Split | Sequence | Camera | Δfx (%) | Δfy (%) | Δcx (px) | Δcy (px) | Δk1 | Δk2 | Δk3 | Δk4 | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Development | room1 | cam0 | +23.88 | +22.12 | +0.26 | -2.74 | -0.0408 | -0.0039 | +0.0018 | -0.0027 | fail |
| Development | room1 | cam1 | +22.58 | +21.24 | -0.44 | -1.89 | -0.0410 | -0.0005 | +0.0008 | -0.0026 | fail |
| Evaluation | room2 | cam0 | +28.66 | +28.48 | +3.01 | +0.84 | -0.0518 | -0.0026 | +0.0062 | -0.0032 | fail |
| Evaluation | room2 | cam1 | +33.99 | +29.32 | +5.88 | +1.34 | -0.0517 | -0.0015 | +0.0071 | -0.0061 | fail |
| Evaluation | room3 | cam0 | +17.92 | +17.81 | +2.38 | -1.88 | -0.0291 | +0.0052 | -0.0027 | -0.0024 | fail |
| Evaluation | room3 | cam1 | +14.58 | +13.91 | +1.40 | -6.14 | -0.0525 | +0.0269 | -0.0177 | +0.0020 | fail |
| Evaluation | room4 | cam0 | +34.61 | +35.19 | +3.96 | -2.99 | -0.0099 | -0.0071 | +0.0077 | -0.0155 | fail |
| Evaluation | room4 | cam1 | +30.65 | +32.71 | +3.22 | -0.63 | -0.0167 | -0.0063 | -0.0028 | -0.0077 | fail |
| Evaluation | room5 | cam0 | +15.19 | +14.87 | +3.12 | -4.84 | -0.0177 | +0.0032 | +0.0241 | -0.0126 | fail |
| Evaluation | room5 | cam1 | +11.70 | +13.38 | -2.60 | -3.26 | -0.0419 | +0.0056 | +0.0240 | -0.0109 | fail |
| Evaluation | room6 | cam0 | +36.49 | +36.95 | +10.98 | -0.53 | -0.1020 | -0.0152 | +0.0337 | -0.0091 | fail |
| Evaluation | room6 | cam1 | +45.71 | +46.14 | +12.20 | +8.31 | -0.1716 | +0.1103 | -0.0736 | +0.0218 | fail |

**Zero of ten evaluation cases passed.** All ten produced finite estimates and
failed the focal gate. Focal estimates were +11.70% to +46.14% above Kalibr.
Four cases also exceeded the principal-point gate. The full estimates, exact
errors, commands, and provenance are in the
[development JSON](../assets/tumvi_intrinsics_rotate/development/room1.json) and
the evaluation JSONs ([room2](../assets/tumvi_intrinsics_rotate/evaluation/room2.json),
[room3](../assets/tumvi_intrinsics_rotate/evaluation/room3.json),
[room4](../assets/tumvi_intrinsics_rotate/evaluation/room4.json),
[room5](../assets/tumvi_intrinsics_rotate/evaluation/room5.json),
[room6](../assets/tumvi_intrinsics_rotate/evaluation/room6.json)).

| Sequence | Window (s) | Rig displacement (m) | Median transfer, cam0 / cam1 (px) |
| --- | ---: | ---: | ---: |
| room1 | 20–30 | 0.987 | 2.13 / 2.00 |
| room2 | 134–144 | 1.453 | 3.19 / 3.26 |
| room3 | 128–138 | 0.622 | 2.97 / 2.63 |
| room4 | 100–110 | 1.049 | 4.67 / 3.83 |
| room5 | 18–28 | 0.304 | 1.48 / 1.14 |
| room6 | 110–120 | 1.460 | 3.54 / 3.46 |


### What limited agreement

- **Translation and parallax.** The qualifying windows still span 0.304–1.460 m
  of OptiTrack rig translation. These are the smallest measured displacements
  among windows that met the rotation requirement. The optical centers also
  move as the rig turns. The solver fits a rotation-only model, so it can absorb
  part of the parallax into the intrinsics. Even `room5`, with the smallest
  measured displacement, produces focal errors of +11.70% to +15.19%.
- **Tracking and image quality.** Every case exceeds the existing 1 px median
  transfer warning threshold. The highest median is 4.67 px on `room4/cam0`.
  These residuals measure disagreement with the fitted model. The audit did
  not independently measure the contribution from exposure changes, motion blur,
  or tracking loss.
- **Fisheye periphery.** The current inverse model caps rays at 1.45 rad
  (83.1°) from the optical axis. Evaluating the reference equidistant polynomial
  at that angle puts about 13% of the full 512 × 512 raster beyond the cap for
  each camera. This is a raster fraction and includes any pixels outside the
  usable image circle. The audit did not measure how many tracked features lie in
  that area or isolate its effect on the fit.

The ROS 1 reader and the bag window option were the only changes to the CLI
path. The estimator settings stayed at their normal defaults. A controlled
follow-up can turn a global-shutter fisheye around its lens on a tripod, repeat
the turn with a measured offset, and compare central and peripheral tracks.
That would separate the effects of parallax and the fisheye angle cap.

## How it works

1. Features are tracked with pyramidal Lucas-Kanade, forward-backward checked; a keyframe
   is taken whenever they moved 15 px (median).
2. The focal length and k1 are found together on a grid: with them, the undistorted
   correspondences between keyframes four apart must be related by a rotation. Searching
   the focal length alone is biased by tens of percent, because barrel distortion bends the
   image motion the way a shorter lens does (on the simulated camera, 490 px instead of
   420).
3. A least-squares fit refines the eight intrinsics and every keyframe's rotation from the
   transfer error: a feature seen in one keyframe is undistorted, rotated into another up
   to four keyframes later and projected with the distortion. The scene points are not
   unknowns, which keeps the fit small (eight intrinsics and three angles per keyframe)
   and quick to converge.

A model of the translation (a fixed lens offset from the rotation centre plus each point's
depth) was tried on the simulated camera and did not converge to the true offset, so the
translation remains a documented limit rather than an estimated quantity.

A gyro does not lift it either. Holding the keyframe rotations to an IMU's (with the
camera-IMU rotation and the gyro bias fitted alongside) left the focal length of the
camera whose lens is 15 cm off the rotation centre 3.4 % long, as without it: the fitted
rotations already matched the gyro's to 0.01 deg. Turning the camera moves the offset
lens sideways with the turn, so a scene point at distance Z moves in the image by the
rotation's f θ plus the translation's f L θ / Z: f (1 + L/Z) θ. Where the scene's depth
varies little that is a longer focal length and the right rotation, which no rotation
measurement can tell apart.
