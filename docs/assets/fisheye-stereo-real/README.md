# Real fisheye stereo photographs

The browser's **Fisheye stereo** demo uses real indoor photographs, not rendered
images. Three pairs (`000`, `014`, `027`) come from OpenCV extra's
`testdata/cv/cameracalibration/fisheye/calib-3_stereo_from_JY` dataset.

## Attribution and source records

Credit: **OpenCV contributors / calib-3_stereo_from_JY**. Original image bytes,
the calibration test source (including its upstream license notice), and the
board-coordinate file are preserved in `source/`. These third-party materials
retain their upstream terms; Calibrex does not relicense them under its own
Apache license. OpenCV extra supplies no separate image-specific license file.
See [OpenCV contribution terms](https://github.com/opencv/opencv/wiki/How_to_contribute)
and the [original test data](https://github.com/opencv/opencv_extra/tree/d244584fd02b509f63db08db89b2d38c97ce990d/testdata/cv/cameracalibration/fisheye/calib-3_stereo_from_JY).

[sources.json](sources.json) binds each original file to an immutable upstream URL
and SHA-256. [calibration.json](calibration.json) transcribes the distinct left and
right intrinsics, distortion, and right-from-left transform from
`TEST_F(fisheyeTest, stereoCalibrate)` in the preserved OpenCV test source.

## Processing and limits

Run `PYTHONPATH=src python tools/build_real_stereo_demo.py` from the repository
root. The generator resizes the original 1280×800 images to 640×400, transforms
intrinsics with the same pixel-centre convention, rectifies through Calibrex's
native camera implementations, and runs its image-only ZNCC patch matcher. RGB is
preserved for display; matching uses grayscale. Board corner coordinates and
known object geometry do not guide matching. There is no rendered scene.

The baseline is approximately 99.4 mm, following the reference translation and
upstream board-coordinate scale (`0.0244`, interpreted as metres). Displayed Z
uses that calibrated scale: `Z = f × B / disparity`. It is depth along the
rectified left camera's Z axis, not ray distance.

**There is no independent depth truth or depth-accuracy score.** Vertical mismatch
is a correspondence diagnostic, not a distance error. Repetitive checkerboard
patterns can match the wrong square; occlusions and weak texture also produce
errors. Integer disparities and outliers are retained. No uniqueness or
left/right consistency filter is applied. Colours clip to 0.3–3.0 m without
changing the displayed point measurements.

These photographs belong to the calibration dataset used to obtain the supplied
reference parameters. This is a saved demonstration, **not a held-out benchmark
or a live calibration run**. KB4 uses the upstream calibration. **MEI and Double
Sphere are newly fit to the measured board corners**, and their real-image
rectification/depth demo uses capture `000`. They are not alternative labels on
the KB4 projection.

Run `PYTHONPATH=src python tools/fit_real_fisheye_models.py` to regenerate these
model calibrations. The native projectors fit 27 captures jointly with board
poses and the stereo transform. Every fifth capture is excluded from the final
fit; those poses are estimated from left corners only before evaluating right
reprojection. Full fitted board poses and split IDs are retained in
`calibration_mei.json` and `calibration_ds.json`, so reported residuals can be
recomputed from the original 2D/3D observations without rerunning optimization.

The supplied all-capture KB4 calibration initializes intrinsics and board poses.
Consequently these fit-excluded diagnostics are **not a fully independent
benchmark**. No metric depth truth or general lens-accuracy claim is made. All
models use a common output focal length for the comparison; each uses its own
fitted stereo transform and baseline. Raw photographs remain identical.

The [generated receipt](../../app/samples/fisheye_real_demo.json) binds the
generator, native implementation, original photographs and calibration to their
SHA-256 values. Each frame is fetched only when selected in the browser.

Before converting rectified floating-point intensities to shared display/matching
8-bit pixels, the producer rounds to eight decimal places and clips to 0–255.
This removes sub-1e-8 numerical differences near integral intensities across
NumPy/SciPy paths, which otherwise change gradient thresholds and seeded patch
selection. Camera maps and disparity/depth equations remain unchanged. Tests
retain exact pixel and correspondence comparisons against the saved receipts.
