# Fisheye stereo: rectify and recover depth

The [browser demo](../app/check.html) shows **real photographed fisheye stereo**
from OpenCV's `calib-3_stereo_from_JY` dataset. Select **Fisheye stereo**, then a
camera model and photographed pair: the colour raw images, native rectification and sparse
matched depth appear together. The other tabs show recorded LiDAR–LiDAR, IMU
deskew and Camera–IMU demonstrations, plus an
[interactive real Camera–LiDAR projection](../assets/camera-lidar-live/README.md).

KB4 uses the supplied calibration. MEI and Double Sphere use new native fits to
the measured board-corner observations; capture `000` is available for both.
Their parameters, fit-excluded right reprojection diagnostics and initialization
limits are recorded in the [real-data receipt](../assets/fisheye-stereo-real/README.md).
There is no independent depth truth for any of the three models.

Click a depth dot, move the **Inspect match** slider (arrow keys supported), or
select **Next match** to highlight the same correspondence in both rectified
images. The inspector shows its measured disparity, vertical mismatch and
estimated depth, including the numerical `Z = f × B / d` calculation. The dots
and measurements come from the native image matcher; they are not scene-truth
correspondences. Depth means Z in the rectified left camera, not ray distance.

## Supported stereo camera models

| Kalibr `camera_model` | `distortion_model` | `intrinsics` order | Distortion coefficients |
|---|---|---|---|
| `pinhole` | `equidistant` | `fx, fy, cx, cy` | KB4 `k1, k2, k3, k4` |
| `omni` | `radtan` or `none` | `xi, fx, fy, cx, cy` | `k1, k2, p1, p2`, or empty |
| `ds` | `none` | `xi, alpha, fx, fy, cx, cy` | empty |
| `pinhole` | `radtan` or `none` | `fx, fy, cx, cy` | `k1, k2, p1, p2 [, k3]`, or empty |

These models are supported by the stereo rectification API and Kalibr pair reader.
They do **not** automatically add model support to every bag-check Camera–IMU or
Camera–LiDAR adapter. Unsupported model/distortion combinations are rejected.

The pair must be adjacent in the Kalibr chain. `T_cn_cnm1` maps left-camera
coordinates into the right camera. Both images must already have the resolution
and intrinsics declared by the calibration. This code rectifies a supplied stereo
calibration; it does not estimate intrinsics or fix an incorrect stereo transform.

## Use the native API

```python
import numpy as np
from calibrex.evaluation.fisheye_stereo import (
    read_kalibr_fisheye_pair, rectify_fisheye_pair, vertical_offset_samples,
)

left_camera, right_camera, right_from_left = read_kalibr_fisheye_pair("camchain.yaml")
rect = rectify_fisheye_pair(left_camera, right_camera, right_from_left)
left_rectified = rect.rectify_left(left_gray)
right_rectified = rect.rectify_right(right_gray)

# Sparse image-only ZNCC matches, with positive horizontal disparity.
samples = vertical_offset_samples(
    [(left_rectified, right_rectified)], min_disparity_px=1,
)
rows_px, columns_px = samples[:, 1], samples[:, 2]
disparity_px = samples[:, 3]
depth_m = rect.stereo.base_px_m / disparity_px
```

Depth is **Z in the rectified left frame**, not distance along a fisheye ray.
Horizontal disparity is integer-valued in this sparse matcher, so distant depths
have coarse quantization. Untextured, occluded or poorly matching regions have
no trustworthy depth. A black region in the resampled images may be outside the
original image/model domain. Reduce `focal_scale` to widen the output pinhole view,
then check how much of that view has valid image coverage.

The `MeiCamera` and `DoubleSphereCamera` projection functions return a validity
mask for their injective model domains; their inverse functions return NaN for
invalid pixels. KB4 retains its existing forward-hemisphere validity convention.
The rectified view is a common pinhole overlap, not a spherical depth map.

## What the real demo demonstrates

The three indoor pairs use separate recorded left/right KB4 intrinsics and
distortion with a baseline of approximately 99.4 mm. Native Calibrex rectification
and image-only ZNCC matching produce the displayed results. No rendered images or
known scene depth enter the real-data path. These are saved results, not live
browser optimization. Source images, upstream calibration, attribution, processing
and scale details are in the [real-data receipt](../assets/fisheye-stereo-real/README.md).

There is **no independent depth truth**, so the real demo reports no depth
accuracy percentage. Vertical mismatch is an image-space correspondence
diagnostic. Repeated board patterns, weak texture and occlusions can yield wrong
matches; no uniqueness or left/right consistency filter is applied. Colour clips
to 0.3–3.0 m while the selected point's numerical depth remains unclipped. These
are calibration-set photographs, not held-out accuracy measurements.

Rebuild with `PYTHONPATH=src python tools/build_real_stereo_demo.py`. Its receipt
binds original images, generator, native implementation and calibration with
SHA-256; only the selected pair's display asset is fetched by the browser.

## Synthetic model regression checks

Separate synthetic fixtures test KB4, MEI and Double Sphere using surfaces at
3, 6 and 12 m and a 0.30 m baseline. Raw images
are rendered from each camera model, rectified by Calibrex, then passed to the
native image patch matcher. True depths are used **after matching** to measure
error, never to guide the correspondence search. Samples that pass matching are
retained, including occlusion mistakes and integer-disparity quantization.

This demonstrates geometric consistency and software behavior; it is not a real
fisheye accuracy benchmark or a guarantee for every lens. Real-data evaluation is
still needed for new lenses, weak texture, limited overlap and poorly constrained
extrinsics. Runtime support requires NumPy/SciPy; OpenCV is not required for this
rectification path.

Rebuild those fixtures with `python tools/build_method_demos.py`. Their receipts bind the
generator and native source files with SHA-256. The scene is synthetic, while the
four other method demos retain their original real-data provenance and licenses.
