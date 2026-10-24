# Fisheye stereo: rectify and recover depth

The [browser demo](../app/check.html) compares **KB4, unified/MEI and Double Sphere**
on a known synthetic scene. Select **Fisheye stereo**, then a model: the raw pair,
native rectification and sparse matched depth are shown together. The other tabs
show recorded LiDAR–LiDAR, IMU deskew, Camera–IMU and Camera–LiDAR demonstrations.

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

## What the demo validates

The synthetic scene has surfaces at 3, 6 and 12 m and a 0.30 m baseline. Raw images
are rendered from each camera model, rectified by Calibrex, then passed to the
native image patch matcher. True depths are used **after matching** to measure
error, never to guide the correspondence search. Samples that pass matching are
retained, including occlusion mistakes and integer-disparity quantization.

This demonstrates geometric consistency and software behavior; it is not a real
fisheye accuracy benchmark or a guarantee for every lens. Real-data evaluation is
still needed for new lenses, weak texture, limited overlap and poorly constrained
extrinsics. Runtime support requires NumPy/SciPy; OpenCV is not required for this
rectification path.

Rebuild the demo with `python tools/build_method_demos.py`. The receipts bind the
generator and native source files with SHA-256. The scene is synthetic, while the
four other method demos retain their original real-data provenance and licenses.
