# Stereo re-rectification development comparison

## Frozen protocol

Compare both frontends on all 12 distinct spent frames and eight saved D2D
initializers from `stereo_priority_20261029/plan.json`. Commit code and protocol
before reading the resulting scores; no post-result tuning.

- **Projection-only control:** reconstruct existing disparity from the historical
  depth and apply the corrected full-projection formula.
- **Re-rectified:** use the original grayscale images, their original K and the
  full camera-to-camera translation. OpenCV stereoRectify uses identity relative
  rotation (both input cameras already use rectified axes), zero-disparity flag,
  alpha=0 and unchanged image dimensions. Remap images with linear interpolation,
  run the same SGBM settings/version, and exclude invalid remap samples on both
  sides. Restore depth to original-left rays with nearest-neighbor sampling and
  optical-z conversion. Invalid pixels remain invalid.

Save new depth arrays separately, with hashes. Keep the original point clouds,
D2D initial transforms, point-priority sampling, uncertainty parameters (including
the recorded legacy baseline), 15 iterations, 0.1 mm convergence and rejection
rules fixed. This tests the translation refinement stage, **not** a full rerun of
D2D or a new calibration audit. References enter only after both fits for a case.

Report all 16 fits, returned versus raw candidate errors, convergence and original
D2D recovery. Per-frame gated surface residuals at reference are descriptive and
have different selected populations; report support and frontend valid-pixel counts.
These cannot alone establish depth accuracy. Re-rectification changes interpolation,
focal length and visible support along with epipolar alignment; causal attribution
to vertical alignment alone is not possible. No reference-fitted depth offsets,
new data, threshold relaxation or default promotion.

Tests verify epipolar rows for known points with a non-horizontal baseline, return
to original optical depth for a known plane, and preservation of invalid depth.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/compare_stereo_rerectification.py \
  docs/assets/stereo_priority_20261029 /tmp/rerectified-depths \
  /tmp/rerectification-comparison.json
```
