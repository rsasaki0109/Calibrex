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


## Results

Source and protocol were frozen at `bae08665`. All 12 frame preparations and
16 fits completed. **Both frontends reject all eight cases at the 15-iteration
limit.** All returned matrices exactly equal the saved D2D initializers. Returned
pipeline recovery remains 4/8 in both arms; neither creates an accepted-stage
recovery. The old priority run had 1/8 accepted, but it used the historical depth
formula; the corrected-projection control is a distinct run, not a relabeling of
that historical result.

### Translation refinement

Raw candidate translation distances to the dataset reference are in metres;
none of these candidates is accepted. Re-rectification improves five and worsens
three relative to the corrected-projection control. Rotation remains fixed.

| Recording / seed | Projection-only raw | Re-rectified raw |
| --- | ---: | ---: |
| 0005 positive | 0.1645 | 0.1734 |
| 0005 negative | 0.1613 | 0.1537 |
| 0002 positive | 0.0876 | 0.0656 |
| 0002 negative | 0.0365 | 0.0236 |
| 0009 positive | 0.1452 | 0.1439 |
| 0009 negative | 0.1705 | 0.1740 |
| 0011 positive | 0.2647 | 0.3244 |
| 0011 negative | 0.1236 | 0.1173 |

### Frontend consistency

All selected sets have 2,000 pairs per frame, but their membership differs.
Median signed z differences below are LiDAR minus stereo at reference, in metres.
They must not be interpreted as paired per-point improvements or unbiased depth
accuracy. Compared with the previous fixed-pair projection audit, this control
recomputes normals, gating and selection using the corrected depths.

| Recording | Frame | Projection-only median z | Re-rectified median z | Control valid pixels | Re-rectified valid pixels |
| --- | --- | ---: | ---: | ---: | ---: |
| 0005 | frame-38 | -0.1150 | -0.1004 | 331917 | 325536 |
| 0005 | frame-76 | -0.1003 | -0.0872 | 371039 | 361931 |
| 0005 | frame-114 | -0.0994 | -0.0871 | 343090 | 331976 |
| 0002 | frame-93 | -0.0596 | -0.0553 | 343379 | 323036 |
| 0002 | frame-187 | -0.0659 | -0.0654 | 342939 | 322547 |
| 0002 | frame-281 | -0.0697 | -0.0652 | 343826 | 323083 |
| 0009 | frame-111 | -0.0561 | -0.0298 | 360514 | 348230 |
| 0009 | frame-223 | -0.0639 | -0.0654 | 352472 | 341257 |
| 0009 | frame-334 | -0.0882 | -0.0841 | 343418 | 330402 |
| 0011 | frame-58 | -0.0714 | -0.0592 | 324657 | 310973 |
| 0011 | frame-116 | -0.1186 | -0.1028 | 307349 | 300244 |
| 0011 | frame-174 | -0.0890 | -0.0786 | 337666 | 326891 |

Median z residual magnitude decreases in 11/12 distinct frames. Median absolute
point-to-plane residual decreases in 7/12. Valid pixel counts decrease in all
12 frames, consistent with changed field of view, support masking, interpolation
and matching. Thus the frontend tradeoff is not a uniform accuracy improvement.

**Decision:** retain the corrected projection frontend and do not promote
re-rectification to a default. Better descriptive depth statistics did not deliver
an accepted calibration under the frozen protocol. Vertical alignment alone is
not established as the cause of the remaining failures; resampling changes the
input population and stereo matching as well.

The next useful step should test the stereo measurement model, including
left/right correspondence consistency and disparity-dependent residuals, before
another optimizer change. Preserve the difficult 0011-positive case, where the
raw candidate worsens from 0.2647 to 0.3244 m. Any new frontend must demonstrate
both usable support and downstream calibration improvement across all cases.

[Complete report](../assets/stereo_rerectification_20261029/comparison.json) and
[schema](../assets/stereo_rerectification_20261029/comparison.schema.json) retain
source/input hashes, generated depth hashes, rectification matrices, settings,
valid counts, per-frame checks and all 16 fit traces. New depth arrays are in the
separate local experiment workspace; no original arrays or verdicts changed.
The publisher verifies exact inventory, frozen rotations and all fallback poses.
