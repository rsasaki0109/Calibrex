# Independent image-matching control: ZNCC versus SGBM

## Frozen development protocol

Challenge the shared SGBM reconstruction rather than changing how the optimizer
aligns it. Use six spent recordings, the middle of each archived three-frame
bundle, and both existing perturbations. All twelve cases run both frontends
with the same single-frame D2D defaults: 24 fits, no outcome-based retries.

Both methods use the original grayscale stereo images and the same full-baseline
re-rectification (alpha=0, zero disparity, original dimensions). Depth returns to
original-left coordinates using the existing nearest-depth restoration. The
existing SGBM settings/version remain fixed. Neither matcher receives LiDAR,
initial pose or reference calibration; stereo intrinsics/baseline are allowed.

The alternative is an independent image-only dense zero-normalized cross-
correlation implementation: 9x9 patches, all integer disparities 0..127, texture
standard deviation >=6, correlation >=0.9, best-to-nonadjacent-peak gap >=0.05.
Reject search boundaries and fit a bounded ±0.5 px parabola around an interior
peak. No semi-global regularization, reference-guided search, learned parameters
or interpolated hole filling is used. Require full patch remap support on both
images; depth remains in the common 0.1–80 m interval. This is a classical control,
not a claim to have implemented a current SOTA stereo network.

Synthetic tests check 12 px recovery under gain/offset, rejection of textureless
images, and ambiguity rejection on repeated patterns. Settings and code are
committed before real execution. The 128-disparity float32 score volume requires
about 240 MB at KITTI image sizes and is released between frames.

Report valid depth counts, every solver output/status and the unchanged <0.5 deg,
<0.20 m recovery gates. At reference, z-buffer projected LiDAR and compare depth
residuals only on the **same pixel/point population valid in both methods**, with
no residual-based depth gate. This common subset is selected and is not unbiased
sensor ground truth; right-camera occlusion, motion and reference error remain.
Report signed median and absolute-z p90 with counts. Reference scoring happens
after both fits, never during depth generation or optimization.

No production defaults or previous verdicts change. A depth statistic alone does
not establish downstream calibration improvement. Preserve failures and both seeds
for every drive. No fresh evaluation data or SOTA claim.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/compare_stereo_matchers.py \
  docs/assets/stereo_matchers_20261029/plan.json /tmp/matcher-depths \
  /tmp/matcher-comparison.json
```
