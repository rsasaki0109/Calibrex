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


## Results

Code and protocol were frozen at `ae028fc7`. All six image pairs produced both
frontends and all 24 D2D fits completed with `converged` status. Recovery is
**SGBM 2/12, ZNCC 1/12** under unchanged full-pose gates. Numerical convergence
alone again fails to indicate accurate calibration.

These are **re-rectified single-frame** SGBM controls. The preceding un-re-rectified
single-frame D2D experiment (3/12) and historical three-frame D2D (5/12) have
different inputs and remain separate, unchanged results.

| Recording / seed | SGBM rotation (deg) | SGBM translation (m) | SGBM recovery | ZNCC rotation (deg) | ZNCC translation (m) | ZNCC recovery |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 0005 positive | 0.9060 | 0.2257 | False | 1.0855 | 0.2716 | False |
| 0005 negative | 0.7027 | 0.2050 | False | 0.5238 | 0.2538 | False |
| 0002 positive | 0.2715 | 0.0395 | True | 1.0389 | 0.2550 | False |
| 0002 negative | 0.4826 | 0.3052 | False | 0.4270 | 0.1075 | True |
| 0009 positive | 1.7845 | 0.2872 | False | 1.3914 | 0.4758 | False |
| 0009 negative | 0.9000 | 0.5563 | False | 1.5082 | 0.2121 | False |
| 0011 positive | 0.3269 | 0.1487 | True | 0.5658 | 0.2032 | False |
| 0011 negative | 0.3706 | 0.5859 | False | 0.5056 | 0.2077 | False |
| 0013 positive | 0.7705 | 0.2670 | False | 1.0577 | 0.3175 | False |
| 0013 negative | 1.3022 | 0.3424 | False | 0.8849 | 0.1837 | False |
| 0015 positive | 0.7243 | 0.3288 | False | 0.8943 | 0.2939 | False |
| 0015 negative | 0.6583 | 0.2019 | False | 0.3264 | 0.2801 | False |

### Common-point depth comparison

The same z-buffered LiDAR returns and pixels are used for both methods in each
row. Median sign is LiDAR z minus stereo z; values are metres. Only frames with
both methods valid are represented, and the two seed labels share the same depth
comparison. These six rows, not twelve, are the distinct frame evidence.

| Recording | Common points | SGBM median | ZNCC median | SGBM absolute p90 | ZNCC absolute p90 | SGBM valid pixels | ZNCC valid pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0005 | 4061 | -0.3146 | -0.3473 | 3.600 | 4.059 | 361931 | 79850 |
| 0002 | 2951 | -0.0775 | -0.0331 | 3.527 | 4.097 | 322547 | 60049 |
| 0009 | 2409 | -0.1468 | -0.1277 | 7.460 | 8.166 | 341257 | 69631 |
| 0011 | 3127 | -0.2263 | -0.1875 | 6.102 | 6.789 | 300244 | 94078 |
| 0013 | 2734 | -0.0970 | -0.0664 | 2.704 | 2.654 | 323879 | 80847 |
| 0015 | 3171 | -0.2049 | -0.1764 | 5.380 | 5.944 | 296940 | 62817 |

ZNCC reduces median residual magnitude in 5/6 frames but worsens absolute p90
in 5/6. Its valid depth count is much smaller in every frame. All six common-point
median residuals remain negative for both methods. The observed disagreement
therefore survives changing the image matcher, but this is **not proof** of a
single bias source: shared image geometry, time alignment, occlusion, reference
uncertainty and matching populations still matter. A conservative local matcher
can also omit scene structure needed for calibration.

### Decision

Do not replace SGBM with this ZNCC frontend or promote any calibration result from
this experiment. There is no improvement in full-pose recovery, and median-only
depth summaries would hide worse tails and reduced support. No thresholds are
tuned, no inputs overwritten, and historical verdicts remain unchanged.

Stop treating another classical matcher or residual variant on this small spent
set as sufficient progress toward SOTA. The next useful external comparison is a
published modern stereo/cross-modal correspondence model with a pinned checkpoint,
license, training-data disclosure and reproducible runtime. Availability and
compute cost must be established before committing to a claimed evaluation.
The existing complete six-drive cases and failure-aware scoring should remain the
comparison denominator; no fresh held-out data should be exposed during setup.

[Complete evaluation](../assets/stereo_matchers_20261029/evaluation.json),
[schema](../assets/stereo_matchers_20261029/evaluation.schema.json),
[execution log](../assets/stereo_matchers_20261029/execution.log) and
[test log](../assets/stereo_matchers_20261029/tests.log) preserve all 24 optimizer
outputs, input/output hashes, settings, rectification matrices and paired depth
statistics. The publisher checks inventory, statuses and equal paired counts.
Sensor arrays remain in a separate workspace; no SOTA claim is made.
