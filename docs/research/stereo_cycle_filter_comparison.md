# Fixed stereo round-trip mask comparison

## Frozen development protocol

Apply the previously fixed independent 1 px reverse-consistency mask to all 12
re-rectified frames, and run translation refinement for all eight saved D2D
initializers. No disparity cutoff, iteration count, uncertainty setting, sampling
rule or acceptance threshold changes. Reference calibration is used only for
scoring after each fit, never to form the mask or optimize translation.

Reproduce the unmasked depth exactly against its archived hash. In newly rectified
coordinates retain only finite reverse consistency <=1 px, set remaining depth
to NaN, then restore to original-left optical z with the same nearest-neighbor
rule. Assert every retained depth equals the unmasked measurement. Recompute
surface normals with the same five-pixel validity rule; report depth and valid
surface counts separately. Do not fill holes or interpolate missing measurements.

The control is the archived unmasked re-rectification run (`bae08665`), with its
source hashes verified unchanged and the same initial matrices. Compare all raw
candidates, accepted results, unchanged-initializer fallbacks and support. Save
new arrays in a separate workspace with hashes. A diagnostic reduction in error
tails does not establish downstream benefit.

Commit code and this protocol before execution. All inputs are spent data; this
is not a fresh audit. Neither default frontend nor historical verdict changes.
No repeated search for a more favorable mask threshold is planned.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/compare_stereo_cycle_filter.py \
  docs/assets/stereo_rerectification_20261029/comparison.json \
  /tmp/cycle-filter-depths /tmp/cycle-filter-comparison.json
```


## Results

Code and protocol were frozen at `0bb89aa4`. All 12 masked depth preparations
and all eight fits completed. Unmasked depths reproduced the archived arrays
exactly; every retained masked depth equals its unmasked value. No image,
reference transform or calibration gate was changed after examining outcomes.

- Accepted masked refinements: **0/8** (all `max_iterations`), same as unmasked.
- Returned pipeline recovery: **4/8**, solely through exact D2D fallback.
- Raw candidate translation improves in **5/8**, worsens in **3/8** relative to
  the archived unmasked re-rectified control. These candidates are all rejected.
- Every one of the 120 outer iterations retains **2,000 correspondences in each
  frame**. Non-convergence is not a failure to reach the minimum point count.

Translation columns below are distances to the dataset reference, in metres.
They are not independent ground-truth accuracy estimates.

| Recording / seed | Unmasked raw | Masked raw | Last masked step (mm) |
| --- | ---: | ---: | ---: |
| 0005 positive | 0.1734 | 0.1615 | 0.892 |
| 0005 negative | 0.1537 | 0.1466 | 0.814 |
| 0002 positive | 0.0656 | 0.0358 | 4.727 |
| 0002 negative | 0.0236 | 0.0307 | 20.066 |
| 0009 positive | 0.1439 | 0.1400 | 0.290 |
| 0009 negative | 0.1740 | 0.1670 | 0.281 |
| 0011 positive | 0.3244 | 0.4546 | 13.179 |
| 0011 negative | 0.1173 | 0.1485 | 11.786 |

The difficult 0011-positive candidate worsens from **0.3244 to 0.4546 m**.
This prevents selecting only favorable cases as evidence for the mask. Point
counts alone do not measure useful geometry, accuracy of normals or observability.

### Support cost

Counts are for the entire restored image, not just projected LiDAR points.
Surface pixels must satisfy the original five-pixel validity and depth-jump rules.
The earlier 90.5% round-trip statistic used a different projected-point population;
it does not predict these whole-image retention fractions.

| Recording | Frame | Depth pixels before | After | Surface pixels before | After |
| --- | --- | ---: | ---: | ---: | ---: |
| 0005 | frame-38 | 325536 | 274859 | 280891 | 235253 |
| 0005 | frame-76 | 361931 | 311419 | 328473 | 281925 |
| 0005 | frame-114 | 331976 | 295658 | 309019 | 274378 |
| 0002 | frame-93 | 323036 | 279955 | 290365 | 251909 |
| 0002 | frame-187 | 322547 | 280481 | 288688 | 250417 |
| 0002 | frame-281 | 323083 | 281300 | 290003 | 252161 |
| 0009 | frame-111 | 348230 | 298855 | 308351 | 262590 |
| 0009 | frame-223 | 341257 | 294294 | 310971 | 269077 |
| 0009 | frame-334 | 330402 | 291526 | 289334 | 255716 |
| 0011 | frame-58 | 310973 | 269660 | 277480 | 241366 |
| 0011 | frame-116 | 300244 | 264116 | 263858 | 234659 |
| 0011 | frame-174 | 326891 | 283608 | 292840 | 255803 |

Depth retention is **84.4–89.1%**, and valid-surface retention **83.7–88.9%**.
The mask reduces support without producing an accepted calibration. This does not
show that reverse consistency is generally useless; it shows that this fixed mask
is insufficient in the present translation-refinement pipeline.

### Decision

Do not promote this mask or any of the tested translation-refinement variants.
Keep the 15-iteration/0.1-mm acceptance rule and exact fallback. The sequence of
controls now covers fixed sampling, reference rotation/pose, corrected stereo
projection, re-rectification and reverse consistency. The projection formula bug
has a geometric fix, but the later frontend changes have not demonstrated a
usable calibration improvement.

Further adjustment of thresholds on the same eight cases is not the next useful
step. A broader comparison on other already-spent recordings and an alternative
measurement model is needed before revisiting this experimental stage. Preserve
0011-positive as a failure case and require both convergence and reference-error
improvement, rather than optimizing descriptive depth statistics alone.

[Complete comparison](../assets/stereo_cycle_filter_20261029/comparison.json) and
[schema](../assets/stereo_cycle_filter_20261029/comparison.schema.json) retain
all fit traces, source/input hashes, depth-output hashes, surface counts, raw
candidates and fallbacks. Original depth arrays and historical verdicts remain
unchanged. There is no fresh evaluation or SOTA claim from these experiments.
