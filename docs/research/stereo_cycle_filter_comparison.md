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
