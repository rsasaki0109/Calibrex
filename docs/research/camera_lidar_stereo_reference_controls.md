# Reference-assisted stereo translation controls

## Frozen diagnostic protocol

Follow the [point-priority comparison](camera_lidar_stereo_priority.md) with the
same eight spent-data cases, settings and deterministic sampling. No fresh data
is read and no production behavior changes. Commit this protocol and driver
before running. Run two controls for every case without tuning:

1. **Reference rotation:** substitute the dataset reference rotation but retain
   the D2D starting translation.
2. **Reference pose:** start at the complete dataset reference pose.

These intentionally reference-assisted fits are diagnostic, **not deployable
calibration results or independent evaluation**. Compare convergence, raw
candidate distance to reference, displacement and the complete iteration traces.
Retain rejection and fallback. The original eight-case recovery verdict is not
revised. Each control keeps the 15-iteration/0.1-mm gates. Reference-pose controls
for the two seeds of one drive are duplicate inputs, not independent evidence.

At reference pose also report per-frame signed optical-z median (LiDAR minus
stereo), absolute-z 90th percentile and absolute point-to-plane median, in metres.
These describe only gated, selected matches; they are not unbiased depth errors.
They cannot isolate stereo bias from visibility, timing, dynamic objects,
reference uncertainty or surface-normal error. A nonzero reference-start drift
shows inconsistency with the reference under this model, not incorrect reference
calibration or proof of a single faulty sensor.

Sources/input hashes are checked against the frozen priority archive and saved
fits; the reference file hash is retained explicitly. All 16 controls, including
rejected candidates, must be reported. No thresholds are changed from results.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/diagnose_stereo_reference_controls.py \
  docs/assets/stereo_priority_20261029 /tmp/reference-controls.json
```
