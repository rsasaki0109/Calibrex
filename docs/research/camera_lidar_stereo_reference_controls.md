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


## Results

Protocol and driver were frozen at `84e87282`. All 16 controls completed with
unchanged settings and verified input/source identities. The 8 reference-pose
runs comprise **4 distinct starts repeated across two seed labels**; duplicate
results are exactly equal, and are not counted as independent trials.

### Substituting reference rotation

Only **1/8** converges (0009 negative). Thus substituting reference rotation does
not resolve the convergence problem in seven cases. This is a counterfactual
sensitivity check, not a measurement of the fraction caused by rotation error.
Initial translation, surface associations and optimization path still matter.

| Recording / seed | Status | Raw candidate translation distance to reference |
| --- | --- | ---: |
| 0005 positive | max_iterations | 0.0839 m |
| 0005 negative | max_iterations | 0.1320 m |
| 0002 positive | max_iterations | 0.0880 m |
| 0002 negative | max_iterations | 0.0800 m |
| 0009 positive | max_iterations | 0.1281 m |
| 0009 negative | converged | 0.1305 m |
| 0011 positive | max_iterations | 0.2838 m |
| 0011 negative | max_iterations | 0.1272 m |

### Starting from the complete reference pose

| Recording | Status | Raw displacement from reference | First step | Last step |
| --- | --- | ---: | ---: | ---: |
| 0005 | max_iterations | 0.1211 m | 29.63 mm | 1.315 mm |
| 0002 | max_iterations | 0.1241 m | 38.69 mm | 0.460 mm |
| 0009 | max_iterations | 0.1304 m | 35.61 mm | 0.282 mm |
| 0011 | converged | 0.0303 m | 23.43 mm | 0.066 mm |

Three of the four distinct reference starts reach the iteration limit; the fourth
converges about 3 cm away. Rejected output poses remain at their reference starts;
the displacement column intentionally reports **raw candidates**, not returned
poses. Nonzero drift even without D2D initialization error means fixing that error
alone is insufficient. It does not establish that the dataset reference is exact,
or identify one cause of the model/reference disagreement.

### Gated surface residuals at reference

All 12 distinct frames have negative median LiDAR-minus-stereo optical-z
residuals: **−0.119 to −0.056 m**. Their absolute-z 90th percentiles span
**0.885–2.134 m**, while median absolute point-to-plane residuals are
**0.023–0.060 m**. Each frame contributes 2,000 selected, already gated pairs.
These are descriptive statistics on a censored sample, not unbiased sensor
accuracy estimates. Large z differences can coexist with small plane residuals
on grazing surfaces; signed z consistency alone does not prove stereo bias.

The next investigation should audit stereo depth generation, camera projection
and reference coordinate conventions, and temporal alignment on these existing
bundles before another solver variation. Visibility, moving objects, normals and
reference uncertainty also remain unresolved. Increasing iteration limits or
subtracting this observed median is not justified by these controls.

No production solver or thresholds changed, and no fresh audit began. The earlier
priority experiment remains 1/8 accepted with no newly recovered case; these
reference-assisted controls do not revise that outcome.

[Complete control traces and per-frame checks](../assets/stereo_reference_controls_20261029/controls.json)
and [schema](../assets/stereo_reference_controls_20261029/controls.schema.json)
retain reference/input/source hashes, initial/reference matrices, raw candidates,
fallbacks and all iteration records. The archived publisher validates all 16
control identities, frozen rotations, exact fallbacks and duplicate reference
starts before generating these tables.
