# Stereo projection convention audit

Post-hoc development audit of the 12 distinct frames from the prior eight cases.
No original arrays, prior results or acceptance gates are overwritten. The audit
reconstructs recorded disparity from existing depth, holds the original selected
point/pixel pairs fixed, and measures only the effect of the projection formula.
It does not recompute image matches or claim improved calibration.

For equal rectified intrinsics K, write P_left = K[I|t_left] and
P_right = K[I|t_right]. With delta = t_right - t_left and d = u_left - u_right,
left-camera optical depth is:

```text
z_left = ((u_right - cx) * delta_z - fx * delta_x) / d
```

The old numerator `P_left[0,3] - P_right[0,3]` equals
`-fx*delta_x - cx*delta_z`; it omits `u_right*delta_z`. It is exact for a purely
horizontal baseline, but the actual KITTI projections have unequal z offsets.
The frontend is corrected to use the full horizontal projection relation. The
recorded baseline is now the actual x separation from K-inverted translation;
the legacy effective baseline is retained as a separate provenance field.

Synthetic known-depth projections test both horizontal and non-horizontal
baselines with nonzero camera origins. This fixes the conversion formula, **not**
SGBM's horizontal-row matching assumption; nonzero vertical offsets remain.

The audit verifies reference transforms independently from vendor calibration,
checks original input hashes and stereo timestamp identities, and reports
predicted vertical mismatch. Timing metadata describes modeled azimuth-based
point acquisition and OXTS interpolation, not measured per-point synchronization.
These checks cannot establish that motion compensation is accurate.


## Results

The fix and diagnostic were committed at `4f1ebd2c` before the numerical audit.
All 12 distinct frames completed. Raw source hashes and stereo timestamp identity
checks passed. The reference extrinsic reconstructed as
`R_rect_00 * T_cam0_lidar`, followed by the camera-02 offset `K^-1 P[:,3]`, agrees
with the saved reference to within **6.94e-18** in matrix entries. No camera-origin
omission was found in the reference transform.

The conversion bug is material for **2011-09-28**: left/right z offsets differ by
about 1.78 mm. At the selected pixels the corrected stereo depths are lower by a
median **3.25–3.59 cm**. For 2011-09-26 the z-offset difference is only about
0.016 mm, and depth corrections are below 0.34 mm in magnitude at the median.

Values below use the same originally selected pairs, not a new correspondence
search. Signed residual is LiDAR z minus stereo z at the dataset reference.

| Recording | Frame | Median depth correction (m) | Old median z residual (m) | Corrected median z residual (m) | Predicted median right-row offset (px) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0005 | frame-38 | -0.000326 | -0.115262 | -0.115001 | 0.183 |
| 0005 | frame-76 | -0.000309 | -0.100500 | -0.100310 | 0.155 |
| 0005 | frame-114 | -0.000332 | -0.099692 | -0.099374 | 0.133 |
| 0002 | frame-93 | -0.032478 | -0.089044 | -0.059214 | 0.288 |
| 0002 | frame-187 | -0.035704 | -0.096958 | -0.064815 | 0.265 |
| 0002 | frame-281 | -0.035855 | -0.101048 | -0.069509 | 0.260 |
| 0009 | frame-111 | -0.000246 | -0.056248 | -0.056067 | 0.167 |
| 0009 | frame-223 | -0.000262 | -0.064184 | -0.063944 | 0.191 |
| 0009 | frame-334 | -0.000231 | -0.088536 | -0.088241 | 0.167 |
| 0011 | frame-58 | -0.000326 | -0.071691 | -0.071430 | 0.150 |
| 0011 | frame-116 | -0.000324 | -0.119086 | -0.118629 | 0.152 |
| 0011 | frame-174 | -0.000323 | -0.089296 | -0.088965 | 0.154 |

The 0002 residual medians decrease in magnitude by roughly 3 cm, but remain
negative (−0.059 to −0.070 m). The other nine frames barely change. This proves a
projection-conversion inconsistency and its local effect; it does **not** show
improved end-to-end calibration or explain most residuals across all four drives.
No optimizer was rerun and no old verdict was updated.

### Timing and remaining assumptions

The source left/right timestamps match the saved headers exactly. Right images
precede left images by **0.473–0.641 ms**, within the existing 1 ms gate. This is
not zero time offset; motion can still produce image disagreement. Saved clouds
use OXTS position interpolation, SLERP orientation and modeled azimuth point times
relative to the scan header, transformed to the left camera timestamp. The
IMU/LiDAR transform is inverted to map LiDAR points into IMU coordinates and
inverted again after transport; that algebra is consistent with the frame names.
Existing synthetic alignment tests pass, but this audit does not reconstruct the
bags or validate the physical acquisition-time model.

There is also a predicted **0.133–0.288 px median vertical stereo mismatch** on
these selected points. The depth formula now accounts for the x/z projection
geometry, but horizontal SGBM still does not compensate this vertical offset.
This remains a concrete frontend hypothesis, alongside stereo matching error,
visibility, motion and reference uncertainty. Next compare a properly rectified
stereo frontend on the same spent frames, preserving a control and evaluating
both depth consistency and downstream calibration. Do not subtract a fitted
median from depth or loosen solver gates.

Artifacts: [audit](../assets/stereo_projection_audit_20261029/audit.json),
[schema](../assets/stereo_projection_audit_20261029/audit.schema.json).
Historical arrays remain unchanged. Future bundles record the corrected formula
and true horizontal baseline. Reproduce the diagnostic with existing bundles:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/audit_stereo_projection_geometry.py \
  docs/assets/stereo_priority_20261029 /tmp/projection-audit.json
```
