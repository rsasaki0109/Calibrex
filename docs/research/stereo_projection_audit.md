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
