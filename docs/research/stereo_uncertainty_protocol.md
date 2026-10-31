# Controlled stereo uncertainty normalization

Pre-registered 2026-10-30; no new real data. Test the range-dependent error
hypothesis after full-cloud real-data metric failure, without tuning on those
failures. Use the three existing independent-raycast sphere layouts and visible/
all-return populations. Reuse the seven-parameter bilinear metric solver as fixed
baseline. Do not mix edge sampling, scalar rescue or a new optimizer into this test.

## Measurement model and fixed candidate

For synthetic stereo use fx*baseline=120 px*m (fx240, baseline .5 m). At every
integer image pixel draw independent zero-mean Gaussian disparity error with
sigma .5 px, then invert d=120/z+noise into depth=120/d. Nonpositive disparity
becomes missing. Run two independently seeded realizations: base seeds 730 and
731, with +100*frame_index for the three frames. The same noise arrays are paired
across solvers, seed directions, visible/all-return populations and bias levels;
they are not independent repeats of each comparison. Also run a no-noise control.
Apply optical-depth offsets 0,+.25,-.25 m AFTER disparity inversion. This preserves
the existing bias sign and gate but is not a claim that real stereo has a constant
metric bias. Do not add LiDAR noise or gross disparity outliers in this first test.

Candidate sigma_z=max(.1 m, measured_z^2*.5 px/120 px*m). Normalize supported
corrected depth residual by .1/sigma_z BEFORE existing positive occlusion saturation
and soft-L1, keeping their numeric scales .2/.1. Thus scales operate in normalized
units; this deliberately changes the effective metric occlusion/loss scale at range.
This is a first-order uncertainty surrogate, not an exact likelihood: interpolation,
measurement-dependent weights and nonlinear inversion are approximations. The
candidate assumes .5 px even on no-noise controls. Weights use bilinearly sampled
measured depth, never ground-truth depth or the fitted bias. Unsupported penalty
stays 1. Raw sensitivity uses physical corrected residuals; robust sensitivity
uses the candidate objective. All bounds, support, rank and optimizer gates stay
fixed, including at most 800 actual calls; diagnostics accounted separately.

## Denominator and decision

For each sphere layout/population, three noise conditions × three bias levels ×
positive/negative/reference starts =27 cases. Across six layout/population pairs:
162 cases. Add the existing nine clean parallel-plane bias/start controls without
noise (their purpose is degeneracy rejection). Run both methods on every case:
342 fits total. Each batch still has three frames. Compare same noise realizations;
report pose+bias recovery (.5 degree/.20 m/.05 m), pose-only recovery, bias error,
convergence, regressions/gains and truth starts separately. All 63 clean fixed fits
must reproduce archived joint_pose_bias results. Report each noise realization,
not just pooled averages. Preserve all failures and plane rejection.

Tests check quadratic scaling/floor, fixed invalid penalties, reproducible disparity
noise, depth error increasing with range and physical bias units. Hash-pin code,
protocol and archived clean baseline before execution. Retain generated depth
array hashes. Abort on execution/hash failure, retain logs and partial output.
No post-result tuning, threshold relaxation, production or SOTA claim. Synthetic
success would justify further controls, not real-data adoption.
