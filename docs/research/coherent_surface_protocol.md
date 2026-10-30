# Larger coherent disparity surface support

Pre-registered 2026-10-30. Evaluate fixed5/11/21 pixel windows and a measurement-only
smallest-passing-window rule on the same27 clean/noisy analytic frames:108 rows.
These are spent controlled geometries; no real-data tuning or calibration fits.
All scales share the ten-pixel inset domain. Noise=.5px, seeds730/731+100*frame,
and the existing clean controls. Hash-paired input depth must remain identical.

Fit affine disparity by least squares with uniform support. Estimate surface
normal analytically and center depth from the fitted disparity. This smooths
position as well as estimating normals, so it is NOT a matched accuracy comparison
with earlier raw-pixel position scoring. Exact analytic center geometry is truth
for scoring only. Near discontinuities, a larger plane can be biased even if
random error decreases. Keep the original per-window plane RMS<=1px test.

Use the conservative normal-space ellipsoid-enclosing ball, not the failed tangent
approximation. For window w, coefficient covariance is
sigma_px²*diag(1/(w²*mean(dx²)),1/(w²*mean(dy²)),1/w²). The normal map and95% chi3
ball construction are unchanged. Center depth sigma now uses fitted mean disparity
sigma=.5/w, propagated by z²/120; require <=.2m. Require angular radius<=15deg.
This variance is valid for an exact affine plane with independent noise, not a
bound on curvature/mixed surfaces. Invalid neighborhoods are rejected.

Adaptive rule: inspect5 then11 then21, select the first accepted estimate per pixel.
No truth, fitted extrinsic or result-driven window selection. For each fixed scale
and adaptive output, report selected counts, domain retention, occupied4x4 cells,
normal spectrum, joint smoothed-XYZ<=.2m AND normal<=15deg precision, and actual
angular coverage. Empty populations have null precision/coverage.

Same necessary continuation screen: every noisy frame joint precision>=80%,
retention>=10%, >=8 image cells and normal singular ratio>=.05. Keep clean model
bias visible and report per-scale failures; do not choose a winning scale after
outcomes or transfer a failed gate to real data. Independent mathematical checks
verify exact affine reconstruction and decreasing variance with support. Freeze
sources/protocol before execution. Preserve all108 rows; no adoption/SOTA claim.
