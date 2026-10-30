# Camera-parametric predictive surface transfer

Frozen before execution,2026-10-30. First require bitwise equality of all arrays
against the prior predictive estimator on27 analytic measurements, then evaluate
18 spent real frames with baseline21 and predictive21:36 diagnostic rows. No ICP.
The generic tools require explicit focal lengths, principal point, positive fB
and noise; no analytic camera constants may leak into real construction.

Use archived restored SGBM depth, recorded original-left intrinsics, and scalar
fB=P_rect_02[0,3]-P_rect_03[0,3] from pinned official calibration. This defines a
surrogate disparity fB/z in the original camera frame. Actual stereo was computed
in another rectified frame and restored with nearest sampling/z conversion; thus
this surrogate is not the raw disparity with independent noise. Assume .5px as
an explicit transfer hypothesis, not a calibrated real sensor model. Pixel noise
correlations and invalid21x21 neighborhoods can limit retained support.

Keep21px fit, covariance, all thresholds and checkerboard predictive test fixed.
No real-reference normals enter fitting, masks or parameter selection. Compare
selected target pixels directly to reference-transformed voxel LiDAR sources;
source normals use the existing12-neighbor/eigengap.02 method. Record distances,
normal agreement among valid-normal pairs within1m, selected count, domain retention
and4x4 image coverage. This uses dense selected target pixels, not the previous
voxel-target/source-FOV diagnostic population, so do not compare percentages as
identical estimands. LiDAR normals are evaluation proxies, not exact surface truth.

Necessary continuation gate, every frame: retention>=10%, >=8cells, nearest-source
within1m>=80%, >=1000 valid normal pairs and >=80% unsigned normal agreement within
15deg among those pairs. These are diagnostic development screens, not guaranteed
calibration gates. If any fails, do not run ICP as an accepted surface provider.
Preserve all36 rows, empty selections and rejection. Schema/hash checks and27-case
analytic parity required. No post-result tuning, production adoption or SOTA claim.
