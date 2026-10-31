# Angular uncertainty of local disparity planes

Pre-registered 2026-10-30. Use the same27 analytic frame conditions and compare
all-valid patches, the prior disparity/depth gate and that gate plus angular
confidence radius<=15 degrees. No ICP/real-data run or tuned alternate windows.

For a 5x5 centered affine disparity design [dx,dy,1], independent .5px Gaussian
noise gives coefficient covariance diag(.25/50,.25/50,.25/25). Map coefficients
[a,b,c] to unnormalized normal [fx*a,fy*b,c-a*(u-cx)-b*(v-cy)]. Covariance maps
by A C A^T. Enclose the95% chi-square3 coefficient ellipsoid by a normal-space
ball R=sqrt(chi2_3(.95)*lambda_max(A C A^T)). Conservative unsigned angular radius
is asin(min(1,R/||normal_hat||)), capped90 degrees. Invalid patches explicitly90.
Assume .5px even on clean controls, as before. This bound is exact for the stated
linear affine-disparity Gaussian model, but curved/cross-surface patches violate
that model; no blanket95% guarantee is made for those measurements.

First unit-test Monte Carlo coverage on20000 independent coefficient realizations
(seed20261030), fixed tilted plane [.3,-.1,24] at principal point: require >=95%
coverage. This is a code/model sanity check, not validation on real stereo noise.
Then run prior27 scene/frame/noise conditions; retain exact source/depth hashes.
Truth normals are analytic and scoring-only. Report radius quantiles, actual
angular coverage and joint XYZ/normal precision for all populations, retention,
spatial cells and normal spectrum. No normal-plane smoothing of scored XYZ.

Keep the prior necessary continuation screen: all18 noisy frames must have
>=80% joint precision, >=10% domain retention, >=8 occupied cells and normal
singular ratio>=.05. Empty selections fail support, with precision/coverage null.
No post-result tuning, transfer or registration if screen fails. Preserve all
cases and clean controls. Source/protocol/baseline hashes frozen before execution.
