# Disjoint spatial prediction test for coherent surfaces

Pre-registered 2026-10-30. Compare the frozen21px estimator with an additional
measurement-only model test, same27 input frames:54 rows. Retain plane depth,
normal, confidence and original selection. Candidate only removes points.

Split each21x21 patch by local checkerboard parity:221 fit pixels,220 holdout.
Fit affine disparity on the221. Predict the disjoint220. Under independent .5px
Gaussian noise and a correct plane, prediction covariance is
.25*(I+Xh*(Xt^T Xt)^-1*Xh^T). Use the exact Mahalanobis statistic via Woodbury,
with220 degrees of freedom, accepting at chi2_220(.99). This incorporates common
coefficient uncertainty rather than treating all predicted residuals as independent.
Verify10000 exact-plane noise draws (seed20261031), acceptance .98–.997.

After this validation rule, report the original full-patch21px estimate for accepted
locations. Full-patch refitting uses holdout pixels; this is a diagnostic model
validation gate, not a claim of independent post-selection confidence coverage.
Checkerboard noises are independent only under the frozen synthetic noise model;
actual stereo errors may be correlated. Curved patches can predict held-out pixels
well without having the correct center normal. Selection must be judged against
analytic center geometry, not just prediction scores. No truth enters the gate.

Keep all clean/noisy/frame conditions and the common domain. Same joint precision
>=80%, retention>=10%, >=8 cells and normal singular ratio>=.05 in EVERY noisy frame
for continuation. Also report angular coverage. Reproduce archived21px baseline
and hashes; missing data explicitly reject. No post-result threshold tuning,
real filtering, ICP rerun, changed accuracy verdict or SOTA claim.
