# First-order tangent angular uncertainty validation

Pre-registered 2026-10-30. Replace the enclosing normal-space ball with the
first-order covariance of the normalized normal: J=(I-n*n^T)/||N||, covariance
J C_N J^T. Bound its95% two-dimensional tangent ellipse by a circle with radius
sqrt(chi2_2(.95)*lambda_max), in radians, capped90 degrees. This is an approximate
delta-method confidence circle, not a rigorous finite-noise angular guarantee.
Same5x5 disparity fit and .5px noise assumptions, no new threshold/window tuning.

Validate on27 exact-plane coefficient scenarios: slopes (0,0),(.3,-.1),(.8,.4),
center disparities6,12,24px; pixel principal point,(40,40),(280,150).20000 independent
Gaussian coefficient samples per scenario, one fixed RNG20261031. Compare empirical
angular coverage and bound medians with the previous enclosing-ball construction;
report fraction whose bound<=15deg. Necessary plane coverage gate: every scenario
>=94% (nominal95% with fixed1 percentage point development tolerance). Failing
this gate blocks adoption even if some intervals shrink. Do not tune after results.

Then diagnose the same27 analytic curved/boundary frame conditions regardless of
coverage-screen outcome, explicitly diagnostic only if coverage failed. Prior
reliability gate plus radius<=15deg; keep previous precision/retention/spatial/
directional criteria. Compare identical depth hashes to prior experiments. Curved
model bias is not described by Gaussian coefficient noise. No ICP/real transfer.
No new calibration fits or changed recovery results. Tests verify analytic tangent
variance at the principal point and missing-patch handling. Pin all code/protocol/
baseline hashes before execution. Report empty selections and all failed cases.
