# Direct disparity-domain control

Pre-registered 2026-10-30. Compare a direct-disparity candidate against all 171
archived fixed-metric cases in stereo_uncertainty_20261030/evaluation.json.
Reuse its exact generation: clean and .5 px noise seeds 730/731, three layouts,
visible/all returns, metric bias 0/+.25/-.25, both perturbations and reference,
plus nine clean parallel-plane controls. Run 171 candidate fits; baseline archived,
not refitted. Verify every generated depth-array hash matches its baseline.

Candidate: convert each valid optical-depth image pixel to disparity 120/depth,
then bilinearly sample disparity. Predict disparity 120/(camera_z+bias). Residual
is .2*(observed_disparity-predicted_disparity), so .5 px maps to .1 residual units.
Positive residual corresponds to a LiDAR point behind the observed surface;
retain positive saturation .2 and soft-L1 .1. This changes measurement space and
interpolation space together. It is not an isolated interpolation experiment or
an exact Gaussian likelihood after robust saturation. Bias stays in metres with
the original bounds/sign. Metric raw residual is retained for raw sensitivity;
robust sensitivity reflects the disparity objective. Unsupported residual stays
1; reject predicted z+bias<=.1. No oracle, fitted variance, scalar rescue or edge
sampler. Otherwise same seven-parameter TRF, bounds, support/rank gates and 800
actual-call budget; diagnostic passes separate.

Injected bias is added after noisy disparity inversion as in the baseline. Thus
the predicted bias model matches clean depth offsets, but noise with nonzero
metric offset is not exactly constant-variance disparity noise. Report this
limitation rather than claiming a calibrated likelihood. No real-data tuning.

Report pose+bias accuracy (.5 degree/.20 m/.05 m), pose accuracy, convergence,
gains/regressions, bias error, each noise seed and reference starts separately;
all plane rejections and denominator retained. No post-result tuning. Freeze
source/protocol/archive hashes before fitting. Abort and preserve partial output
on execution/hash failure. No production/SOTA claim from this synthetic control.
