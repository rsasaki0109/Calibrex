# Joint pose and camera-depth bias protocol

Follow the [fixed-bias response](depth_bias_response_results.md) with experimental
seven-parameter fitting. Keep the frozen visibility residual/scene layouts and
all 54 previous conditions. Add nine parallel-plane controls (three known biases ×
two signed initializations + true start): total 63 new fits. The first 54 use
archived six-parameter baseline reports with pinned file/code hashes; do not imply
they are rerun or compare their times as simultaneous measurements. Parallel planes
have no matched baseline and must be reported separately.

Fit one shared optical-depth correction b across the three observations. Residual
is transformed LiDAR z minus supplied camera depth PLUS b, so a +.25 m injected
measurement bias should yield b=+.25 m. Add b only to valid samples, retaining
unsupported 1 m penalties. Keep positive-residual saturation and outer soft-L1,
pose bounds/scales/tolerances and solver. Initialize all seven deltas at zero;
bias bound ±.5 m and scale .1 m, no prior or true-bias input. This bound is a fixed
synthetic development choice, not a general camera-error model. 100 nfev and hard
800 actual residual-call ceiling include numerical Jacobian work. Abort with logs
if exceeded. No outcome-based restart or parameter changes.

After fitting, report raw and effective-robust local Jacobians on the intersection
of supported points over ±.001 degree/m changes in all seven parameters. The
robust equivalent residual is s*sqrt(2/(sqrt(1+(s/.1)^2)+1)), where s is the saturated
residual; its squared norm equals twice the soft-L1 cost. Scale columns by 1 degree
for rotation and .1 m for translations/bias. Record singular spectra and the bias
column's unexplained fraction after projection onto pose columns. Flag robust
singular ratio <1e-6 OR bias unexplained fraction <.05 as locally near-degenerate;
<100 common points is unavailable/degenerate. These provisional diagnostics do
not prove global identifiability or provide uncertainty intervals. Fifteen diagnostic
residual passes are separate from optimizer calls; reported wall time includes them.

Numerical convergence/support checks and original pose gates stay unchanged.
Record pose-only recovery separately from **joint recovery**, which additionally
requires estimated-bias absolute error <.05 m and no robust local-degeneracy flag.
Report 36 perturbed sphere-scene fits, 18 sphere truth-start controls, and 9 plane
controls separately. Do not count arbitrary plane solutions as uniquely identified,
even if their pose happens to pass. No held-out acceptance scores select settings.
All zero-bias regressions, boundary fits, nonconvergence and inaccurate numerical
successes remain in the denominator. No production/SOTA claim from this scene family.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_pose_bias.py docs/assets/joint_pose_bias_20261029/plan.json \
  docs/assets/joint_pose_bias_20261029/evaluation.json
```
