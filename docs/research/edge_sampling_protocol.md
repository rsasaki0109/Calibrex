# Image-only edge-preserving sampling control

Pre-registered 2026-10-30 before running the evaluation. Test whether avoiding
foreground/background interpolation removes the bilinear depth/bias offset,
without oracle scene access. This is synthetic development evidence, not SOTA.

Use the same seven-parameter solver function definitions as joint_pose_bias;
only change the sampler import. Keep optical-z conversion, all-four-neighbor
validity, fixed 1 m invalid penalty and point populations unchanged. In each
2x2 patch select the nearest pixel (half-pixel ties right/down), retain neighbors
within 0.2 m optical depth of that anchor, and renormalize bilinear weights.
This fixed threshold matches the existing occlusion saturation scale, not a
fit to evaluation outcomes. Constant additive optical-depth bias leaves the
selection unchanged. Smooth patches with range ≤0.2 m exactly retain bilinear
sampling. The method uses only the depth image and projected coordinates; it
cannot select a surface using the LiDAR point depth or the known scene.

Nearest-pixel and gating transitions can introduce discontinuities. This is
not an assumption of solver smoothness or a guarantee of improvement. Do not
retune threshold after the results. Do not add scalar rescue or change loss,
optimizer, bounds, budget, convergence tests, support or identifiability gates.

Evaluate all 54 sphere cases (three layouts, two populations, biases 0/+.25/-.25,
positive/negative seeds and truth starts), plus nine parallel-plane controls.
Compare against pinned archived joint_pose_bias results, not the oracle sampler
or scalar rescue. All denominators retained; truth-start recovery separate.
Report joint pose+bias recovery, pose-only recovery, bias errors, convergence,
regressions/gains, actual residual calls ≤800 and plane degeneracy rejection.
Accuracy gates remain .5 degree/.20 m/.05 m. Support/sensitivity passes remain
outside optimizer accounting as before. No new real data used.

Tests cover foreground/background separation, additive bias equivariance, smooth
patch equivalence, metric-range conversion and invalid-neighbor support. Freeze
source/protocol/baseline hashes in plan.json and commit before executing.
