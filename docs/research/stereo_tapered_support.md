# Continuous support weighting for inverse-depth pose fitting

## Frozen development protocol

Address the hard missing-data residual switch in the prior six-DoF inverse-depth
estimator. Use the exact same twelve cases, initial point populations, seeds,
search bounds, optimizer tolerances, budgets and acceptance gates. Compare to
both archived hard-support fitting (1/12 recovery) and matched-input D2D (5/12).
No input regeneration, new data or favorable-case selection.

Build an image-only support field. A valid pixel has finite positive depth and
3x3 equivalent-disparity range <=2 px. The image boundary is unsafe. Compute
Euclidean distance to unsafe pixels, map `(distance-1)/2` into [0,1] and apply
smoothstep. Bilinearly sample the resulting confidence c at the projected point.
Depth uses a nearest-safe-pixel extension solely to keep interpolation finite;
this extension never becomes a depth measurement or final acceptance support.

Replace the hard switch with two residual channels:

```text
measurement = sin(pi*c/2) * equivalent_disparity_residual / sigma_px
missing     = cos(pi*c/2) * (8 px / sigma_px)
```

Both channels receive the existing soft-L1 loss. This is a continuous penalty
blend, not a probabilistically calibrated mixture. Out-of-image confidence is zero
by boundary clamping; behind-camera confidence fades between z=0.1 and 0.2 m.
Bilinear fields remain piecewise differentiable, not globally smooth. Support
confidence can bias pose toward valid areas; it is not physical visibility.

Final support still uses the original four-neighbor/range check and requires
90% retention and at least 128 points/frame. The condition check uses only the
measurement rows of the robust weighted Jacobian, excluding the missing-support
channel; even those rows include support gradients and are not an independent
accuracy criterion. Rejected poses return the initializer exactly. Defaults stay
`hard`; the new mode is opt-in and remains unregistered.

Before any real run, tests verify finite cost continuity across artificial holes
and depth steps, nonzero penalty for missing support, zero observation weight
when all depth is absent, and six-axis recovery from independent known planes.
All tests passed. Freeze code/options/protocol before the twelve-case execution;
no post-result threshold adjustment or rerun.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_disparity_pose.py \
  docs/assets/tapered_disparity_20261029/plan.json /tmp/tapered-disparity.json
```
