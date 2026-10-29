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


## Results

Source and protocol were frozen at `c9f24ea1`. All twelve cases completed with
identical sensor/metadata hashes and initial matrices to the previous run.
No post-result tuning or reruns were performed.

**Accepted-stage recovery is 0/12**, versus hard-support fitting **1/12** and
matched-input D2D **5/12**. Five tapered fits pass the numerical acceptance rules,
but all five fail the joint reference-accuracy gates. Rejections comprise three
`lost_support`, two `max_evaluations` and two `at_bound`; each rejected output is
exactly its initializer. Returned-pipeline recovery is also 0/12.

| Recording / seed | Tapered status | Raw rotation (deg) | Raw translation (m) | Hard-support accepted recovery |
| --- | --- | ---: | ---: | --- |
| 0005 positive | lost_support | 1.2501 | 0.4326 | False |
| 0005 negative | max_evaluations | 1.0093 | 0.1909 | False |
| 0002 positive | converged | 0.7208 | 0.3848 | False |
| 0002 negative | converged | 0.9195 | 0.1652 | False |
| 0009 positive | converged | 0.9774 | 0.4058 | False |
| 0009 negative | max_evaluations | 0.4356 | 0.0511 | True |
| 0011 positive | at_bound | 0.7308 | 0.5800 | False |
| 0011 negative | converged | 0.7817 | 0.3427 | False |
| 0013 positive | lost_support | 0.5145 | 0.3101 | False |
| 0013 negative | converged | 0.5707 | 0.1308 | False |
| 0015 positive | at_bound | 0.6730 | 0.5790 | False |
| 0015 negative | lost_support | 0.9356 | 0.3109 | False |

The 0009-negative raw candidate meets reference gates, but exhausts the unchanged
150-evaluation budget and is rejected. It is not counted as a recovery and is
not rerun with a larger budget. Two other cases reach search bounds. Fewer
support rejections than the hard mode do not demonstrate improved calibration:
other failure modes increased and every accepted pose is inaccurate.

### Interpretation and decision

Controlled tests establish that this penalty blend removes finite cost jumps
at the tested hole/step boundaries. They do **not** establish a correct visibility
model. The support field penalizes projection into invalid areas; such a penalty
can favor moving projections toward valid image regions regardless of correct
geometry. These outcomes do not isolate that effect causally, but they refute
adoption of this particular approximation under the frozen protocol.

Keep the new mode experimental and the default unchanged. Do not claim that
smoothness fixes calibration, increase the budget for the one favorable raw
candidate, relax the gates, or present numerical conditioning as independent
pose validation. This 0/12 result remains immutable.

The next modeling step should require independently supported correspondences
and controlled tests where true geometry crosses holes/occlusions—not merely
continuity and unobstructed-plane recovery. Stop extending this image-support
penalty approximation through threshold variants. Full-pose acceptance remains
unvalidated, as five numerically accepted outputs are false accepts on this
spent-data comparison. No production behavior or SOTA standing changed.

[Complete evaluation](../assets/tapered_disparity_20261029/evaluation.json),
[schema](../assets/tapered_disparity_20261029/evaluation.schema.json) and
[execution log](../assets/tapered_disparity_20261029/execution.log) retain all raw
candidates, fallback matrices, source/input identities and optimizer diagnostics.
The publisher checks paired input/initializer equality with the hard-support
experiment and exact rejection fallback. Historical replay requires the original
frozen checkout because source-hash validation intentionally detects later changes.
