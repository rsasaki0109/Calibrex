# ICP effective-observability research note

## Scope

Calibrex's native point-to-point ICP is a local registration baseline. It does
not claim global convergence, and it does not interpret a frozen nearest-pair
least-squares matrix as calibration covariance.

Primary references:

- Besl and McKay, *A Method for Registration of 3-D Shapes*, IEEE TPAMI 1992,
  DOI `10.1109/34.121791`, for the refreshed closest-point iteration.
- Chetverikov et al., *The Trimmed Iterative Closest Point Algorithm*, ICPR
  2002, DOI `10.1109/ICPR.2002.1047997`, for fixed-overlap trimming.
- Censi, *An Accurate Closed-form Estimate of ICP's Covariance*, ICRA 2007,
  DOI `10.1109/ROBOT.2007.363961`, for correspondence dependence and explicit
  observability analysis.
- Bonnabel, Barczyk, and Goulette, *On the Covariance of ICP-based
  Scan-matching Techniques*, ACC 2016, DOI `10.1109/ACC.2016.7526532`, for the
  central warning that point-to-point ICP rematching invalidates the ordinary
  fixed-correspondence Hessian covariance argument.

## Diagnostic contract

The existing `information_singular_values` remain a frozen-pair geometric
diagnostic. They are never serialized as covariance.

Effective local response is evaluated as a black box at the final transform.
For each of `x`, `y`, `z`, `roll`, `pitch`, and `yaw`, Calibrex applies symmetric
positive and negative perturbations and reruns the complete correspondence
policy:

```text
transform source
  -> nearest target
  -> optional mutual check
  -> distance gate
  -> stable trimming
  -> mean squared residual
```

The central second difference is reported as rematched objective curvature.
Translation and rotation use separately declared steps and thresholds because
their units differ. Negative finite-difference curvature is clamped to zero and
treated as weak; this remains a diagnostic, not a covariance estimate.

For the same perturbations, correspondence stability is the Jaccard ratio of
the `(source index, target index)` pair sets against the final baseline. The
minimum and mean across the six directions are recorded. A high curvature with
unstable pair identity is therefore distinguishable from a stable local basin.

## Chen-Medioni native tangent-plane baseline

Primary references are Yang Chen and Gerard Medioni, *Object Modeling by
Registration of Multiple Range Images*, IEEE ICRA 1991, DOI
[`10.1109/ROBOT.1991.132043`](https://doi.org/10.1109/ROBOT.1991.132043),
equations (10)-(12), and the expanded Image and Vision Computing article, DOI
[`10.1016/0262-8856(92)90066-C`](https://doi.org/10.1016/0262-8856(92)90066-C).
The [primary conference PDF](https://graphics.stanford.edu/~smr/ICP/comparison/chen-medioni-align-rob91.pdf)
defines a refreshed tangent-plane distance and a small-rotation least-squares
update.

The paper finds the intersection between a transformed source-normal line and
the continuous target surface, then uses the tangent plane at that intersection.
A PCD contains discrete samples rather than a continuous surface. Calibrex
therefore implements and names a **discrete tangent-plane specialization**: the
closest target sample supplies a tangent plane estimated from its deterministic
`k`-nearest PCA neighborhood. This approximation boundary is result provenance;
the implementation is not described as the paper's full line/surface
intersection algorithm.

Each iteration refreshes Euclidean correspondences, applies the common mutual,
distance, valid-normal, and trimming policies, and solves

```text
r_i = transpose(n_i) (p_i - q_i)
J_i = [transpose(n_i), transpose(p_i cross n_i)]
J delta = -r
```

for a left SE(3) small-motion update. The six singular values, rank, and
condition number of this actual point-to-plane system are gates, not covariance.
A single plane correctly has rank 3 because its two tangent translations and
normal-axis rotation are unobservable. Line-like PCA neighborhoods fail the
normal eigengap gate before optimization.

The native result includes train and spatial-holdout point-to-plane RMSE,
normal count/fraction and eigengap statistics, all iteration updates, and twelve
signed ±5 cm/±5 degree probes with fresh rematching. Its final transform is also
passed through the unchanged backend-neutral point-to-point holdout,
correspondence Jaccard, curvature, and inlier metrics. This separates local
surface fit from whole-cloud generalization. The implementation is typed,
ROS-independent NumPy and copies no publisher or external solver code.

## Global symmetry falsification

Local curvature cannot discover a distant equivalent solution. Calibrex also
runs six independent ICP trials from declared translational and rotational
initialization perturbations. Each trial records its status, holdout residual,
and transform distance from the baseline solution.

A trial is a symmetry ambiguity only when both conditions hold:

1. its output transform is materially distinct from the baseline; and
2. its unchanged spatial holdout RMSE is equivalent within the declared margin.

The baseline is not silently replaced by the best multi-start result. The
diagnostic falsifies uniqueness. A synthetic repeated-structure test contains
two translated copies of the same asymmetric 3D cloud and proves that the
multi-start path reports the second exact solution.

## Adapter boundary

Primary GICP reference: A. Segal, D. Haehnel, and S. Thrun,
[*Generalized-ICP*](https://www.roboticsproceedings.org/rss05/p21.pdf), Robotics:
Science and Systems V, 2009, DOI
[`10.15607/RSS.2009.V.021`](https://doi.org/10.15607/RSS.2009.V.021). The paper
models local surface structure in both scans and minimizes a probabilistic
plane-to-plane objective. Calibrex calls Open3D's MIT implementation through an
optional import; it does not re-label point-to-point ICP as GICP and does not
copy the paper or Open3D implementation.

Primary NDT reference: P. Biber and W. Strasser, *The Normal Distributions
Transform: A New Approach to Laser Scan Matching*, IROS 2003, pp. 2743-2748,
DOI [`10.1109/IROS.2003.1249285`](https://doi.org/10.1109/IROS.2003.1249285).
The paper represents fixed spatial cells by normal distributions and optimizes
the likelihood of transformed scan samples. Calibrex does not implement an
unverified approximation: PCL or Autoware NDT remains a declared subprocess or
precomputed-result boundary.

Open3D Generalized ICP and PCL/Autoware NDT are distinct optional adapters.
Open3D is imported only when installed and receives train source points from
the same deterministic spatial split. PCL/Autoware NDT remains a subprocess or
precomputed-transform boundary, recording tool version, license, command, and
whether train isolation was declared.

Both adapters pass their output transform back through
`evaluate_icp_candidate`, which recomputes train residual, fresh spatial
holdout nearest-neighbor RMSE, inlier fraction, rematching curvature, and
correspondence Jaccard inside Calibrex. These values are exported as parallel
native/GICP/NDT metric families with the same fixed gates and split IDs.
Backend-specific fitness, probability, or inlier RMSE values remain raw
provenance and are not compared as common metrics. Transform deltas against the
native estimate are ungated diagnostics because this dataset has no extrinsic
ground truth. GPL or license-uncertain implementations are not copied into
`src/calibrex`.

Adapter inputs are fingerprinted as ordered stable point IDs plus big-endian
float64 XYZ values. GICP records full-input fingerprints, the train-source
fingerprint, resolved options, and the MIT boundary even when Open3D is absent.
NDT records the external result hash, declared SPDX license, train-isolation
declaration, command, timeout, working directory, and input fingerprints.
Subprocess evidence distinguishes completed, timeout, and OS-error states and
retains return code, elapsed time, output byte counts, and SHA-256 hashes.
Arbitrary stdout/stderr contents are not embedded in calibration results.
Unknown NDT SPDX is an explicit warning rather than an implicit license claim.

## Public Livox Horizon-Horizon result

The checked public-data protocol is
`examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml`.
It maps the target-sensor PCD into the base-sensor PCD, forms lexicographically
ordered 0.5 m voxel centroids, and deterministically caps each cloud at 500
points. The cap and all raw/downsampled counts are result provenance rather
than an undocumented performance optimization.

On the downloaded Livox pair, native ICP reported train RMSE 0.4283 m,
spatial-holdout RMSE 5.1132 m, and train inlier fraction 0.1839. It therefore
honestly **fails** the predeclared 0.75 m holdout and 0.25 inlier gates. The
frozen-pair information diagnostic nevertheless had rank 6 and condition
number 27.39, while the minimum rematched correspondence Jaccard was 0.7858,
with zero curvature-weak directions and no multi-start symmetry flag. This is
evidence for a locally stable basin, not evidence that the estimate generalizes
over the low-overlap fields of view.

The Chen-Medioni discrete specialization converges in 11 iterations. PCA
produces valid tangent planes for 499/500 target samples (fraction 0.998), with
minimum accepted eigengap 0.02123 and median 0.3790. The final tangent system
has rank 6/6 and condition number 90.22. Point-to-plane RMSE is 0.1533 m on
train and 0.2777 m on spatial holdout, passing its predeclared 0.5 m surface
gate. Only 6/12 signed controls exceed the fixed 5 mm margin.

The unchanged common evaluation reaches train RMSE 0.4385 m but holdout RMSE
5.2634 m and inlier fraction 0.1713, failing the same 0.75 m and 0.25 gates as
the other backends. Minimum rematched correspondence Jaccard is 0.7603, while
three curvature directions remain weak. The public result therefore records a
locally well-constrained tangent-plane fit that does **not** generalize over the
low-overlap fields of view; it remains an honest **FAIL**, not a PASS inferred
from the smaller surface residual.

Open3D 0.19.0 GICP is now executed on exactly the native train source IDs. Its
Open3D-native fitness and inlier RMSE are retained only as raw provenance. The
common Calibrex scores are train RMSE 0.4863 m, holdout RMSE 2.9039 m, inlier
fraction 0.2771, and minimum rematched correspondence Jaccard 0.6873. The
holdout gate remains the unchanged 0.75 m and therefore honestly **fails**.
Rematching curvature reports `y`, `roll`, and `yaw` as weak even though the
Jaccard gate passes. GICP differs from the native estimate by 18.71 degrees and
0.855 m; this is an ungated disagreement diagnostic, not a ground-truth error.
The common split-ID check passes.

External NDT remains unconfigured and is serialized as `not_executed`; no NDT
value is synthesized. Its metric slots are WARN with null values rather than
being omitted or populated from another backend. The schema-valid bundle is
written to `outputs/livox_horizon_horizon_icp_comparison` by:

```bash
calibrex calibrate \
  examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml
calibrex verify \
  outputs/livox_horizon_horizon_icp_comparison/bundle.json \
  --require-raw-recomputed
```
