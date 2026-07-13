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

Open3D Generalized ICP and PCL/Autoware NDT are distinct optional adapters.
Open3D is imported only when installed and receives train source points from
the same deterministic spatial split. PCL/Autoware NDT remains a subprocess or
precomputed-transform boundary, recording tool version, license, command, and
whether train isolation was declared.

Both adapters pass their output transform back through
`evaluate_icp_candidate`, which recomputes train residual, fresh spatial
holdout nearest-neighbor RMSE, rematching curvature, and correspondence Jaccard
inside Calibrex. Backend-specific fitness, probability, or inlier RMSE values
remain raw provenance and are not compared as common metrics. GPL or
license-uncertain implementations are not copied into `src/calibrex`.

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

Open3D was not installed in the evaluation environment and external NDT was
not configured. Both states are serialized as `unavailable`/`not_executed`;
no comparison value is synthesized. The complete reproducible output is
written to `outputs/livox_horizon_horizon_icp_comparison` by:

```bash
calibrex calibrate \
  examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml
calibrex verify \
  outputs/livox_horizon_horizon_icp_comparison/bundle.json \
  --require-raw-recomputed
```
