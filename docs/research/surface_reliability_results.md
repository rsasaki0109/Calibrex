# Local disparity reliability does not certify accurate surface normals

Frozen `54c0a3e0`, evaluated 2026-10-30. See
[protocol](surface_reliability_protocol.md),
[plan](../assets/surface_reliability_20261030/plan.json),
[all 27 frame conditions](../assets/surface_reliability_20261030/evaluation.json)
and [schema](../assets/surface_reliability_20261030/evaluation.schema.json).

Reject this gate before any real-data transfer or ICP rerun. All18 noisy frame
conditions fail the prespecified >=80% joint surface precision requirement.
Selecting near, locally planar disparity patches improves precision relative to
all valid pixels, but it remains far below a usable quality screen.

| Condition | All-valid median joint precision | Selected median joint precision | Selected precision range | Median domain retained |
| --- | ---: | ---: | ---: | ---: |
| Clean | 94.14% | 100% | 100–100% | 15.19% |
| .5 px, seed730 | 4.80% | 13.31% | 7.92–28.81% | 15.13% |
| .5 px, seed731 | 4.88% | 13.05% | 8.19–29.33% | 15.08% |

Joint precision means noisy pixel XYZ within .2 m of clean pixel XYZ AND fitted
normal within15 degrees of the exact analytic normal. Position uses the original
pixel depth, not the smoothed plane depth. The denominator is the population's
valid/selected interior pixels; this is not cross-modal correspondence recovery.
Analytic surface labels and normals enter scoring only.

Retention spans 5.17–28.58% of the interior domain across all conditions. Selected
points occupy only 4–13 of16 image cells; even clean selections have median7 cells.
Thus high clean precision also hides reduced spatial support. Noisy selections
have median8 cells, but do not meet the accuracy criterion in any frame. All noisy
normal spectra pass the .05 ratio screen despite poor accuracy: noise can spread
normal directions and make a spectrum appear well conditioned. Directional spread
is not proof of trustworthy directional information or six-DoF observability.

The assumed .5 px noise and .2 m depth-uncertainty gate favor near surfaces. A
5x5 affine disparity fit may describe a locally smooth noisy patch while its
slopes—and therefore its normal—remain uncertain. A bound on depth error and a
small plane residual do not bound angular error. This result tests the declared
combination, not all possible reliability providers; no alternate window or gate
was selected after seeing outcomes. The synthetic independent pixel noise also
does not reproduce all correlations/outliers of a real stereo frontend.

Four focused tests, repository Ruff and strict documentation build pass. Checks
validate schema, frozen hashes and completion of all27 conditions. Tests verify
near-plane acceptance, far-plane rejection, analytic front-plane normal, depth
boundary rejection and missing-neighborhood handling. No calibration, real-data
filtering, solver change or revised recovery verdict occurred.

Next derive and test angular uncertainty of the fitted disparity slopes rather
than treating planarity/eigengap as normal confidence. Specify the covariance from
the known pixel-noise model, check its coverage against exact normals, and retain
empty selections or insufficient spatial support as failures. This can establish
whether the measurement scale supports usable normal estimates at all. Do not
hand-tune the 18 spent real frames or start a new ICP run with this failed gate.
Full-cloud MI remains the existing real baseline; no adoption or SOTA claim.
