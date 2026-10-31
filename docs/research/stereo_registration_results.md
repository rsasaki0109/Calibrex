# Frame-wise stereo geometry registration pilot fails drive-level recovery

Frozen at `bd9709ee`, evaluated 2026-10-30. See
[protocol](stereo_registration_protocol.md),
[executable plan](../assets/stereo_registration_20261030/pilot-plan.json),
[all 108 fits](../assets/stereo_registration_20261030/evaluation.json) and
[schema](../assets/stereo_registration_20261030/evaluation.schema.json).

Reject adoption of these two native ICP pilots on the prepared stereo geometry.
Neither method recovers all three frames of any perturbed drive/start. No
confidence, distance, voxel or iteration threshold was retuned after the results.

| Outcome | Point-to-point | Point-to-plane |
| --- | ---: | ---: |
| Perturbed frame recovery | 1/36 | 0/36 |
| Perturbed drive/start recovery (all three frames) | 0/12 | 0/12 |
| Reference-start frame controls | 6/18 | 2/18 |
| Reference-start drive controls (all three frames) | 1/6 | 0/6 |
| Perturbed numerical convergence | 8/36 | 9/36 |
| Reference numerical convergence | 6/18 | 3/18 |
| Perturbed median rotation error | .8828 deg | .8221 deg |
| Perturbed median translation error | .1700 m | .3097 m |
| Reference median rotation drift | .2223 deg | .2633 deg |
| Reference median translation drift | .0938 m | .1688 m |

The remaining statuses are max_iterations: perturbed 28/36 and 27/36; reference
12/18 and 15/18. Ignoring numerical status, five perturbed frames per method
meet the geometric accuracy gates; they are not all accepted recoveries. Reference
geometry alone passes in 17/18 point-to-point and 11/18 point-to-plane cases, also
not substituted for the declared convergence requirement. This exposes a termination
limitation as well as inaccurate converged cases; no iteration-cap extension is
applied after observing it.

All three frames of each recording were independently paired. Reference poses
were used only in the labeled controls and scoring. No cross-time cloud matching,
pose averaging, reference-dependent filtering or best-frame selection occurred.
Historical joint MI6/12 uses a three-frame estimator and remains contextual; it
is not a matched numerical comparator for this single-frame pilot.

## Evidence and verification

Inputs are the previously prepared 18 arrays, with 25536–42927 source points and
38351–71126 target points. Source/target cardinality is not overlap. The report
retains correspondence counts, train/holdout counts, error diagnostics, conditions,
iteration history and resolved options. Complete results including point split IDs
are gzip artifacts under /workspace/work/stereo-registration-pilot-20261030;
paths and SHA-256 values are retained in every case. These local runtime artifacts
must be copied or regenerated when moving the workspace.

Total solver time including diagnostics: point-to-point 257.56 seconds,
point-to-plane 293.13 seconds. Input loading and report compression are excluded;
this is not a controlled performance benchmark. Point-to-point multi-start
inspection was explicitly disabled per the prior protocol; all other settings and
spatial holdout policy remain as declared.

Thirteen focused tests, repository Ruff and strict documentation build pass.
Validation checks report/schema, 108-case completion, all frozen source and input
hashes, every detailed-result artifact hash, split counts, and disjoint training/
holdout point IDs. Native solver and production defaults are unchanged. No fresh
recording, trained checkpoint, external code execution or SOTA result is added.

## Next decision

These inputs do not yet establish a dependable geometric correspondence provider.
Before another registration variant, quantify whether each prepared stereo target
actually provides accurate local geometry around the LiDAR surfaces: at reference
pose, measure nearest-target distances and coverage, and compare stereo target
normals with local LiDAR normals where supported. Keep this as offline measurement
diagnosis, not truth-filtered fitting. Include all 18 frames and retain missing/
weak neighborhoods. This separates insufficient overlap/noisy surfaces from the
solver's iteration limit before spending effort on another optimizer or adding a
new dependency. Preserve the existing full-cloud MI baseline and all failures.
