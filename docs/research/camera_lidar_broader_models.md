# Broader spent-drive measurement-model comparison

## Frozen protocol

Stop tuning the four-drive translation-refinement experiment. Compare the existing
joint six-DoF Borer D2D solver with the existing native Pandey image/intensity
mutual-information solver on two further **already-spent** KITTI recordings:
2011_09_26_0013 (prior camera audit) and 0015 (prior motion development). Exposure
is documented in `docs/sota_handoff.md` and the prior camera/lidar-vehicle reports.
These add recording coverage, not dataset families, rigs or held-out evaluation.

Use the fixed quartile rule floor((N-1)*q/4), q=1,2,3. Metadata-only archive
inspection selected 0013 frames 35/71/107 and 0015 frames 74/148/222. The
[plan](../assets/broader_models_20261029/plan.json) pins archive size/ETag and sources
before any new sensor payload acquisition. Retain both pre-existing positive and
negative perturbations and report all four cases for each method.

Acquire only the six selected stereo/point frames and adjacent OXTS poses using
the existing 64 MiB-per-drive sparse-fetch guard. Build the corrected-projection
stereo frontend, with camera-time point compensation unchanged. No re-rectification,
cycle mask or experimental translation refinement. Both solvers see the same
three clouds, initial transforms and camera model. D2D additionally sees stereo
depth; Pandey uses equalized luminance and rank-equalized LiDAR intensity. This
unequal-information comparison is not a monocular-superiority claim.

Use unchanged solver settings; Pandey retains its existing shared-bundle setting
`min_train_observations=1, holdout_ratio=0`. Both optimize all six DoF. Each fit
runs once, sequentially, with four CPU affinity slots, BLAS threads=1 and a
180-second timeout. Timeout/non-convergence is a failure, never silently counted
as recovery. Retain raw transforms and reference differences separately from
converged recovery. References seed the controlled perturbation and score outputs;
they are not supplied as objective inputs. Existing seed files contain reference
metadata, but solver drivers read only the initial matrix.

No retry based on accuracy, threshold tuning, default changes or SOTA claim.
Archive acquisition/preparation failure separately if it blocks any planned case.
