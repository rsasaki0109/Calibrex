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


## Results

Source and plan were frozen at `fa09ba7e`. Both sparse preparations completed:
0013 used a planned 11,481,974-byte range budget and 0015 10,819,028 bytes,
well below the 64 MiB/drive guard. All six prepared frames have source/output
hashes. All **eight solver executions completed and reported convergence**;
there were no timeouts or accuracy-driven retries.

The unchanged recovery gates require both rotation <0.5 degrees and translation
<0.20 m against the dataset reference. Initial perturbations are about 1.50 degrees
and 0.1225 m. Initial translation is already inside its gate; preserving it is
not evidence of translation recovery.

| Recording / seed | D2D rotation (deg) | D2D translation (m) | D2D recovered | Pandey rotation (deg) | Pandey translation (m) | Pandey recovered |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 0013 positive | 0.6570 | 0.3021 | False | 3.2005 | 0.1256 | False |
| 0013 negative | 0.9426 | 0.5408 | False | 1.4744 | 0.1270 | False |
| 0015 positive | 0.2688 | 0.2019 | False | 2.8105 | 0.1240 | False |
| 0015 negative | 0.2416 | 0.1663 | True | 17.4007 | 0.1035 | False |

- **D2D recovers 1/4.** Rotation improves from the initial perturbation in all four
  cases, while translation worsens in all four. 0015 positive misses the translation
  gate at 0.20194 m; do not round it into a pass or relax the 0.20 m requirement.
- **Pandey recovers 0/4.** Rotation improves slightly in only one case; translation
  improves in only one case. 0015 negative reports convergence at a **17.4007-degree**
  rotation difference, despite a 0.1035 m translation difference.

Convergence here is an optimizer termination status, not validation of calibration.
The large Pandey failure shows why a single translation metric or optimizer
success flag is insufficient. The native Pandey implementation/configuration
must not be presented as the official paper's achievable accuracy.

### Decision and scope

The observed D2D rotation/translation tradeoff persists on two recordings outside
the earlier four-drive refinement loop. The alternative image/intensity objective
also does not solve the problem under its existing settings. No frontend,
optimizer defaults or recovery thresholds are promoted or tuned from this result.

The assisted-input method-comparison series now has samples from **six distinct
KITTI raw recordings**. This is sparse three-frame coverage, not completion of
the separate all-55-drive capture/replay requirement. Same-day recordings do not
add rigs or dataset families; no fresh evaluation or SOTA claim is supported.

Next prioritize a failure-aware calibration quality check on the full six-DoF
output, using independent measurements/frames and reporting false acceptance,
before trusting an objective's convergence flag. A calibrated rejection mechanism
must preserve coverage as well as reject the 0015-negative failure; fitting a
threshold directly to these four outcomes would not validate it.

[Matrix](../assets/broader_models_20261029/runs/matrix.json),
[preparation provenance](../assets/broader_models_20261029/preparation.json),
[execution log](../assets/broader_models_20261029/execution.log) and the per-case
reports retain the complete outputs, source identities, objective diagnostics,
seeds, references, settings and input hashes. Sensor arrays remain in the separate
experiment workspace. No historical data/results were overwritten.

Reproduce with the frozen checkout, the existing calibration directory, and new
output directories:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/prepare_sparse_joint_inputs.py \
  docs/assets/broader_models_20261029/plan.json /tmp/broader-inputs CALIBRATION_ROOT
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/run_broader_camera_lidar_comparison.py \
  docs/assets/broader_models_20261029/plan.json /tmp/broader-inputs /tmp/broader-results
```
