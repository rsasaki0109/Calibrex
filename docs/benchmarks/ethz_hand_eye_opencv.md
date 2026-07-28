# ETHZ native and OpenCV hand-eye benchmarks

These benchmarks execute all 13 Calibrex native hand-eye methods and the seven
OpenCV 4 hand-eye variants on common, leakage-safe splits of the ETHZ ASL real
robot-arm pose streams. The two equations use different estimands, so Calibrex
publishes and ranks them separately.

## Hand-eye `AX=XB`

{% include-markdown "../assets/ethz-hand-eye-ax-xb-benchmark.md" %}

## Robot-world hand-eye `AX=YB`

{% include-markdown "../assets/ethz-hand-eye-ax-yb-benchmark.md" %}

## Protocol

- Dataset: ETHZ ASL `robot_arm_w_color_camera_real.zip`, DOI
  [`10.3929/ethz-c-000788527`](https://doi.org/10.3929/ethz-c-000788527)
- Source archive SHA-256:
  `2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3`
- Repetitions: seeds 0 through 4
- Per split: 32 fit and 16 holdout absolute poses
- Leakage control: absolute poses are split before relative motions are built
- `AX=XB` native input: the exact pairwise relative-motion convention used by
  OpenCV 4 (`inv(A_j) A_i`, `inv(B_j) B_i`)
- Tuning: fixed defaults; outer holdout data is evaluation-only
- Failure policy: every method-by-split failure remains in the denominator
- OpenCV boundary: optional `opencv-python-headless>=4.8,<5`, Apache-2.0;
  version `4.13.0.92` in the committed run

The five split means are accompanied by deterministic 95% bootstrap intervals.
The pairwise relative motions share source poses, so they are not statistically
independent observations; the split, rather than each pair, is the resampling
unit.

## Interpretation

On `AX=XB`, Calibrex Andreff has the lowest mean rotation closure RMSE and
Calibrex Daniilidis has the lowest mean translation closure RMSE. The native
Horaud-Dornaika nonlinear refinement does not win either metric and is much
slower; this negative result is retained.

On `AX=YB`, OpenCV Shah and native Shah are numerically equivalent on rotation.
The native nonlinear refinement has the lowest mean translation closure RMSE,
but its intervals overlap the Shah and Dornaika-Horaud results. No accepted
ground-truth extrinsic is supplied for this run, so closure measures equation
consistency rather than absolute calibration accuracy.

## Reproduce

```bash
python3 tools/download_public_dataset.py ethz_hand_eye_robot_arm_real
python3 tools/generate_ethz_hand_eye_benchmarks.py \
  data/public/ethz_hand_eye_robot_arm_real/robot_arm_w_color_camera_real.zip \
  --output-dir docs/assets \
  --seeds 0,1,2,3,4 \
  --fit-count 32 \
  --holdout-count 16
```

Each equation family has a raw benchmark definition, an aggregated
`slac.benchmark/v0.1` artifact, and generated Markdown in `docs/assets`. Every
trial records the input/config/output digests, runtime, peak Python memory,
method identity, paper DOI, tool version, license, and source archive digest.
