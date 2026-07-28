# ETHZ real robot-world hand-eye benchmark

Calibrex recomputes five `AX=YB` methods on the public ETHZ ASL real
robot-arm pose streams. Every method uses the same 1,350 fit pairs and 338
held-out pairs.

{% include-markdown "../assets/ethz-robot-world-hand-eye-benchmark.md" %}

The nonlinear refinement reduces translation holdout RMSE from the
Dornaika–Horaud closed-form initialization by **0.23%**. Shah has the lowest
rotation holdout RMSE, by less than `0.000001°` relative to the closed-form
Dornaika–Horaud result.

This benchmark supports only the scoped claim above. Its metrics are
rotation/translation closure errors on unseen pose pairs, not errors against a
ground-truth extrinsic. The run therefore does not establish that one method
has universally better calibration accuracy.

## Protocol

- Dataset: ETHZ ASL `robot_arm_w_color_camera_real.zip`
  ([dataset page](https://projects.asl.ethz.ch/datasets/hand-eye-calibration-2017/),
  DOI [`10.3929/ethz-c-000788527`](https://doi.org/10.3929/ethz-c-000788527))
- Pairing: disjoint absolute-pose pairs
- Split: seeded shared 80/20 fit/holdout split
- Fit pairs: 1,350
- Holdout pairs: 338
- Known-bad controls: ±5 cm translation and ±5° rotation probes on both
  recovered transforms, 24 per method
- External solver code: not executed
- Verdict thresholds: rotation RMSE ≤ 2°; translation RMSE ≤ 30 mm;
  known-bad detection ≥ 75%

All methods passed the three gates. The table highlights the minimum of each
holdout metric; it does not combine unlike units into an unreported aggregate
score.

## Methods

- Shah, *Solving the Robot-World/Hand-Eye Calibration Problem Using the
  Kronecker Product*, DOI
  [`10.1115/1.4024473`](https://doi.org/10.1115/1.4024473)
- Li, Wang, and Wu, *Simultaneous Robot-World and Hand-Eye Calibration Using
  Dual-Quaternions and Kronecker Product*, DOI
  [`10.5897/IJPS.9000501`](https://doi.org/10.5897/IJPS.9000501)
- Dornaika and Horaud, *Simultaneous Robot-World and Hand-Eye Calibration*,
  DOI [`10.1109/70.704233`](https://doi.org/10.1109/70.704233)
- Zhuang, Roth, and Sudhakar, *Simultaneous Robot/World and Tool/Flange
  Calibration by Solving Homogeneous Transformation Equations of the Form
  AX=YB*, DOI [`10.1109/70.313105`](https://doi.org/10.1109/70.313105)
- Calibrex nonlinear refinement: a native nonlinear refinement initialized
  from the Dornaika–Horaud closed-form estimate; no external implementation is
  copied or executed

## Reproduce

Download the declared archive, run the comparison, then verify the evidence
bundle:

```bash
python3 tools/download_public_dataset.py ethz_hand_eye_robot_arm_real
calibrex calibrate \
  examples/public_datasets/ethz_hand_eye_robot_arm_real/config.yaml \
  --output-dir outputs/ethz_hand_eye_robot_arm_real
calibrex verify outputs/ethz_hand_eye_robot_arm_real/bundle.json
```

The committed raw
[`benchmark definition`](../assets/ethz-robot-world-hand-eye-benchmark.definition.json)
is aggregated into
[`ethz-robot-world-hand-eye-benchmark.json`](../assets/ethz-robot-world-hand-eye-benchmark.json)
and the table included above by `calibrex benchmark`. Both artifacts record
the exact values, source digests, run identity, method citations, and
reproduction command. They validate against the generated
[`benchmark_definition`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/benchmark_definition.schema.json)
and
[`benchmark`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/benchmark.schema.json)
JSON Schemas.
