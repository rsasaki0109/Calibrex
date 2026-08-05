# Solid-state LiDAR synthetic ground-truth benchmark

- Schema: `slac.solid_state_synthetic_benchmark/v0.1`
- Generator: `tools/run_solid_state_synthetic_benchmark.py`
- Generator SHA-256: `16b4220ebad999968f8300eca75e193447f76de8020103239ad11e0d0c892389`

PASS

This is a deterministic solver gate, not a public-dataset accuracy claim.

| Case | Expected | Status | Rotation error (deg) | Translation error (m) | Clock error (ms) | Gate | Detected |
|---|---|---|---:|---:|---:|---|---|
| joint_extrinsic_clock_reference | pass | converged | 0.000000 | 0.000000 | 0.000 | PASS | yes |
| fixed_clock_known_bad_control | fail | converged | 1.216728 | 0.034480 | 30.000 | FAIL | yes |

## Interpretation

- The reference case must pass all absolute parameter-error gates.
- The fixed-clock control must fail the time-error gate.
- Passing here does not replace independent extrinsic/clock ground truth on a real sensor pair.
