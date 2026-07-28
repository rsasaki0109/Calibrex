<!-- Generated from slac.benchmark/v0.1; do not edit by hand. -->

## ETHZ real robot-world hand-eye shared-split benchmark

| Method | Rotation holdout RMSE deg ↓ | Translation holdout RMSE mm ↓ | Known-bad detection fraction ↑ | Failure rate | Runtime s |
|---|---:|---:|---:|---:|---:|
| Shah | **0.585795** | 10.0628 | **1** | 0.0% | — |
| Li-Wang-Wu | 0.58728 | 17.458 | **1** | 0.0% | — |
| Dornaika-Horaud | 0.585795 | 10.0628 | **1** | 0.0% | — |
| Zhuang-Roth-Sudhakar | 0.590747 | 10.132 | **1** | 0.0% | — |
| Calibrex nonlinear refinement | 0.58721 | **10.0394** | **1** | 0.0% | — |

Shared protocol: `ethz-real-robot-world-hand-eye/shared-split/v0.1`; 1 split(s); 2000 paired bootstrap samples.

Limitations:

- Metrics are closure errors on held-out pose pairs, not ground-truth extrinsic errors.
- This historical run contains one shared split, so its bootstrap interval cannot measure split-to-split uncertainty.
- Per-method runtime and peak memory were not recorded by the original combined run and remain explicitly unavailable.
- The scoped win is translation holdout RMSE on this dataset and split; Shah has the lowest rotation holdout RMSE.
- All compared solvers are independent native Calibrex implementations of the cited methods.
