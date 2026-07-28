<!-- Generated from slac.benchmark/v0.1; do not edit by hand. -->

## ETHZ real robot-world hand-eye AX=YB native and OpenCV benchmark

| Method | Holdout rotation RMSE deg ↓ | Holdout translation RMSE mm ↓ | Failure rate | Runtime s |
|---|---:|---:|---:|---:|
| Calibrex Shah | 0.624739 [0.559308, 0.688861] | 10.7793 [9.53397, 11.9952] | 0.0% | 0.0206115 |
| Calibrex Li-Wang-Wu | 0.627131 [0.561467, 0.692796] | 19.651 [15.2668, 23.3423] | 0.0% | 0.0321106 |
| Calibrex Dornaika-Horaud | 0.624742 [0.559311, 0.688865] | 10.7793 [9.53399, 11.9952] | 0.0% | 0.0286939 |
| Calibrex Zhuang-Roth-Sudhakar | 0.651861 [0.569557, 0.733996] | 10.961 [9.65144, 12.1956] | 0.0% | 0.0213057 |
| Calibrex D-H nonlinear | 0.6265 [0.560987, 0.692012] | **10.7752 [9.52577, 12.0161]** | 0.0% | 0.217955 |
| OpenCV Shah | **0.624739 [0.559308, 0.688861]** | 10.7793 [9.53397, 11.9952] | 0.0% | 0.00343018 |
| OpenCV Li | 0.627131 [0.561467, 0.692796] | 19.651 [15.2668, 23.3423] | 0.0% | 0.00570322 |

Shared protocol: `ethz-real-hand-eye/ax-yb/absolute-pose-split/v0.1`; 5 split(s); 5000 paired bootstrap samples.

Limitations:

- The dataset has no accepted ground-truth extrinsic; closure is consistency, not absolute accuracy.
- AX=YB results are ranked separately from the AX=XB equation family.
- Runtime is single-process wall time on the recorded host; Python peak memory excludes native allocator visibility.
