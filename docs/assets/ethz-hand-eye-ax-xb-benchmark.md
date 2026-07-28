<!-- Generated from slac.benchmark/v0.1; do not edit by hand. -->

## ETHZ real hand-eye AX=XB native and OpenCV shared-split benchmark

| Method | Holdout rotation RMSE deg ↓ | Holdout translation RMSE mm ↓ | Failure rate | Runtime s |
|---|---:|---:|---:|---:|
| Calibrex Park-Martin | 0.867227 [0.759962, 0.958155] | 13.8759 [11.7678, 15.8266] | 0.0% | 0.153451 |
| Calibrex Tsai-Lenz | 0.875291 [0.77004, 0.967317] | 13.8652 [11.7326, 15.8293] | 0.0% | 0.126921 |
| Calibrex Daniilidis | 0.868969 [0.761845, 0.960775] | **13.8212 [11.9121, 15.6887]** | 0.0% | 0.322403 |
| Calibrex Andreff | **0.867199 [0.759958, 0.958223]** | 13.879 [11.7674, 15.8309] | 0.0% | 0.334728 |
| Calibrex Shiu-Ahmad | 1.6191 [0.787831, 3.10277] | 31.9219 [12.7334, 52.761] | 0.0% | 14.1537 |
| Calibrex Chou-Kamel | 0.867205 [0.759953, 0.958241] | 13.8795 [11.7684, 15.8315] | 0.0% | 0.129661 |
| Calibrex Horaud-Dornaika | 0.890804 [0.782116, 0.996268] | 14.1434 [11.8641, 16.1088] | 0.0% | 0.139859 |
| Calibrex H-D nonlinear | 0.885789 [0.777979, 0.991045] | 14.0634 [11.8408, 16.0027] | 0.0% | 19.4512 |
| OpenCV Tsai | 0.871872 [0.769051, 0.961725] | 13.8681 [11.8255, 15.6992] | 0.0% | 0.0368567 |
| OpenCV Park | 0.867225 [0.759969, 0.958144] | 13.8754 [11.7667, 15.8261] | 0.0% | 0.0280751 |
| OpenCV Horaud | 0.867203 [0.759961, 0.958229] | 13.879 [11.7674, 15.831] | 0.0% | 0.0248639 |
| OpenCV Andreff | 0.868623 [0.763653, 0.959819] | 15.4855 [13.6068, 17.338] | 0.0% | 0.0327896 |
| OpenCV Daniilidis | 0.871005 [0.764385, 0.963618] | 13.9265 [11.9635, 15.6961] | 0.0% | 0.0284915 |

Shared protocol: `ethz-real-hand-eye/ax-xb/absolute-pose-split/v0.1`; 5 split(s); 5000 paired bootstrap samples.

Limitations:

- The dataset has no accepted ground-truth extrinsic; closure is consistency, not absolute accuracy.
- Pairwise relative motions share absolute poses and therefore are not independent samples.
- Runtime is single-process wall time on the recorded host; Python peak memory excludes native allocator visibility.
