# Complete LiDAR–wheel evaluation preparation

The complete prepared-input evaluation phase falls from **2.389020 to 0.491396
seconds (4.861703x)**. The whole fresh worker, including imports and audit
instrumentation, falls from **5.353725 to 3.397488 seconds (1.575789x)**.
Startup remains substantial; the larger ratio applies to the calibration API.
See [the benchmark](../../benchmarks/lidar_wheel_evaluation_preparation.md).

All 2,762 original OXTS poses and proxy samples of KITTI
`2011_09_30_drive_0018_sync` yield 2,760 generated sensor motions in 29 blocks.
The common time origin is removed to exercise finite-difference clock steps.
KITTI has no wheel odometry: OXTS `vf` and `wu` are proxies. These are generated
sensor motions, without raw LiDAR registration or independent accuracy evidence.
The unchanged policy is **warn**, including a clock control that is not detected
and unobservable roll. No accuracy or GPU acceleration claim is made.

This archive retains the original packets/timestamps and download provenance,
prepared control and generator, both complete core source capsules/manifests,
measurement code and schemas, all six fresh runs, every residual request and
least-squares result, official artifacts, and validation receipts.
`index.json` binds every publication byte other than itself.

The audit independently reconstructs all inputs, train/holdout/jackknife
subsets, all 301 scan candidates per fit, selection visits and exact clock
LRU work. All results, residual digests and full Jacobian digests remain exact.
Only offline motion regeneration allows platform roundoff: `rtol=0`,
`2e-7` m/s velocity and `1e-12` rad/s angular rate. Original proxy samples,
timestamps, IDs and raw bytes remain exact. Generator source comparison allows
CRLF/LF differences only; its archived original bytes retain a strict hash pin.

`qa/cohort-selection.json` records all timings and hashes from the earlier,
excluded absolute-time cohort. Its clock finite differences were unresolved;
the selected cohort exercises 13–15 distinct clock states per fit. Both use
the same core source. The selected cohort is the sole published speed claim.

KITTI attribution: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel
Urtasun. Original data and generated derivatives retain CC BY-NC-SA 3.0:
[dataset](https://www.cvlibs.net/datasets/kitti/) and
[license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Calibrex source snapshots and measurement code retain Apache-2.0.

```bash
python -m tools.benchmark_lidar_wheel_preparation \
  --audit-only docs/assets/lidar_wheel_evaluation_preparation
```
