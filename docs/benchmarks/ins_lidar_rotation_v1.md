# INS-LiDAR rotation audit v1: refuted, 6 of 8 gates

The first pre-registered INS-LiDAR rotation audit is **refuted**. All four
calibration commands completed, but KITTI `2011_09_29_0071` produced no
data-estimated rotation axis. The claim required a scorable rotation on every
recording and evaluable controls; that recording fails both integrity gates.
The supported-pair count remains **three**. Seven pairs now have an audit.

On the eight axes that were estimated, the maximum difference to a dataset
reference was **0.2857285446678126 deg**, below the frozen 0.5 deg ceiling.
All eight observed-axis 1 deg held-out controls were detected. Those successes
do not override the failed requirements. Dataset calibration is a reference
estimate, not independent metrology. No translation, full-extrinsic, absolute
clock-accuracy or general superiority claim is made.

## Frozen contract and execution

- [Pre-registration](ins_lidar_rotation_v1_preregistration.yaml), merged in
  [PR #301](https://github.com/rsasaki0109/Calibrex/pull/301) at
  **2026-10-22 11:33:32 UTC**, before evaluation sensor downloads.
- Frozen code: `5cc11d43e313bec6d29f3de5e549493eee594352`.
- Evaluation checkout: merged `main`
  `e2ac3e20e9e3bcc0aa0aeaa3f93f2a25c374292f`; `src/` and `tools/` match the freeze.
  The imported package path and numerical-library versions were checked before
  input acquisition.
- First 600 scans of every declared recording; default LiDAR-only odometry and
  native joint trajectory hand-eye fitting, with no priors. Reference extrinsics
  enter only the comparison. Clock offset and reference scale are nuisance fits.
- Every method case ran once. All 608 recorded object/whole-fetch acquisition
  attempts passed; no acquisition retry or method rerun occurred. Inputs were
  re-hashed before fitting, after fitting, and by the frozen audit builder.
- Two dataset families and two physical platforms: all KITTI dates share one
  platform; Boreas supplies the other. The registration discloses earlier
  camera-LiDAR reads of the selected KITTI streams.

The [development record](../research/ins_lidar_sota_plan.md) preserves threshold
derivation and earlier development-only transfer failures separately.

## Results

| Recording | Estimated rotation axes / 3 | Maximum observed-axis reference difference | Detected observed-axis controls |
| --- | ---: | ---: | ---: |
| KITTI `2011_09_29_0071` | 0/3 | not scorable | not evaluable |
| KITTI `2011_09_30_0020` | 2/3 | 0.285729 deg | 2/2 |
| KITTI `2011_09_30_0027` | 3/3 | 0.157743 deg | 3/3 |
| Boreas `2020-12-01-13-26` | 3/3 | 0.132141 deg | 3/3 |

All twelve declared rotation axes stay in the coverage denominator. KITTI
observed **5/9 axes**, and Boreas **3/3**. The descriptive scorer's
`recording_completion=1.0` means a converged, structurally complete artifact;
it does **not** mean a nonempty rotation estimate. The frozen benchmark marks
the zero-axis recording as a failed scorable case. Thus nonempty rotation
availability is **3/4**, while command/artifact completion is 4/4.

| Family | Requirement | Observed | Frozen bound | Verdict |
| --- | --- | ---: | ---: | --- |
| KITTI | Failed scorable-recording fraction | 1/3 | <= 0 | contradicted |
| KITTI | Maximum observed-axis reference difference | 0.285729 deg | <= 0.5 deg | achieved |
| KITTI | Mean unobserved-axis fraction | 4/9 | <= 0.5 | achieved |
| KITTI | Maximum per-recording control-miss fraction | 1 | <= 0 | contradicted |
| Boreas | Failed scorable-recording fraction | 0 | <= 0 | achieved |
| Boreas | Maximum observed-axis reference difference | 0.132141 deg | <= 0.5 deg | achieved |
| Boreas | Mean unobserved-axis fraction | 0 | <= 0.5 | achieved |
| Boreas | Maximum per-recording control-miss fraction | 0 | <= 0 | achieved |

The control value **1** for KITTI is the pre-registered worst-case assignment
for a recording with no observed axes, not a measured miss among the eight
evaluable controls. Empty evidence cannot count as perfect detection. The
coverage gate passes at the family level; the separate all-recordings and
control-integrity gates still fail. Eight finite controls do not establish
population sensitivity or future detection guarantees.

## Post hoc inspection of 0071

This inspection explains the immutable verdict; it does not change any gate,
axis status, score or result. The saved artifact reports rotation uncertainties
of **0.378, 0.226 and 1.225 deg** for roll, pitch and yaw, all above the default
**0.1 deg** observability ceiling. Its analytic/jackknife larger-of rule correctly
leaves every rotation unobservable, despite numerical solver convergence.

The unobserved pitch and yaw differ from the reference by about **0.558** and
**1.976 deg**. Its roll and yaw held-out control delta chi-squares are negative.
These quantities were not added to the observed-axis agreement score. Simply
promoting uncertain axes would not demonstrate reliable calibration. The cause
has not been isolated: motion excitation, registration failures (597/599
converged), model mismatch and block stability remain development questions.

Next research should test whether additional informative motion or a frozen
data-sufficiency policy can produce reliable estimates on new recordings.
All four windows are now spent for INS-LiDAR evaluation. They may be used for
development; another audit needs new held-out data and a new pre-registration.

## Evidence and replay

The [publication manifest](../assets/ins_lidar_sota_readiness/evaluation/publication.json)
pins the original bytes of the native artifacts, input receipts, transfer
histories, commands, execution environment, CI/merge record, benchmarks,
protocol and audit. Raw sensor payloads stay outside Git. Primary datasets are
[KITTI raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) and
[Boreas](https://github.com/utiasASRL/pyboreas); licenses and source pins are in
the registration and benchmark artifacts.

- [Scores](../assets/ins_lidar_sota_readiness/evaluation/audit/scores.json)
- [Frozen protocol](../assets/ins_lidar_sota_readiness/evaluation/audit/ins_lidar_sota_protocol.yaml)
- [Original audit](../assets/ins_lidar_sota_readiness/evaluation/audit/ins_lidar_sota_audit.yaml)
- [Run manifest](../assets/ins_lidar_sota_readiness/evaluation/runs/run.json)
- [Transport history](../assets/ins_lidar_sota_readiness/evaluation/transport-history.json)
- [CI and merge proof](../assets/ins_lidar_sota_readiness/evaluation/execution/ci-proof.json)

Replay the verdict from the published benchmarks, without running calibration:

```bash
calibrex sota audit \
  docs/assets/ins_lidar_sota_readiness/evaluation/audit/ins_lidar_sota_protocol.yaml \
  --output /tmp/ins_lidar_rotation_v1_replay.yaml
```

Exit **2** means the claim is refuted. The published replay reproduces the same
six achieved and two contradicted requirements. The benchmark zero anchor is a
mathematical schema fixture, not an empirical competing calibration method.
