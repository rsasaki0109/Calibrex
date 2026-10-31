# KITTI camera-LiDAR v3: cross-start agreement rule (development results)

[Protocol](kitti_camera_lidar_v3_start_agreement.md) frozen at `98262f84` before
any replay of the rule on real inputs. All 55 spent drives were screened once:
23 drives from cached inputs and 32 drives re-captured with the batch tool.
There were no capture, parity or scoring failures. The current-rule replay
matched the captured evidence on all 55 drives and again equals v2's committed
aggregate.

**Frozen verdict: `not_carried_forward`.** Nine of ten checks pass; the 3 deg
detection check fails.

| Measure, 55 spent drives | Current rule | Agreement rule (0.5 deg) | Check |
| --- | ---: | ---: | --- |
| Baselines flagged | 0 | 0 | pass |
| Maximum judged baseline axis error | 0.450 deg | 0.450 deg | pass (≤ 0.75) |
| Maximum judged turned-run axis error | 1.503 deg | **0.532 deg** | pass (≤ 1.5) |
| Turned-run judged axes beyond 0.5 deg | 10 | **2**, none new | pass, pass |
| Judged baseline axes | 125/156 | 124/156 | pass (≥ 0.5) |
| 3 deg turns flagged | 122/125 | **82/124 (66.1 %)** | **fail** (≤ 10 % missed) |
| 1 deg turns flagged | 125/125 | 117/124 (94.4 %) | pass (≤ 25 % missed) |
| Scored drives | 52 | 52 | pass |

The rule removes eight large errors:

- 2011_09_28_0002: roll on roll +3, pitch +3, yaw +3 and pitch +1, the v2 failure;
- 0057 pitch +3: roll and pitch;
- 0023 yaw +3: yaw;
- 2011_09_28_0043 roll +3: yaw.

Two remain, both just above 0.5 deg: 0093 yaw +1 and 2011_09_30_0016 pitch +3.
It removes errors by **withholding** axes. In a turned run, starts often reach
different optima, so the run judges fewer axes and is not flagged. The
3.5-margin trade-off was the opposite: it kept detections and moved an error.
Withholding instead gives up detections.

## Post hoc diagnostic (after the verdict; changes nothing)

Counted from the published proposed-rule artifacts: a run counts if any axis
was withheld because starts disagreed.

| Runs | With a withheld axis |
| --- | ---: |
| Baselines | 1/55 (0029) |
| 3 deg turned runs | 55/165 |
| 1 deg turned runs | 17/165 |
| Counted 3 deg trials not flagged by the rule | **40/42** |
| Counted 1 deg trials not flagged by the rule | **7/7** |

Cross-start disagreement therefore marks almost every missed turn. A rule
could report agreeing axes only and flag a run whenever starts disagree. On
these numbers it would flag about 122/124 of 3 deg and 124/124 of 1 deg trials.
It would leave the two errors near 0.5 deg, but it would flag the 0029 baseline,
where v2 allows none. This is a **hypothesis for a new frozen protocol**, read
from data that were used to motivate it. It is not a result. The 0029 cost
and the 55 spent drives' exposure must be stated in that protocol.

## Shipped afterwards as an opt-in check warning

The user chose to ship the flagging variant as a practical, opt-in `calibrex check` option
(`--camera-lidar-start-agreement-deg`, see the [check page](camera_lidar_check.md#experimental-cross-start-disagreement-warning-2026-10-31-opt-in)),
not as a v3 claim. The shipped verdict code was applied to the 385 published
artifacts above, with each withheld axis restored from its recorded reason:

- KITTI calibration warned on 1/55 baselines (0029 yaw);
- 3 deg turns get a warn or fail verdict in 122/124 counted runs, 1 deg in 124/124;
- judged-axis errors are those of the agreement rule above.

[Measurement](../assets/kitti_camera_lidar_v3/start_agreement_20261031/disagreement_warning.json).
Re-running the shipped estimator on 0029 and 0002 matched the restored artifacts
in all 14 runs ([spot check](../assets/kitti_camera_lidar_v3/start_agreement_20261031/disagreement_spot_check.json)).
These are the data that suggested the variant, so the rates describe the feature,
not its accuracy on new recordings.

## Evidence

- **Screen and inputs.** The
  [screen](../assets/kitti_camera_lidar_v3/start_agreement_20261031/screen.json)
  records all checks, both rules' metrics, the large-error lists, the input
  digests and the frozen screen tool.
- **Cached stage.** Paired [cached replays](../assets/kitti_camera_lidar_v3/start_agreement_20261031/cached/manifest.json),
  with the runner source.
- **Re-capture stage.** [Re-capture receipts](../assets/kitti_camera_lidar_v3/start_agreement_20261031/recaptured/manifest.json).
  The first launch of all 32 drives stopped at the free-disk preflight. Those
  32 attempts are retained, and every drive passed on its second attempt.
- **No sensor data published.** No NPZ arrays, images, clouds or bags are
  included.

Core defaults and CLI are unchanged; the option stays opt-in and development-only.
