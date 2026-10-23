# KITTI projected-azimuth orientation probe

**Post hoc development; this proxy is not adopted.** This experiment extends
the [nuisance-profiled controls](kitti_camera_lidar_profile_controls.md) with
image-gradient direction. The existing scalar edge/discontinuity cues have
[Levinson–Thrun, RSS 2013](https://www.roboticsproceedings.org/rss09/p29.html)
as a primary source; the direction penalty below is an independent development
hypothesis, not a reproduction of that paper.

## Fixed model and scope

The diagnostic constructs a polarity-independent image-gradient tensor using
the scalar objective's image reduction, Gaussian blur, Sobel derivatives and
99th-percentile gradient normalization. Its trace supplies bounded gradient
confidence; no additional pixel-strength threshold is selected.

For a prepared LiDAR point `(x, y, z)`, the azimuth-motion proxy is `(-y, x, 0)`.
Both point and proxy are rotated by the trial transform. The pinhole projection
derivative supplies its image direction, and a tensor cosine-squared term
measures compatibility with the local image gradient. This is **scan direction,
not a recovered boundary normal**. It need not align with a true image edge,
and deriving it from motion-compensated coordinates adds another approximation.
Only organized frames and undistorted pinhole cameras are supported. Image
gradient orientation is sampled locally; it is not transported along the spill.

The score subtracts `raw_response * confidence * (1 - compatibility)` from
the original signed response. Thus orientation mismatch cannot turn a negative
background score into a reward: only the nonnegative foreground contribution
is attenuated, while the original background penalty remains. Flat gradients
and zero projected directions are neutral. Point support, denominator and
weights stay fixed, including candidate-outside points entering a trial image.

```bash
python -m tools.kitti_camera_lidar_profile_probe SAVED_SNAPSHOT \
  --conditional-report PREVIOUS_CONDITIONAL_REPORT.json \
  --multi-start --azimuth-orientation --output NEW_DIRECTORY
```

Orientation and the previously rejected spatial balancing are mutually
exclusive ablations. Both default to off. Training frames select nuisance
starts; held-out blocks score only fixed poses. Initialization still comes
from the pooled conditional fit, so these are not nested validation folds.

## Preselected six-input comparison

Source `c237f78d` was committed before launch. The producer fixed baseline,
pitch +3 and roll +3 on each of the two cached spent drives before reading
the new outcomes: **six inputs, eighteen axis-control comparisons**. These
post hoc probes target the known detection/false-confidence cases. They do
not cover yaw +3, one-degree controls or the full fourteen-input recipe.

Two independent worker processes ran the inputs, with BLAS thread limits
requested as one via environment variables. Timing is an execution receipt,
not a speed benchmark. All six completed without errors, checking identical
snapshot/NPZ hashes, conditional receipts and optimizer settings against their
previous scalar multiple-start reports. Source bytes and Git HEAD remained
unchanged. Different objectives can produce different baseline fold poses;
pose parity across the ablation is not asserted.

| Drive, selected three inputs only | Scalar multiple-start controls detected | Orientation-proxy controls detected |
| --- | ---: | ---: |
| 2011_09_26_0005 | 8/9 | 0/9 |
| 2011_09_28_0002 | 0/9 | 0/9 |
| Total | 8/18 | 0/18 |

These counts are **control outcomes, not calibration accuracy, observed-axis
coverage or native verdicts**. Pitch +3 on 0002 remains undetected. The known
roll +3 false control also remains rejected, but the scalar multiple-start
method already rejected it. Losing every previously detected 0005 control
does not constitute a useful confidence improvement. The scan-direction proxy
is not recommended or promoted, and no threshold is relaxed after the result.
No new SOTA conclusion follows from six deliberately selected spent inputs.

A separate replay of the spent 0005 baseline, with orientation/spatial flags
off and multiple starts on, matches the previous scalar report's full numerical
evidence excluding provenance within `1e-9`. This checks that the revised
objective factory preserves that existing default path.

## Receipts and next step

The [batch](../assets/kitti_camera_lidar_v3/orientation_proxy/report.json),
[default confirmation](../assets/kitti_camera_lidar_v3/orientation_proxy/default_confirmation/confirmation.json)
and [manifest](../assets/kitti_camera_lidar_v3/orientation_proxy/manifest.json)
bind schemas, all six probe outputs, the scalar replay and exact source/producer
bytes, about 373 KB excluding the manifest. Images, point clouds, gradient
tensors, NPZ payloads and bags remain local; no data was downloaded.

Tests cover gradient polarity, scale, weak/unknown orientation, rotated scan
direction, signed-score behavior, margin support, unsupported inputs, frozen
input/source bindings, the negative comparison and scalar-path parity.
Broader orientation claims require genuine boundary geometry rather than
assuming that scan direction is a boundary normal. Further development must
also preserve turn sensitivity. Capture progress remains **2/55 spent drives**;
the remaining comparisons, exposure audit and independent pre-registration
are still pending. Existing defaults and SOTA standings are unchanged.
