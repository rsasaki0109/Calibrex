# Stereo/LiDAR geometry has overlap but inconsistent local surfaces

Frozen `c7ddb8c6`, evaluated 2026-10-30. See
[protocol](stereo_geometry_diagnostic_protocol.md),
[plan](../assets/stereo_geometry_diagnostic_20261030/plan.json),
[all 18 frames](../assets/stereo_geometry_diagnostic_20261030/evaluation.json) and
[schema](../assets/stereo_geometry_diagnostic_20261030/evaluation.schema.json).

At reference poses, camera-visible LiDAR points generally find nearby stereo
geometry. However, local normal agreement is limited despite nearly all stereo
neighborhoods passing the normal eigengap test. This separates usable overlap
from correctness of local surfaces; it does not identify a single ICP failure cause.

| Per-frame statistic | Minimum | Median across frames | Maximum |
| --- | ---: | ---: | ---: |
| All-source fraction within .2 m | 8.30% | 9.59% | 12.61% |
| All-source fraction within .5 m | 12.64% | 14.33% | 21.74% |
| All-source fraction within 1 m | 15.77% | 17.82% | 26.82% |
| Camera-FOV source fraction within .2 m | 44.58% | 46.97% | 64.28% |
| Camera-FOV source fraction within .5 m | 66.17% | 71.17% | 82.50% |
| Camera-FOV source fraction within 1 m | 80.97% | 86.15% | 93.89% |
| Source valid PCA-normal fraction | 88.29% | 94.33% | 96.40% |
| Target valid PCA-normal fraction | 98.75% | 99.31% | 99.48% |
| Supported normal-pair count | 4397 | 5211 | 7446 |
| Normal pairs within 15 degrees | 28.90% | 49.22% | 63.64% |

Normal pairs require camera FOV, source-to-target distance <=1 m, and a valid
12-neighbor normal on both clouds. The angle uses abs(dot) to ignore normal sign.
The 15-degree percentage is conditional on those supported normal pairs, not all
LiDAR points. Per-frame median FOV nearest-target distance spans .1350–.2395 m.
Mutual nearest pairs within 1 m number 2303–3719 per frame. Reverse-target distance
quantiles and all explicit populations are retained in the machine report.

A low overall-source overlap fraction partly reflects the LiDAR's wider view;
it must not be interpreted as equivalent failure inside the image. Conversely,
81–94% coverage at a 1 m radius does not imply sub-.20 m calibration accuracy
or correct physical associations. First-point .2 m voxel discretization, stereo
noise, different sampling density, occlusion and motion can affect nearest distances
and normals. A valid PCA eigengap only establishes a locally defined plane, not a
correct plane. This diagnostic does not prove the source normals are ground truth.

## Every frame

| Drive / frame | FOV count | Median distance (m) | Within 1 m | Normal pairs | Within 15 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2011_09_26_0005 / frame-38 | 6941 | 0.2371 | 86.01% | 5723 | 28.90% |
| 2011_09_26_0005 / frame-76 | 7537 | 0.2226 | 85.76% | 6279 | 33.40% |
| 2011_09_26_0005 / frame-114 | 6906 | 0.2219 | 93.89% | 6041 | 37.25% |
| 2011_09_28_0002 / frame-93 | 7205 | 0.2395 | 82.10% | 4674 | 47.75% |
| 2011_09_28_0002 / frame-187 | 7518 | 0.2340 | 80.97% | 4698 | 50.68% |
| 2011_09_28_0002 / frame-281 | 7551 | 0.2346 | 81.42% | 4526 | 53.69% |
| 2011_09_26_0009 / frame-111 | 5776 | 0.1783 | 92.68% | 4934 | 34.15% |
| 2011_09_26_0009 / frame-223 | 6649 | 0.2274 | 85.64% | 5211 | 40.15% |
| 2011_09_26_0009 / frame-334 | 5934 | 0.1983 | 89.06% | 4770 | 39.62% |
| 2011_09_26_0011 / frame-58 | 7658 | 0.1893 | 87.23% | 5771 | 55.73% |
| 2011_09_26_0011 / frame-116 | 6794 | 0.2084 | 86.30% | 4432 | 53.34% |
| 2011_09_26_0011 / frame-174 | 7082 | 0.2106 | 84.44% | 4397 | 54.17% |
| 2011_09_26_0013 / frame-35 | 6861 | 0.2151 | 85.48% | 5423 | 59.08% |
| 2011_09_26_0013 / frame-71 | 6330 | 0.1648 | 87.73% | 5211 | 55.19% |
| 2011_09_26_0013 / frame-107 | 6021 | 0.1350 | 87.81% | 4953 | 63.64% |
| 2011_09_26_0015 / frame-74 | 9092 | 0.2291 | 87.85% | 7446 | 46.87% |
| 2011_09_26_0015 / frame-148 | 8788 | 0.2285 | 85.23% | 7153 | 45.77% |
| 2011_09_26_0015 / frame-222 | 8217 | 0.2223 | 88.84% | 6880 | 55.83% |

Four focused tests, repository Ruff and strict documentation build pass. Checks
verify report/schema, all18 frames, all source/input hashes and unchanged prepared
arrays. Tests include identical planes (zero distance and sign-invariant normal
agreement) and empty-FOV/no-overlap denominator handling. No registration rerun,
accuracy verdict change, production mask or reference-derived fit population occurs.

Next prioritize a correspondence quality gate that can be computed from sensor
measurements, accounting for partial overlap and stereo surface reliability.
First specify it on independent controlled geometry with known depth noise and
occlusion; test whether high-confidence surfaces are actually correct and retain
sufficient spatial/directional support. These spent real reference diagnostics
must remain scoring evidence, not training labels for hand-tuning a gate. Compare
that frozen construction to the current unfiltered pairs before a new ICP run.
Do not merely extend iteration limits, assume eigengap validity equals accuracy,
or discard frames with poor agreement. Full-cloud MI remains the existing baseline;
no new SOTA or adoption claim is justified.
