# Short-baseline neighbors restore image-only motion support

Acquisition scope frozen `550055c0`; exact byte ranges frozen `ce811402` before
payload reads. Audit freeze `11f93d00`, input preprocessing reconciliation
`79ce9e80`. See [protocol](temporal_neighbors_protocol.md),
[metadata index](../assets/temporal_neighbors_20261030/index.json),
[acquisition manifest](../assets/temporal_neighbors_20261030/acquisition.json),
[acquisition schema](../assets/temporal_neighbors_20261030/acquisition.schema.json),
[final audit plan](../assets/temporal_neighbors_20261030/audit-plan.json),
[all36 pair results](../assets/temporal_neighbors_20261030/evaluation.json),
[result schema](../assets/temporal_neighbors_20261030/evaluation.schema.json) and
[verification](../assets/temporal_neighbors_20261030/validation.json).

Image-only directional-support screening passes **28/36 short-baseline pairs**.
Intervals are .103129–.103757 seconds, versus3.728–11.590 seconds for the earlier
sparse triplets (1/12 passed). These have different pair denominators, so this
is an input-feasibility finding, not a paired accuracy improvement or a recovered
camera/LiDAR calibration. Metric translation scale remains unvalidated.

| Already-spent recording | Adjacent pairs passing | All pairs retained |
| --- | ---: | ---: |
| 2011_09_26_0005 | 6 | 6 |
| 2011_09_26_0009 | 6 | 6 |
| 2011_09_26_0011 | 4 | 6 |
| 2011_09_26_0013 | 6 | 6 |
| 2011_09_26_0015 | 6 | 6 |
| 2011_09_28_0002 | 0 | 6 |

Median cheirality-supported feature count is340.5. The two0011 failures correspond
to frame173→174 and174→175:120/116 inliers but only5/4 cells per image, below the
six-cell rule. All six0002 pairs fail through count or spatial coverage. These
failures remain; thresholds were not lowered and no pair was replaced.

## Acquisition and input identity

Only72 PNG images (36 stereo neighbors around18 existing centers) were fetched,
from the six declared spent archives. Archive size/ETag, exact ranges, compressed
sizes, CRC32 and decoded sizes were pinned; If-Match and extraction checks held.
Total requested payload ranges:59,606,871 bytes (56.85MiB); decoded image payload:
59,303,247 bytes. All per-drive32MiB and aggregate128MiB range limits hold.
Metadata/HEAD and bounded transport retries are separately disclosed overhead.
No full archive, LiDAR scan, new recording or fresh holdout was downloaded.
All images decode, SHA256 values verify, and paths are retained under
/workspace/work/temporal-neighbors-20261030. Right images are prepared for future
work but not used by this monocular support audit.

The first audit aborted before producing a case: original-center grayscale differed
from archived images, as the builder applies histogram equalization. The original
[audit plan](../assets/temporal_neighbors_20261030/audit-plan.initial.json) and
[traceback](../assets/temporal_neighbors_20261030/audit.initial-failure.txt) are kept.
The [reconciliation note](temporal_neighbors_photometric_note.md) pins the builder's
existing operation. Applying equalizeHist to every raw center reproduces the
archived center pixel-for-pixel (18/18). The same operation is applied to all new
left neighbors, with derived paths/hashes retained. No matching, RANSAC or support
threshold changes occurred. This restores consistent photometric preprocessing;
it is not an outcome-selected image enhancement.

## Every short-baseline pair

| Drive | Frame pair | Cheirality inliers | Cells | Pass |
| --- | --- | ---: | --- | --- |
| 2011_09_26_0005 | 37→38 | 336 | [11, 12] | True |
| 2011_09_26_0005 | 38→39 | 394 | [11, 12] | True |
| 2011_09_26_0005 | 75→76 | 724 | [11, 11] | True |
| 2011_09_26_0005 | 76→77 | 624 | [12, 13] | True |
| 2011_09_26_0005 | 113→114 | 686 | [11, 11] | True |
| 2011_09_26_0005 | 114→115 | 608 | [11, 11] | True |
| 2011_09_26_0009 | 110→111 | 468 | [15, 15] | True |
| 2011_09_26_0009 | 111→112 | 487 | [12, 14] | True |
| 2011_09_26_0009 | 222→223 | 323 | [9, 11] | True |
| 2011_09_26_0009 | 223→224 | 338 | [9, 10] | True |
| 2011_09_26_0009 | 333→334 | 492 | [10, 11] | True |
| 2011_09_26_0009 | 334→335 | 462 | [11, 13] | True |
| 2011_09_26_0011 | 57→58 | 520 | [12, 12] | True |
| 2011_09_26_0011 | 58→59 | 440 | [12, 12] | True |
| 2011_09_26_0011 | 115→116 | 80 | [7, 7] | True |
| 2011_09_26_0011 | 116→117 | 156 | [7, 7] | True |
| 2011_09_26_0011 | 173→174 | 120 | [5, 5] | False |
| 2011_09_26_0011 | 174→175 | 116 | [4, 4] | False |
| 2011_09_26_0013 | 34→35 | 389 | [10, 10] | True |
| 2011_09_26_0013 | 35→36 | 401 | [10, 10] | True |
| 2011_09_26_0013 | 70→71 | 405 | [13, 12] | True |
| 2011_09_26_0013 | 71→72 | 412 | [11, 13] | True |
| 2011_09_26_0013 | 106→107 | 343 | [13, 13] | True |
| 2011_09_26_0013 | 107→108 | 320 | [12, 13] | True |
| 2011_09_26_0015 | 73→74 | 324 | [10, 10] | True |
| 2011_09_26_0015 | 74→75 | 290 | [11, 11] | True |
| 2011_09_26_0015 | 147→148 | 175 | [8, 8] | True |
| 2011_09_26_0015 | 148→149 | 245 | [7, 7] | True |
| 2011_09_26_0015 | 221→222 | 530 | [10, 11] | True |
| 2011_09_26_0015 | 222→223 | 456 | [10, 10] | True |
| 2011_09_28_0002 | 92→93 | 37 | [5, 5] | False |
| 2011_09_28_0002 | 93→94 | 0 | [0, 0] | False |
| 2011_09_28_0002 | 186→187 | 60 | [5, 5] | False |
| 2011_09_28_0002 | 187→188 | 25 | [4, 4] | False |
| 2011_09_28_0002 | 280→281 | 2 | [2, 2] | False |
| 2011_09_28_0002 | 281→282 | 0 | [0, 0] | False |

Twenty-seven focused tests pass (24 archive/range/extraction tests, three
image-audit/camera tests, latter rerun after preprocessing reconciliation).
Repository Ruff, strict documentation build, schemas, source/input/derived-image
hashes, byte caps,72-image completion,18-center pixel parity and36-case completion
were checked. No production/default changes or extrinsic fits occurred.

Next validate metric stereo visual odometry on these short intervals, using known
camera baseline for scale and an independently specified controlled-motion test.
It must not consume reference camera/LiDAR extrinsics, and must retain all eight
failed support pairs rather than silently evaluating only28. Depth and motion
inferred from the same stereo stream have correlated errors; temporal agreement
would be an extra measurement check, not independent ground truth. Freeze motion
estimator, failure accounting and scale/time/frame conventions before execution.
Only then decide whether a temporal surface-consistency experiment is justified.
