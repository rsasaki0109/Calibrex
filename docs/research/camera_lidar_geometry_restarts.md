# Camera–LiDAR: restart only from a better training candidate

This paired development experiment tests the search deficits identified in the
[geometry failure analysis](camera_lidar_geometry_failures.md). It compares the
existing neighbor-geometry objective against the same objective with bounded
training-only baseline restarts. It is not another objective change.

## Fixed restart rule

Start with the existing training-only baseline fit and its two nuisance-refitted
signed controls. Clip each control seed to the existing rotation bounds and
score it on training frames. Refine only seeds strictly better than the original
baseline, releasing all three rotation axes. Run at most two refinements, using
the existing search resolution. Keep the highest training score, retaining the
original baseline on ties. If the selected baseline changes, recompute both
controls once. Never iterate until a control passes.

The gate compares both seeds with the original baseline score, not with the
result of the first restart. Held-out frames are consulted only after all poses
are fixed. Reference calibration is never used as a seed or selector. This gate
also avoids spending extra search on the targeted 0002 pitch fits that showed
no better training candidate in the previous analysis.

Source `308885cd` was frozen in an isolated worktree before running all fourteen
reacquired spent inputs. The arm called `original` in the report is **unrestarted
neighbor geometry**, not the old scalar objective. Its saved axis reports must
match the preceding geometry experiment within absolute 1e-9 before the restarted
arm runs. All input hashes, folds, errors and failures are retained. Training-only
initialization, nuisance multi-starts, translation, control sizes and the 2.5 t
threshold are unchanged.

The focused 26 tests pass. They cover strict improvement, skipping ties and worse
seeds, clipping before gating, comparison against the original score for both
seeds, and the existing training/held-out separation and geometry invariants.

Reproduce from the frozen source:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python -m tools.kitti_camera_lidar_boundary_batch \
  /path/to/capture --output /path/to/new/output --geometry-restart
```

These are 42 correlated axis controls from two development drives, not independent
calibration evaluations. Full rotation norms are reference disagreements of
training-fold fits, not ground-truth accuracy. Concurrent-worker timings are not
a repeated runtime benchmark. No result alone authorizes a default change or a
SOTA claim.

## Results and decision

**The two lost detections are recovered, with no detected-control losses in this
batch. Retain experimentally; do not promote to a native default.** All fourteen
unrestarted-arm parity checks pass within absolute 1e-9, and all paired inputs
complete.

| Diagnostic | Neighbor geometry | With deficit-gated restarts |
| --- | ---: | ---: |
| Detected controls, 0005 | 19/21 | 21/21 |
| Detected controls, 0002 | 0/21 | 0/21 |
| Total detected controls | 19/42 | 21/42 |
| Worst full rotation error among detected folds | 0.296 deg | 0.296 deg |
| Worst full rotation error across all axis folds | 12.067 deg | 13.568 deg |
| Target 0002 pitch +3 control t | 1.53886 | 1.53886 |
| Target pitch +3 detected | No | No |
| Sum of profile wall times (concurrent workers) | 940.0 s | 1000.4 s |

The recovered controls are roll and yaw on 0005 pitch +3. Roll's t statistic
increases from 1.99254 to 2.63272; yaw's increases from 2.20727 to 3.79188. The
previously failing fold rotation errors decrease from 2.002 to 0.206 deg and
from 2.101 to 0.274 deg respectively. The roll result is only modestly above
the unchanged 2.5 threshold. All nineteen previously
detected controls remain detected; no additional control is detected on 0002.
The worst detected-fold rotation disagreement does not increase. This does not
prove a general false-positive rate or safety: all-fold disagreement worsens,
and the recordings are the same repeatedly used development data.

Fifty of 252 axis-fold baselines change: the two diagnosed 0005 failures and
48 folds on 0002. Thus the strict training gate is not a universal accuracy
safeguard. The difficult 0002 data still supports distant high-scoring basins,
even though they do not become detected controls in this experiment. The targeted
pitch-axis baseline and its six control pairs remain unchanged, consistent with
the previous finding that they contained no better training control seeds.

This separates the recovered search regressions from the unresolved correspondence
ambiguity. Keep the bounded restart alongside the geometry method as a development
candidate. Future work should focus on the remaining 0002 failures, not lower the
detection threshold or infer SOTA performance from 0005 reaching 21/21.

The [complete report](../assets/geometry_restart_20261028/report.json),
[summary including all changed baselines](../assets/geometry_restart_20261028/summary.json)
and [hash manifest](../assets/geometry_restart_20261028/manifest.json) retain all
fourteen pairs, full rotation vectors, control outcomes and source/input digests.
