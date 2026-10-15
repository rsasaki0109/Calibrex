# Odometry scheduling and point placement screen

No candidate is adopted. Eight strategies retain all 492 observations in
four distinct cohorts. Their scopes and baselines differ; ratios are not pooled
or multiplied, and no whole-calibration improvement is established.

| Candidate | Ratio | Timed scope |
| --- | ---: | --- |
| Parent-linked stackless traversal | 1.000x | target_and_registration |
| Heap-path stackless traversal | 0.986x | target_and_registration |
| Normals only for hit target rows | 0.854x | target_and_registration |
| Initial-query normal prefetch | 0.827x | target_and_registration |
| Transposed contiguous bounds reductions | 0.909x | bounds_capture |
| Separate coordinate bounds reductions | 0.973x | bounds_capture |
| Pose-only point placement | 1.090x | placement |
| Pose-only placement with referenced-edge reuse | 1.423x | placement |

Traversal and normal scheduling use five rotated-order rounds over eight fixed
public GLIM registration inputs, with one untimed first-case warmup per variant.
Their timing includes tree capture, candidate setup, normal preparation, native
repairs and registration. All physical registration fields match exactly,
including transforms, decisions, iteration/correspondence counts, RMSE and
the Hessian eigenvalue; execution identities may differ. Traversal work counters
also match the original. All variant-specific counters repeat consistently.

The stackless variants preserve the original descent order and deferred-branch
decisions using a bit mask. Parent navigation or a bounded heap lookup replaces
the local stack. Recorded compiled local storage falls from 560 to 296 bytes per
thread for normal queries, and 288 to 24 bytes for fused registration. These are
kernel local-frame attributes, not total GPU-memory measurements. Complete
operation time does not improve. The experimental heap allocation is bounded
and asserts capacity on these inputs; deep/unbalanced cases are not validated.

Demand-driven normals reduce prepared target rows from 337,110 to 189,555 per
eight-case round (43.8% fewer). Extra queries, compaction and system refreshes
outweigh that reduction. Prefetching the initial correspondence set also loses.

Bounds layouts use seven rounds over the same eight targets. Only complete
bounds calculation and validation is timed; native tree construction and
topology capture are excluded. Point placement uses seven rounds over four
earlier public GLIM spline-fit captures. It times complete point placement,
comparing full derivatives with pose-only evaluation and per-control-edge log
reuse. All bound/placement outputs match reference hashes and dimensions, and
the launchers assert bit equality with coordinates present. Those coordinate
arrays are identified by NPZ hashes but not redistributed. CPU audit verifies
saved identities, not geometric recomputation. Pose-only work is not evaluated
in a complete calibration worker, and its modest component gain is not adopted.

Every pilot uses one BLAS thread on an unreserved host. First-use allocation and
imports outside the described warm scopes are not a cold-start measurement.
There is no independent calibration truth in these screens. The first traversal
launcher failed during warmup because its source transformer matched a nested
branch; it produced no timed observations and remains in local outputs.

The runtime capsule and manifest are shared by digest with the sibling
[incremental-map archive](../incremental_local_map_screen/README.md). Keep that
directory when copying this archive. The CPU-only auditor verifies the shared
capsule, all launchers/input identities/observations, exact registration fields,
output identities, work counters, sampling contracts and timing aggregates.
Experimental source is retained as text and is not a supported runtime API.

```bash
python -m tools.audit_odometry_scheduling_screen docs/assets/odometry_scheduling_screen
```
