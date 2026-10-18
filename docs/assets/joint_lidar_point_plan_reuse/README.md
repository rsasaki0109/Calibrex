# Complete guarded point-plan reuse comparison

Whole-worker median: 23.324646 -> 9.522338 seconds
(2.449466x). See [workload and bounds](../../benchmarks/joint_lidar_point_plan_reuse.md).
Each run retains 106 preparation requests; actual builds fall from 106 to two.

`protocol.json` binds both source revisions, full-point config, input hashes/counts,
exact launcher, schemas and source capsules. `attempts.json` retains every worker.
Each worker folder contains the complete result, full event/parameter trace and
array hashes, measurement and both logs. `summary.json` is recomputed by the
public auditor; `index.json` binds every published file. Source capsules and
measured files preserve their original bytes. The measured launcher equals
the public auditor.

Raw PCDs are external public inputs. The historical upstream manifest is copied
verbatim; actual header counts are pinned in the protocol. This development
capture remains degenerate with rank 9; this archive supports timing/parity and
does not establish independent calibration accuracy.

Baseline: `c517dca9b98c6bc092a184707406db4f0c56a908`.
Measured runtime: `dfd9df76515fc36b3b252e48a6c58d605b636b93`.
Protocol SHA-256: `e3491dbe5a3bf15ebffccff56db23859545d9ff27461fbb3cf3ef12fb473b7ff`.

```bash
python -m tools.benchmark_joint_point_plan_reuse --audit-only docs/assets/joint_lidar_point_plan_reuse
```
