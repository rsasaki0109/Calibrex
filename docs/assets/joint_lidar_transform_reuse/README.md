# Complete public Livox joint transform comparison

Whole-worker median: 800.253310 -> 231.253153 seconds
(3.460508x). See [workload and limitations](../../benchmarks/joint_lidar_transform_reuse.md).

`protocol.json` binds both source revisions, the full-point configuration,
input hashes/counts, launcher, schemas and source capsules. `attempts.json`
retains every alternating worker. Each worker folder contains its complete
calibration result, parameter/event trace and array digests, measurement and
both logs. `summary.json` is recomputed by the public auditor. `index.json`
binds every published file. Source capsules preserve original measured bytes.

The raw PCDs are external public inputs. The archived upstream manifest is a
verbatim historical metadata snapshot; actual header counts are pinned in the
protocol. This development capture remains degenerate with rank 9. No independent
accuracy or broader speedup claim is made.

Runtime baseline: `c0fbf5e2ddbebd589fab9e11b235e9f7ab84d5e6`.
Measured runtime: `8e0a7fe645b07bc10badc641e9902e35362c8925`.
Protocol SHA-256: `21e880b4a0080c4e8b7b583ce496a753c24dc967df3fe492e25aef531f240fa7`.

```bash
python -m tools.benchmark_joint_lidar_transform_reuse --audit-only docs/assets/joint_lidar_transform_reuse
```
