# Complete ordered point-plane batch comparison

Whole-worker median: 248.161254 -> 22.305351 seconds
(11.125638x). See [workload, eligibility and limitations](../../benchmarks/joint_lidar_point_batches.md).

`protocol.json` binds both source revisions, the full-point configuration,
input hashes/counts, exact launcher, schemas and source capsules.
`attempts.json` retains every alternating worker. Each worker folder contains
its complete calibration result, full event/parameter trace and array digests,
measurement and both logs. `summary.json` is recomputed by the public auditor;
`index.json` binds every published file. Source capsules and all measured files
preserve their original bytes. The measured launcher equals the public auditor.

Raw PCDs are external public inputs. The archived upstream manifest is a
verbatim historical snapshot; actual header counts are pinned in the protocol.
This development capture remains degenerate with rank 9 and is used only for
execution timing/parity. No independent accuracy claim is made.

Baseline: `f5c52fc018f298627598920db2225042fe8ce133`.
Measured runtime: `8461408bbbe90531fb3c299dc0035cd0ba20ca30`.
Protocol SHA-256: `e7cb18b88e734b920fa46d18470a8fe4f315fe2def51db037c6e591522904c24`.

```bash
python -m tools.benchmark_joint_point_batches --audit-only docs/assets/joint_lidar_point_batches
```
