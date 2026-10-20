# Complete IMU execution comparison

These are already-spent synthetic development records. The whole-API paired
median is 1.452x, below the declared 2x target. No independent accuracy claim
is made. See the [measurement report](../../benchmarks/screw_imu_evaluation.md).

- `complete-evidence.tar.gz`: every worker input/options/full fitted and native
  result, log and timing; original toolchain; both complete normalized source
  capsules; schemas, source regression tests and original packaging producer.
- `receipt.json`: archive size/digest, member count, index/summary digests and
  both source commits with producer provenance.
- `summary.json` and `summary.schema.json`: the measured target miss and its
  schema. No observations were discarded or rerun.

The archive's `index.json` binds every other member by size and SHA-256. The
CPU replay command in the report verifies this index, both source manifests,
the complete six-worker cohort, provenance bindings, input/options and whole
outcome equality, and reconstructs the summary. It never executes archived
source. Its exit status is 2 for these valid records because the 2x target was
missed. A new run uses a new directory and retains all attempts.
