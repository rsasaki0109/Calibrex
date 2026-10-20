# Complete batched screw evaluation evidence

`complete-evidence.tar.gz` retains the first complete twelve-worker comparison
against `5278168adbe8d1cffc4abbcd13800d901d0139e7`. The numerical candidate is
`df4144a4d135dea3b0863847f26bc778b56fc719`.

The archive contains the baseline's 401-module and candidate's 402-module source
capsules, frozen benchmark tools, derived public GLIM geometry, all twelve
complete input/options/result/fit/timing records and logs, validation schemas,
regression test sources and the original archive producer. The development
record retains the initial smoke failure and corrected runs. The
[receipt](receipt.json) and its [schema](receipt.schema.json) bind the archive's
exact size, digest, file count, source commits and index.

The [summary](summary.json) and [schema](summary.schema.json), together with the
[development record](development.json) and [schema](development.schema.json),
retain the original measured bytes and the comparison's limits. Paired median
API ratios are 2.023x for spent synthetic joint recovery and 4.709x for spent
public geometry. Including fresh-worker startup and output, they are 1.462x
and 2.189x. One joint API pair is 1.641x; the near-threshold joint median does
not establish a universal 2x improvement.

Replay without a GPU or original recording:

```console
python tools/audit_screw_residual_archive.py docs/assets/screw_batched_evaluation/complete-evidence.tar.gz
```

The command exits 0 after validating the complete records and both 2x median
API gates. The auditor checks safe paths, schema and digest bindings and full
input/result equality without importing archived calibration code.

All paired inputs and complete results match exactly. Joint fits retain
21/23/29 iterations, policy passes and all 32 detected controls; geometry fits
retain 15 iterations. Assigned GLIM times and capture-local nearest planes
serve a processing fixture, without a physical trajectory or accuracy claim.
See the [full comparison and scope](../../benchmarks/screw_batched_evaluation.md).
This is CPU development evidence; it makes no independent accuracy, reliability
or SOTA claim.
