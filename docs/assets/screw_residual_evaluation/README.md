# Sparse screw trajectory evidence

`complete-evidence.tar.gz` contains 876 indexed files in 4,319,551 bytes.
It retains both complete 401-module source trees, frozen measurement tools,
result and record schemas, all twelve complete input/result/fit/measurement
records and logs, the derived public geometry fixture, and packaging source.
The original recording is not included.

`inventory.json` binds the archive, index and original `summary.json` bytes.
The maintained baseline is `5a80e7470487ceab148f7d49b560f48c1158665e` and
the measured candidate is `38b98d03be01919abb5d7a12cab5c939001eef95`.

The median of three paired complete-API ratios is 2.366735x for synthetic
LiDAR–IMU joint recovery and 3.142851x for the public GLIM geometry fit.
Fresh-worker paired ratios are 2.203778x and 2.878685x, respectively.
Every input, fitted trajectory/calibration/covariance and full saved result
agrees exactly within each pair, excluding provenance only.

These are spent-data performance comparisons. The geometry case assigns
timestamps and fixes capture-local plane associations; it is not a physical
bag trajectory or an end-to-end LiDAR–IMU calibration. Joint seed 20261011
fails the same holdout gyro policy in both variants, and seed 20261010
retains its existing Windows `max_iterations` status. No quality threshold
or outcome is changed to obtain a performance pass.

Replay the bounded, digest-indexed archive with the current CPU-only auditor:

```bash
python tools/audit_screw_residual_archive.py docs/assets/screw_residual_evaluation/complete-evidence.tar.gz
```

It validates record/artifact schemas, source and tool bytes, imported-module
bindings, all observations and logs, complete pair equality and the
recomputed medians/gates. It does not import Calibrex, execute archived code,
or require the original bag or a GPU. Temporary extraction stays below the
selected audit directory and is removed after replay.

See the [scopes, limitations and reproduction command](../../benchmarks/screw_residual_evaluation.md).
