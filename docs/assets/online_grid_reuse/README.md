# Maintained CPU grid rotation sharing

[Complete evidence](complete-evidence.tar.gz), 3,490,476 bytes, contains 871
indexed files: both 401-module source trees, one frozen timing toolchain, input
receipt, schemas, all six complete results/configurations/measurements, all 180
per-update observations, original logs, and packaging provenance. Raw images and
scans are read in place and are not included.

Archive SHA-256:
`8e604eba37e5b12e2ba0e8dc1d5ec292f75d6cdf51cc7c6b2acffa6d44ab28bb`.
The [inventory](inventory.json) binds the archive, internal index and original
[summary](summary.json), which follows [this schema](summary.schema.json).

The baseline is the maintained Numba backend at
`b7d6203fa3024a78b6693e7536329b75d2b1dc6e`; the candidate is maintained source
`c2e694e106ebdfec290538b7fd018f45c17ead7f`. Both include the typed PNG array reader.
All 48 prepared frame snapshots and all six complete quality payloads match.
Each worker executes 30 rolling updates, 21,958 objective evaluations and
395,476,000 logical point rows, including holdout, controls and curvature.

| Median of three fresh workers | Previous maintained CPU | Shared rotations | Ratio |
| --- | ---: | ---: | ---: |
| Complete solve API, including first projection JIT | 22.018838 s | 5.964614 s | 3.691578x |
| Median online update, all 30 retained | 0.681927 s | 0.126579 s | 5.387368x |
| First update, including projection JIT | 2.420839 s | 2.167978 s | 1.116635x |
| Complete worker, including startup and preparation | 32.604431 s | 16.219636 s | 2.010183x |

All workers use the same 48 already-spent KITTI image/cloud pairs, three ordered
preparation workers, one BLAS thread and no discarded warmup. The host is
unreserved. Input verification warms the filesystem cache. The predeclared 2x
gate concerns the complete solve API; startup is a separate timing scope. This
cohort is not pooled with earlier prototypes and establishes no new calibration
accuracy claim or portable latency guarantee.

```bash
python tools/audit_online_grid_reuse_archive.py \
  docs/assets/online_grid_reuse/complete-evidence.tar.gz --verify-git
```

The audit checks the indexed bytes and both committed source trees without
executing archived Python or importing optional acceleration packages. Omit
`--verify-git` for an offline archive-only audit. See the
[benchmark description](../../benchmarks/online_edge_cpu.md#rotation-sharing-in-the-maintained-cpu-grid)
for timing boundaries, bounded buffers and reproduction.
