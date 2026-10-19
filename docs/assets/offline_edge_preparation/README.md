# Offline edge preparation evidence

`complete-evidence.tar.gz` contains 870 indexed files, including both complete
401-module source trees, frozen tools and result schemas, acquisition receipts,
all twelve full calibration results, measurements and logs, and packaging source.
The archive is 3,471,025 bytes. It contains no raw image or scan files.

`inventory.json` pins the archive and index bytes. `summary.json` is the original
formal-cohort summary, with these median ratios:

- Frame preparation: 3.880811x.
- Complete estimation API, including all preparation, fits and controls: 2.594888x.
- Complete fresh measurement worker, additionally including decoding and startup: 1.509598x.

The formal source is `73b155344a85b1caa6e659639be8e6bdf2de6abd`. All six complete
results and prepared inputs agree. The predeclared 2x estimation gate passes.
The initial cohort is retained separately, including its original CRLF-related
Git audit failure and corrected post hoc validation. No observations are pooled.

Extract the tracked archive into a new directory and replay the maintained
CPU-only auditor:

```bash
python -m tarfile -e docs/assets/offline_edge_preparation/complete-evidence.tar.gz outputs/offline-edge-replay
python tools/benchmark_offline_edge_preparation.py --audit outputs/offline-edge-replay/formal --verify-git
python tools/benchmark_offline_edge_preparation.py --audit outputs/offline-edge-replay/initial --verify-git
```

Git verification requires the recorded source commits locally. Omit
`--verify-git` for schema, byte, import, timing and complete-result replay without
Git history. Neither mode imports Calibrex or runs archived code. The timings
are spent-data performance observations without calibration accuracy or SOTA
claims. See the [measurement scope and API](../../benchmarks/offline_edge_cpu.md).
