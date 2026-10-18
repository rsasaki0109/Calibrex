# Ordered Doppler IRLS execution comparison

Generated 256-return Doppler scans for all 2758 public KITTI-derived Radar velocities; no measured Radar, raw LiDAR registration or independent accuracy claim.

Three alternating fresh-process pairs retain all 2,758 scan results and every
initial/IRLS normal system. The full worker includes imports, source identity
checks, input loading, generation of all 706,048 weighted returns, estimation,
binary history capture, compression, result serialization and saving. Source
freezing, input download and external auditing are outside the timed workers.
The API time sums complete `RadarEgoVelocitySolver.solve` calls, including
normalization, preparation, robust iterations, diagnostics and trace capture.

| Scope | Previous median s | Ordered batches median s | Ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 13.412055 | 4.976802 | 2.694915x |
| Solver API | 11.682495 | 3.150524 | 3.708111x |

The unchanged defaults retain 20 maximum iterations, 0.2 m/s Huber threshold,
1e-8 m/s convergence tolerance, rank checks and 1 m/s inlier threshold. All
17,658 IRLS iterations and final numeric fields match exactly across all six
workers. No numerical tolerance or measurement reduction is applied.

The original [public velocity control](../radar_reference_indexing/README.md)
derives 2,758 interior Radar-frame velocities from all 2,762 KITTI OXTS samples
of `2011_09_30_drive_0018_sync`. Its original JSON and provenance are retained
byte for byte in [control.json](control.json). KITTI data: CC BY-NC-SA 3.0.
The declared generator makes 256 noncoplanar directions per scan, deterministic
0.015 m/s sinusoidal noise, 16 outliers (every seventeenth return) with a 4 m/s
cosine term, and weights 1.0 through 1.4. It supplies no independent Radar truth.
The fixed geometry and moderate scan size do not establish a speedup for every
Radar, dataset, point count or calibration pipeline. Host load is unreserved;
one BLAS thread is used. Python/package versions are in the protocol.

Baseline: `36446a4fabdbfe8d406ea1908677357f4f027c0f`.
Measured runtime: `782d4cfdb314985e5991a0d816b24a137f5c75ae`.
Only `solvers/radar_ego_velocity_solver.py` changes across the 380-module trees.

- [protocol.json](protocol.json), [attempts.json](attempts.json), [summary.json](summary.json)
- Exact measured [launcher](launcher.py.txt), [schema](schema.json), source capsules/manifests
- Six worker directories, each with full `result.json.gz`, full `history.bin.gz`,
  measurements, actual full/short Git identities, import hashes and empty logs
- [Post-measurement auditor](validation/auditor.py.txt), publication code,
  regression tests and validation logs in `validation/`
- [index.json](index.json) binds every other file's raw size and SHA-256

`history.bin.gz` decompresses to sequential scan records: `B` followed by nine
little-endian doubles for the initial 3x3 normal matrix, then one `I` and fifteen
doubles per IRLS iteration (3x3 matrix, 3-vector RHS, 3-vector solved velocity).
Scan order and iteration counts come from that worker's full result. The audit
checks the record inventory, final solution bits and complete history equality.

```bash
python -m tools.benchmark_radar_doppler_batches --audit-only docs/assets/radar_doppler_ordered_batches
python -m tools.benchmark_radar_doppler_batches --baseline-commit 36446a4fabdbfe8d406ea1908677357f4f027c0f --output outputs/radar-doppler-reproduction --repetitions 3
```

The measured launcher remains unchanged. Later auditor additions bind the
publication inventory and verify the saved summary; no numerical runtime or
measurement was rerun after observing the result.
