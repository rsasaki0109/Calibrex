# Radar clock-profile reference indexing comparison

All six fresh-process runs are retained in their original alternating order.
The whole prepared-kinematics calibration workflow changes from a median
90.695003 to 11.618504 s (7.806083x). The solver API changes from 88.714816 to
6.589154 s (13.463765x). The host was unreserved; variation is material and
every timing is reported. No warmup, slow run or failure is omitted.

The input is all 2,762 OXTS samples from public KITTI raw drive
2011_09_30_drive_0018_sync. The 2,758 Radar velocities are generated execution
controls from those kinematics, with a declared lever arm, rotation, clock
offset and deterministic noise. They are not independently measured Radar.
This comparison establishes performance and parity, not Radar calibration
accuracy. The archive contains no locked accuracy-cohort evaluation.

Timing includes process startup, imports, JSON parsing, object construction,
all 201 offset profiles, least squares, observability, unchanged holdout and
eight signed probes, full trace capture, serialization and saving. Source
freezing, input generation, downloading and external auditing are untimed.
Both variants use the same dependencies, one BLAS thread and no measurement
budget. Every profile candidate and complete result match exactly across all
six runs. Only the separate provenance wrapper has a variant-specific commit.

`protocol.json` pins the source commits, capsules/manifests, control and actual
launcher/helper/schema bytes. Each worker records its actual imported source
hashes and normal Git identity. The only changed runtime module is
`calibrex/solvers/radar_spatiotemporal_lever_arm_solver.py`. The source capsules
omit scratch Git directories. `input-download.json` binds the original ZIP
members with SHA-256 and CRC-32; `input-fetcher.py.txt` is its exact generator.
`control.json` binds the files read to generate the control and records the
field convention and generation code identity. `index.json` binds every other
published byte, including empty stderr files and regression receipts.

The source data and derived numerical control are attributed to Andreas
Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, KITTI, and are
provided under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The [official KITTI raw page](https://www.cvlibs.net/datasets/kitti/raw_data.php)
and original archive URL are retained in the control/download provenance.
Source-code capsules retain the project's Apache-2.0 license.

Audit without a GPU, raw dataset download or calibration run:

```bash
python -m tools.benchmark_radar_reference_indexing --audit-only docs/assets/radar_reference_indexing
```

See the [benchmark report](../../benchmarks/radar_reference_indexing.md) for
all timings, scope, regression coverage and remeasurement commands.
