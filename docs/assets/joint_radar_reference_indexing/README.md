# Joint Radar reference indexing comparison

All six alternating fresh-process runs are retained. The complete prepared-input
calibration workflow changes from a median 49.293604 to 11.866233 s
(4.154107x). The solver API changes from 45.601982 to
9.320259 s (4.892780x). The host was unreserved;
variation is material and every timing is reported. No run is discarded.

The complete input control is byte-identical to the preceding profiled-Radar
comparison. It contains all 2,762 public KITTI OXTS reference samples and 2,758
generated Radar-frame velocities, with deterministic noise and unequal weights.
This is performance and parity evidence, not independently measured Radar or
calibration accuracy evidence. No locked accuracy cohort is evaluated.

This workload fits all seven variables from an identity transform and zero
clock offset, with the original 80-iteration bound, robust LM, initial-residual
gate, train/holdout split, diagnostics and fourteen signed holdout probes.
All six runs converge in five iterations, retain 2,099 eligible measurements
and the same 659 initially rejected IDs, and make 171,502 interpolation calls.
All numerical output, optimizer history and diagnostics match exactly.

Whole-process timing includes imports, input parsing, object construction,
initial eligibility checks, complete optimization, interpolation counting,
all diagnostics, serialization and saving. Downloading, control generation,
source freezing and external auditing are untimed. Both variants use the same
dependencies and one BLAS thread. `protocol.json` pins the initial transform,
all solver options, actual source roots/commits, complete source capsules and
manifests, launcher/helpers/schema, and input/control-generator hashes. Each
worker records its actual imported files and normal Git identity. Scratch Git
directories are not redistributed. The only changed runtime module is
`calibrex/solvers/radar_joint_spatiotemporal_solver.py`.

`control-generator.py.txt` retains the original input-generation source;
`launcher.py.txt` retains the measured workflow source. The download receipt
and fetcher bind the original ZIP members by SHA-256 and CRC-32. `index.json`
binds every other published byte, including empty stderr and validation logs.

The KITTI source data and derived numeric control are attributed to Andreas
Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, KITTI, under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The [official KITTI raw page](https://www.cvlibs.net/datasets/kitti/raw_data.php)
and original archive URL are retained in provenance. Source-code capsules
retain the project's Apache-2.0 license.

Audit without raw data, a GPU or a calibration run:

```bash
python -m tools.benchmark_radar_reference_indexing --audit-only docs/assets/joint_radar_reference_indexing
```

The [benchmark report](../../benchmarks/joint_radar_reference_indexing.md)
contains every timing and the remeasurement command.
