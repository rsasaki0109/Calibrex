# Joint Radar evaluation reuse comparison

The complete prepared-input workflow changes from a median 9.230905
to 4.094905 s (2.254242x), and the solver API from
7.305833 to 2.174879 s (3.359191x).
All six alternating fresh-process runs are retained. The host was unreserved;
every timing, including variation, is published in the
[benchmark report](../../benchmarks/joint_radar_evaluation_reuse.md).

The control is byte-identical to the preceding joint Radar comparison:
2,762 public KITTI OXTS samples and 2,758 generated Radar-frame velocities,
with deterministic noise and unequal weights. These are execution controls,
not independent Radar accuracy evidence. No locked accuracy cohort is used.

All complete results match exactly: five optimizer iterations, 2,099 eligible
measurements, the same 659 initially rejected IDs, complete optimizer/family
diagnostics, and fourteen signed holdout probes. The same 162,024 factor
evaluations occur in every run. Repeated interpolation falls from 171,502 to
41,799 calls; SE(3) tangent decodes fall from 162,037 to 91. The original
80-iteration bound, initial identity transform and zero clock, finite
differences, robust weighting and split are retained.

Whole-process timing includes startup/imports, parsing/construction, gating,
optimization, work counters, all diagnostics, serialization and saving.
Downloading, generation, source freezing and external audits are untimed.
Both variants use one BLAS thread and identical dependencies. The protocol
binds complete source capsules/manifests, actual Git identities and imports,
solver settings, the launcher/helpers/schema, input and original generator.
Only `calibrex/solvers/radar_joint_spatiotemporal_solver.py` changes at runtime.
`index.json` binds every other archive byte, including validation receipts.

KITTI source data and the derived numeric control are attributed to Andreas
Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, KITTI, under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The [official KITTI raw page](https://www.cvlibs.net/datasets/kitti/raw_data.php)
and original archive URL remain in provenance. Source-code capsules retain
the project's Apache-2.0 license. The original ZIP download receipt/fetcher
bind all input members by SHA-256 and CRC-32.

Audit without raw data, a GPU or calibration:

```bash
python -m tools.benchmark_radar_reference_indexing --audit-only docs/assets/joint_radar_evaluation_reuse
```
