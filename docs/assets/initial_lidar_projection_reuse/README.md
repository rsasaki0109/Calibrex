# Retired initial LiDAR projection reuse experiment

Decision: rejected. Production continuous-time code was restored to source
`39867cfe33ac5f88ca8e102f363158025982b406` after measuring candidate
`fdfeeb976221bb6aa3292de1177c4e6e25521330`. The added cache did not provide
a useful whole-process improvement. The optional point-to-point CUDA adapter
is a separate implementation and workload.

The experiment reused an exact immutable initial projection for MAD and RMS
with a 16 MiB per-batch budget. It retained native reductions, sampling,
calibration controls and holdout. Large GLIM public-data cohorts contain
768,000 train and 192,000 holdout targets from copied sensor controls;
these establish execution parity rather than independent-sensor accuracy.

| Cohort | Baseline whole median | Candidate whole median | Ratio |
|---|---:|---:|---:|
| gpu-dense | 31.737829 s | 30.245593 s | 1.049x |
| cpu-dense | 52.497014 s | 55.126768 s | 0.952x |
| default-smoke | 18.750465 s | 14.331307 s | 1.308x |

GPU dense gained only 1.049x; CPU dense regressed by about 5%.
Default sampling did not use the cache: its one-pair result is a smoke
observation and does not demonstrate acceleration.

| Cohort | Pair | Baseline whole wall | Candidate whole wall |
|---|---:|---:|---:|
| gpu-dense | 1 | 44.375212 s | 40.054054 s |
| gpu-dense | 2 | 31.732940 s | 30.245593 s |
| gpu-dense | 3 | 31.737829 s | 30.147427 s |
| cpu-dense | 1 | 63.001994 s | 62.562599 s |
| cpu-dense | 2 | 52.188450 s | 50.007449 s |
| cpu-dense | 3 | 52.497014 s | 55.126768 s |
| default-smoke | 1 | 18.750465 s | 14.331307 s |

All fourteen formal trials are retained, including cold compilation and
slower observations. Three alternating pairs per dense cohort and one
default pair used fresh interpreters; compiler disk caches persisted in
separate per-method directories. Whole wall includes startup, replay,
all candidate evaluations, refinement, transfers, diagnostics, hashes and saves.
This small local sample does not establish a portable speed guarantee.

The archive contains seventeen complete API/saved result pairs: fourteen
formal workers, two corrected diagnostic workers and one completed worker
whose diagnostic counter assertion failed afterwards. Complete calibration
payloads match within each cohort after path separator normalization and
removal of strictly validated execution identity notes. YAML bytes,
histories, controls, inputs and measured source snapshots are retained.

The first diagnostic incorrectly expected one standalone CPU residual call.
The reused final projection eliminated that call, making the observed count
zero. Its original launcher, failure and completed result remain under
`failed-diagnostic`. The corrected v2 diagnostics use new output directories.
No original failed-run counter record is invented.

Both corrected diagnostics observed 81 cache hits and 36,373,267 reused rows;
28 cached summaries replaced 15,331,753 CPU summary rows. CPU normal-equation
reduction order remained unchanged. These counts did not translate to a
useful total-time gain. Diagnostic call-time sums overlap across workers
and are not critical-path percentages or formal speed measurements.

Candidate validation: 330 targeted tests, mypy on 356 source modules,
and 126 byte-identical exported schemas. This validation applies to the
retired candidate, not to later point-to-point changes.

`publication-index.json` binds exact archive bytes and retains original
measurement digests. Only local path-bearing metadata/logs were made portable;
calibration payloads and YAML bytes were preserved. The measured comparator
is loaded only by the auditor, so historical identity handling does not
require enabling the rejected cache.

Audit from repository root:

```shell
python tools/audit_initial_projection_archive.py
```
