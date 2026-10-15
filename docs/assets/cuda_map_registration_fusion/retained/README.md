# Fused CUDA map registration: retained compiler cache

| Scope | Previous (s) | Fused registration (s) | Ratio |
| --- | ---: | ---: | ---: |
| Complete fresh process | 10.048840 | 11.392684 | 0.882x |
| Complete map evaluation API | 4.248614 | 3.920983 | 1.084x |

All 8 processes are retained, including first initialization and
compilation. The retained cohort has 4 alternating pairs.
Cold mode starts each worker with a separate empty cache. Retained mode starts
each method's cache empty and retains it across its subsequent fresh processes.
No warmup, timing exclusions or discarded observations enter these medians.

Every original byte is preserved, including operator input paths in provenance.
The prior revision is `22c19ebc75ec61715c83a2e77e9c3e8eea34dffc`; the current revision is
`8fcc8d63052908e8fb06c0153a2e975acfb0dc9f`. Both source trees, actual-import hashes, exact
launchers/schema, complete observations/results, timings and cache-file hashes
are bound by publication-index.json. Raw scans and compiler binaries are external.
The measurements use an unreserved host and are descriptive, with no portable
speed guarantee. This is the existing copied-sensor KITTI execution control,
not independent two-LiDAR calibration accuracy or bag decoding/odometry timing.

See [scope and reproduction](../../../benchmarks/cuda_map_registration_fusion.md).

```bash
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_registration_fusion/retained
```
