# Fused CUDA map registration: cold compiler cache

| Scope | Previous (s) | Fused registration (s) | Ratio |
| --- | ---: | ---: | ---: |
| Complete fresh process | 19.201556 | 13.253226 | 1.449x |
| Complete map evaluation API | 12.900916 | 7.182622 | 1.796x |

All 6 processes are retained, including first initialization and
compilation. The cold cohort has 3 alternating pairs.
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
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_registration_fusion/cold
```
