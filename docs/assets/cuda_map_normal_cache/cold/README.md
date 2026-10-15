# CUDA map normal cache: cold compiler cache

| Scope | Previous (s) | Lazy fused normals (s) | Ratio |
| --- | ---: | ---: | ---: |
| Complete fresh process | 18.812822 | 16.997728 | 1.107x |
| Complete map evaluation API | 13.667212 | 11.648084 | 1.173x |

All 6 processes are retained, including first initialization and
compilation. The cold cohort has 3 alternating pairs.
Cold mode starts each worker with a separate empty cache. Retained mode starts
each method's cache empty and retains it across its subsequent fresh processes.
No warmup, timing exclusions or discarded observations enter these medians.

Every original byte is preserved, including operator input paths in provenance.
The prior revision is `3340116c2b0c1a7c0cd1aa3b29468a44d4fc330f`; the current revision is
`8d7a6477343d2a727ae02c86704fe8c76f894c5c`. Both source trees, actual-import hashes, exact
launchers/schema, complete observations/results, timings and cache-file hashes
are bound by publication-index.json. Raw scans and compiler binaries are external.
The measurements use an unreserved host and are descriptive, with no portable
speed guarantee. This is the existing copied-sensor KITTI execution control,
not independent two-LiDAR calibration accuracy or bag decoding/odometry timing.

See [scope and reproduction](../../../benchmarks/cuda_map_normal_cache.md).

```bash
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_normal_cache/cold
```
