# Ordered sparse adaptive plane search on public Livox data

The CPU acceleration replaces empty-cell enumeration in the range-adaptive
plane map with binary searches over occupied x/y/z rows. It visits exactly the
same occupied cells in the same level and lexical key order. Scalar distance,
first nearest tie, gate inclusivity, selected plane identity and downstream
normalization stay unchanged. Caller-supplied ordinary mappings retain the
original query implementation.

The workload is the existing bounded continuous-time configuration for the
[AIST Livox recording](https://zenodo.org/records/6864654), published under
CC-BY-4.0. The recording and accompanying supplied trajectory are converted with
the existing public-dataset tutorial's clock normalization. The second LiDAR
topic duplicates the first: this is an identity self-consistency control,
without an absolute two-sensor calibration or clock-accuracy claim.

## Measurement

The baseline is `240ed2cf885f9bbcffc0be68a6afac1d51b29280` (the preceding
accelerated runtime). The measured source is
`419f78cd1c0fc0789c91906115e1a4b11e0d976c`. Only
`calibrex.data.adaptive_voxel` is substituted from the baseline archive.
All other source, imports, budgets, robust filtering, finite differences,
optimization and output writers are shared.

<!-- measurement:start -->
| Metric | Baseline median (s) | Sparse median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 67.379710 | 14.587231 | 4.62× |
| Full evaluation API | 65.546480 | 12.866240 | 5.09× |
| Clock profile + extrinsic refinement | 63.387604 | 10.479196 | 6.05× |
| Adaptive nearest-plane calls | 53.207906 | 1.417437 | 37.54× |
| Worker imports | 1.413521 | 1.294041 | 1.09× |

| Pair | Order | Baseline wall (s) | Sparse wall (s) |
|---|---|---:|---:|
| 1 | baseline → sparse | 76.512152 | 17.178317 |
| 2 | sparse → baseline | 67.379710 | 16.296313 |
| 3 | baseline → sparse | 52.012553 | 12.216585 |
| 4 | sparse → baseline | 69.272399 | 12.470526 |
| 5 | baseline → sparse | 60.868472 | 14.587231 |

Baseline wall range: 52.012553–76.512152 s.
Sparse wall range: 12.216585–17.178317 s.
<!-- measurement:end -->

There are five fresh-process pairs, with order reversed every other pair and
`OMP_NUM_THREADS=1` in each child. No warmup or slow sample is excluded. The
whole-process timer includes interpreter startup, imports, bag replay, all
26 clock-candidate/refinement evaluations, extrinsic optimization, holdout
diagnostics, input hashing, result YAML and measurement saves. Parent audits
and CLI presentation are outside that timer. API and query timers are subsets
of that same run, obtained by wrapping existing calls. Split provenance hashes
the actual ordered sampled target index, capture timestamp and XYZ, and checks
the distinct train/holdout partitions in every process.

The unchanged configuration samples 2,000 source records and up to 1,200 target
points in each temporal split. It retains adaptive planes with uniform fallback,
MAD filtering, Huber loss, eight clock-profile rounds and eight fixed-extrinsic
iterations per candidate. Both methods make 62,400 adaptive queries. The final
result has 894 train and 876 holdout correspondences, rank 6, no weak directions,
selected offset -0.04 s, train RMSE 0.03199025786252043 m and holdout RMSE
0.053133620013224026 m. The offset is the unchanged profile output, not a verified
clock reference.

All ten API results and actual saved schema-valid YAML artifacts match in full,
including every candidate score, selected pose/offset, train diagnostics,
rejections, reasons and provenance. Only path separators are normalized; the
shared checkout revision stays in the result. The separate baseline/candidate
source archives bind the substituted implementation.

These are descriptive local timings on one public recording, with persistent
OS caches and unrelated machine activity. They do not multiply gains from
earlier point-cloud-pair or Camera–LiDAR workloads.

## Bottleneck and verification

A complete instrumented baseline profile made 302,785,888 calls, including
257,860,070 dictionary `get` calls. The adaptive query accounted for about
204.6 s of the instrumented 246.3 s execution. These profiler times identify
the bottleneck and are not used as speed measurements. The profile used an
earlier uncommitted measurement launcher, whose contents are reconstructed and
checked against its captured digest; `profile-provenance.json` records the
baseline import substitution, frozen inputs and working source bindings.

Fifty relevant local tests cover exact cell identity on random levels/gates,
first ties, nonzero origins, negative and large integer keys, invalid inputs,
empty maps, existing plain-mapping behavior, owned snapshots, sparse queries
without empty lookups, actual sampled split IDs, and continuous-time solver,
CLI, robust filtering, odometry and conversion regressions. Ruff and strict
mypy pass; all 125 exported schemas remain byte-identical.

## Inputs and archives

- Official ROS 1 bag: 613,350,674 bytes;
  MD5 `fb17a6d4e0bc1765e276f0dcca8a5063`;
  SHA-256 `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`.
- Official GT archive: 140,570 bytes;
  MD5 `ea25f7703ec5d076ba91dba63666d92c`;
  SHA-256 `4bfd871527d00065898153e77a6764ac331421aca16ff88bde778994f70f5bc9`.
- Livox trajectory SHA-256:
  `bcc75ff7985f2e140e514c69f029153e0127a33ee7abc1b757e077386f5059ad`.
- Converted replay: 618 messages each for points, copied points and supplied
  trajectory envelopes; its database/metadata digests are in
  `conversion-provenance.json`, alongside the exact conversion command.

`pipeline.definition.json` retains every raw trial and full canonical result;
`pipeline.benchmark.json` is its recomputed schema-valid aggregate.
`publication-index.json` binds the published archives and records original
measurement hashes and working-file source hashes. Git source archives use LF;
captured Windows working bytes can have CRLF. The publisher verifies that they
differ only in line endings. Private executable/workspace prefixes are replaced
with placeholders; numerical payloads remain unchanged.

## Reproduce

Install Calibrex and the standalone conversion adapter's optional dependencies:

```bash
python -m pip install -e ".[dev]" rosbags
```

Download the official `livox.bag` and `gt.tar.gz` from the AIST record into
`data/public/glim_versatile/`. The benchmark preparer verifies both official
checksums before running the tutorial's unchanged conversion. It refuses to
overwrite an existing replay. For a previously prepared replay, keep its
conversion lineage and skip preparation.

```bash
python tools/benchmark_adaptive_plane_search.py \
  --prepare --output-dir outputs/glim-adaptive-preparation
python tools/benchmark_adaptive_plane_search.py \
  --baseline-commit 240ed2cf885f9bbcffc0be68a6afac1d51b29280 \
  --repetitions 5 --output-dir outputs/glim-adaptive-search
```

The ordinary command uses the same acceleration automatically:

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_benchmark_config.yaml \
  --output outputs/glim-continuous-time.yaml
```

The benchmark verifies official input checksums, conversion lineage, current
source/config/input bytes, loaded module filenames, complete result equality,
saved YAML equality and sampled split identities. Keep the source checkout and
inputs fixed throughout measurement. Converted SQLite bytes can depend on the
adapter version; preparation records their actual lineage for each checkout.
