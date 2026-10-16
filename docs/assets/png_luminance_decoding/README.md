# Complete PNG decoding comparison archive

Twelve complete calibration CLI workers are retained: three alternating pairs
for the Levinson--Thrun online monitor and Pandey mutual-information calibration
on the existing authenticated A2D2 public two-frame sample. No worker or timing
warmup is omitted. Two additional untimed source-bound workers retain the exact
float64 pixel digests for all 4,638,720 real luminance samples.

The [benchmark report](../../benchmarks/png_luminance_decoding.md) defines the
scope, installation, results and limitations. `protocol.json` binds all five
external input hashes, both configs, source commits/manifests, launcher/freezer,
schema and dependency versions. Raw images and point clouds remain external.
Both variants use the same installed Pillow/NumPy and one BLAS thread. Host
activity is unreserved. The adapter requires the optional `images` extra.

Every complete result, HTML/JSON report, evidence bundle and verification is
retained with stdout/stderr, import paths and actual decoder work counts. Source
capsules contain all executed Calibrex Python sources. The capsule's commit is
injected into metadata because a source capsule has no Git directory; no solver
or mathematical operation is patched. The candidate's only runtime changes are
the PNG reader and the optional image adapter. Only run ID, timestamps, source
commit and declared output paths are normalized in full-result comparisons.

The pre-registered camera-view clouds and two-frame w=1 monitor remain
inconclusive accuracy/deployment evidence. These timings do not establish a
whole-bag or general image-reader speedup.

`validation/` retains the final regression run, executed test sources, schema
parity and a receipt binding the final live runtime to the measured candidate.
`index.json` binds every other file by SHA256, including empty stderr files.
The archive can be audited with ordinary CPU dependencies, without raw data,
Pillow or a GPU:

```bash
python -m tools.benchmark_png_luminance --audit docs/assets/png_luminance_decoding
```
