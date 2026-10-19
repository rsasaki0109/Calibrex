# Compiled PNG decoding for camera–LiDAR calibration

The optional `images` extra uses Pillow to decode larger PNG scanlines and
NumPy to calculate the existing floating-point luminance. Install it from the
repository checkout:

```bash
python -m pip install -e ".[images]"
```

Normal `calibrex calibrate` commands use the adapter automatically for supported
images of at least 65,536 pixels. Smaller images use the native decoder. Missing
Pillow, a decoder failure, or an image exceeding Pillow's configured pixel limit
also uses the native decoder. Importing the reader or displaying CLI help does
not import Pillow. The core remains ROS independent.

## Preserved input and calibration contract

The reader still accepts only non-interlaced, 8-bit grayscale, grayscale-alpha,
RGB and RGBA PNGs. Its original header, decompression, row-length and filter
checks remain in effect. The adapter constructs a minimal PNG from the selected
header and IDAT stream, preserving the native handling of ancillary chunks,
original CRCs and trailing bytes. It leaves Pillow's global limits unchanged.

RGB luminance remains `0.299*red + 0.587*green + 0.114*blue`, evaluated as three
float64 multiplications and two additions. Alpha is ignored in the same way as
the native reader. There is no conversion to integer grayscale, compositing or
gamma adjustment. The returned immutable rows retain the original API.

The numerical solvers, train/holdout splits, known-bad probes, curvature,
tracking decisions, work budgets and output schemas are unchanged. Tests cover
all twenty combinations of color layout and PNG filter, byte wraparound,
invisible alpha, malformed streams, optional-dependency absence, decoder failure
and the original acceptance of CRC/trailing-stream variations.

Pillow is an optional external dependency under its
[MIT-CMU license](https://pillow.readthedocs.io/en/stable/about.html#license);
its codec implementation is not copied into the core.

## Complete public-data comparison

The comparison uses the two authenticated A2D2 `20180810_150607` front-left
camera/LiDAR pairs and calibration file already used by the public examples.
All five raw input hashes are retained. Both source variants use the same
installed Pillow and NumPy, one BLAS thread and alternating fresh processes.
Host CPU/GPU activity is unreserved.

The complete commands include startup, input reads and decoding, all solver
evaluations and diagnostics, output saving, HTML/JSON report generation and
bundle verification. Independent untimed workers hash every decoded float64
luminance sample from both real images. No timing warmup or failed attempt is
discarded. Only the declared output paths, run ID, timestamps and source commit
are normalized when comparing complete calibration results.

The [indexed evidence archive](../assets/png_luminance_decoding/README.md)
contains every worker, complete result/report bundle, source capsules and
manifests, input/config hashes, pixel digests and the frozen launcher/schema.
Each capsule's commit is injected into source metadata because extracted source
trees have no Git directory; this does not change a numerical operation.


This is execution and numerical-parity evidence. A2D2's camera-view point clouds
are already registered, and the two-frame online example uses `w=1`; the public
example's inconclusive accuracy and deployment status remain unchanged. These
timings do not establish gains for other datasets or other image readers.

Three alternating pairs per command (twelve timed workers) give these medians:

| Complete CLI command | Native decoder | Optional adapter | Ratio |
| --- | ---: | ---: | ---: |
| Levinson–Thrun online monitor | 19.455 s | 5.384 s | 3.614x |
| Pandey mutual-information calibration | 16.769 s | 5.185 s | 3.234x |

All twelve complete calibration results match after the declared metadata/path
normalization, and every output bundle verifies. All **4,638,720 luminance
samples** from the two images have identical float64 bytes. The candidate is
`183fb3d1221d118a20006b615d594bbf4fdc9875`; its baseline is
`5478ec42493fb9495cbabba39452c70bff8890be`. The measured environment uses Python
3.12.10, Pillow 12.3.0 and NumPy 2.3.4. These are CPU image-decoding gains; the
solver implementations and GPU kernels are unchanged.

## Reproduce and audit

```bash
python tools/download_public_dataset.py a2d2_pandey_mutual_information
python -m tools.benchmark_png_luminance \
  --baseline-commit 5478ec42493fb9495cbabba39452c70bff8890be \
  --dataset data/public/a2d2_pandey_mutual_information \
  --output-dir outputs/png-luminance-comparison --repetitions 3
python -m tools.benchmark_png_luminance --audit docs/assets/png_luminance_decoding
```

Run reproduction from the measured candidate revision recorded in the archive's
protocol. Auditing the retained evidence requires the ordinary CPU dependencies,
without Pillow, the external data or a GPU.

## Array consumers

`calibrex.data.read_png_luminance_array(path)` returns caller-owned,
C-contiguous float64 luminance with shape `(height, width)`. It shares the
existing PNG validation and optional Pillow decoder. Array consumers avoid
creating Python pixel tuples and converting them back into arrays. The
immutable `read_png_luminance` API is preserved, including its native fallback.
The online A2D2 loader and KITTI online benchmark now use the array API.

A separate comparison uses all **48 already-spent public KITTI images** from
`2011_09_30_drive_0018_sync`, with three ordered read workers per arm. Both
controls already use Pillow. All **21,773,760** decoded float64 samples match
bit for bit in every worker. Three alternating pairs of fresh processes give:

| Scope | Rows followed by array conversion | Direct array reader | Ratio |
| --- | ---: | ---: | ---: |
| Complete PNG-to-array read stage | 2.854628 s | 0.768915 s | 3.713x |
| Measurement process including startup and verification | 4.791209 s | 2.460095 s | 1.948x |

The read stage includes file reads, shared PNG validation, lazy Pillow import,
parallel decoding and the required representation conversion. Parent timing
also includes source/input verification, pixel hashing and saving evidence.
Input verification warms the operating system file cache. The host is
unreserved. These scopes exclude calibration and are reported separately from
the earlier full CLI comparison below; the ratios must not be multiplied.

The source is frozen at `87fbc7d9` and all 401 source modules are bound by a
manifest and source capsule. An earlier six-worker pilot overlapped validation
tests; its original evidence and context are retained separately and are not
pooled with the formal six workers. The [array-reader evidence](../assets/png_array_decoding/README.md)
includes both cohorts without images or scans. A public A2D2 regression compares
the complete calibration result against the rows control, normalizing only
run/source metadata and output paths; all serialized numerical values match.

```bash
python -m tools.benchmark_png_array \
  --fixture outputs/levinson-online-screen-v1/fixture-48 \
  --output-dir outputs/png-array-comparison --workers 3 --repetitions 3
python -m tools.benchmark_png_array --audit docs/assets/png_array_decoding/formal
```

Reproduction requires the existing fixture and the optional `images` extra.
The CPU-only evidence audit reads source bytes without executing the archived
launcher or source capsule and needs neither the inputs nor Pillow.
