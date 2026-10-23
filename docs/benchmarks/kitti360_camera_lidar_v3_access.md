# KITTI-360 camera-LiDAR v3: access and conversion readiness

**Current status, 2026-10-23: official access verified from local Windows;
spent-data conversion validated; no v3 pre-registration or evaluation.**
Camera-LiDAR v2 remains **refuted (8/9)**. These checks establish input
readiness, not calibration accuracy or SOTA. The dated cloud failure below is
preserved as history and does not describe current local access.

## Local recheck, 2026-10-23

The official [download page](https://www.cvlibs.net/datasets/kitti-360/download.php)
links the distribution-script ZIPs. They were inspected, never executed;
no account, interactive login or credentials were used. The scripts specify
rectified perspective archives as `data_2d_raw/SEQUENCE_image_00.zip` and
point clouds as `data_3d_raw/SEQUENCE_velodyne.zip` under the official S3 base.

Twenty archives passed metadata inventory: camera and Velodyne for the eight
sequences below, calibration, OXTS and the two timestamp archives. A separate
published-poses archive also passed: **21 archives total**. Every metadata GET
required the pinned ETag, HTTP 206, exact Content-Range and exact body length.
Only ZIP end records and central directories were read at this stage;
sensor payload bytes were **zero**, including for provisional candidates.

| Sequence | Rectified images | Point clouds | First common ID | Eligibility |
| --- | ---: | ---: | ---: | --- |
| `0000` | 11,518 | 11,518 | 0 | spent; development only |
| `0002` | 14,607 | 14,849 | 4391 | spent; development only |
| `0004` | 11,587 | 11,587 | 0 | provisional candidate |
| `0005` | 6,743 | 6,743 | 0 | provisional candidate |
| `0006` | 9,699 | 9,699 | 0 | provisional candidate |
| `0007` | 3,396 | 3,396 | 0 | provisional candidate |
| `0009` | 14,056 | 14,056 | 0 | provisional candidate |
| `0010` | 3,836 | 3,836 | 0 | provisional candidate |

`0003` remains spent and excluded. The six provisional candidates' sensor
contents have not been opened in this work. Their exposure audit is incomplete;
member counts alone do not select 60-second evaluation intervals.

The original [access report](../assets/kitti360_sota_readiness/access-report.json)
retains a failed ancillary-member assertion while extracting the perspective
script ZIP. Both selected scripts had already been extracted. The saved ZIP
passed an offline CRC/identity check; its `README.md` was ignored in the separate
[recovery receipt](../assets/kitti360_sota_readiness/distribution-script-recovery.json).
There was no redownload and the failed report was not rewritten as a pass.

### Real conversion on spent data

Source `7f652896` validated the existing published-pose converter. Source
`036c67ba7dc9eaba878d734430fbea6b8d70abbf` added and validated explicit
`--motion-source oxts`. The default remains `published_poses`.

| Motion source | Sequence | Selected frame IDs | Result |
| --- | --- | --- | --- |
| published poses | `0000` | 1, 9, 14 | PASS |
| published poses | `0002` | 4391, 4392, 4393 | PASS |
| raw OXTS | `0000` | 0, 1, 2 | PASS |
| raw OXTS | `0002` | 4391, 4392, 4393 | PASS |

Each bag was reopened and contained three images, three camera-info messages,
three point clouds, three odometry messages and one static-TF message. Selected
official members were acquired through exact pinned ranges and verified for
size and CRC. Each run had a 128 MiB uncompressed input cap; hash-verified files
were reused for OXTS checks. This stage **did read spent sensor payloads**;
fresh candidate payload bytes stayed zero. These tiny checks do not establish
full-sequence completeness, motion observability or calibration quality.

OXTS mode uses its own packet timestamps and records the earliest supplied
packet as its local Mercator origin, including its hash when outside the frame
window. It uses `kitti360_oxts_world` and does not read or hash optimized
`poses.txt`. Missing OXTS is an error, without automatic fallback. This avoids
using image/LiDAR-optimized poses as independent motion evidence; raw GNSS/INS
is still not metrology. Twists remain finite-difference approximations and point
times remain a declared azimuth model. Frame 4391 is where `0002` images and
point clouds start; published pose rows themselves start at frame 1.

```bash
calibrex convert kitti-360 DATASET_ROOT --sequence 2013_05_28_drive_0000_sync \
  --output BAG --motion-source oxts --start-frame 0 --max-frames 3
```

### Evidence and remaining work

The [manifest](../assets/kitti360_sota_readiness/manifest.json) binds unmodified
reports, schemas and exact producer/source capsules by SHA-256 and byte count.
The [range bindings](../assets/kitti360_sota_readiness/metadata-range-bindings.json)
include all 21 archive sizes/ETags, inventory hashes, code identity and each
metadata range's offsets/hash. Full inventories stay local (about 62 MB);
the public bundle is about 0.22 MB. Images, point clouds, bags and third-party
distribution scripts are not republished. Reports retain machine-local paths
as provenance, not portable download locations. The
[frame summary](../assets/kitti360_sota_readiness/paired-frame-counts.json)
uses archive member names only.

Generate the [55 spent-drive comparison](kitti_camera_lidar_v3_development.md),
confirm the selected rule through real checks, and derive gates from those
development results. Complete exposure auditing and freeze metadata-based
interval selection, timestamp-boundary checks, reference hashes and failure
policy. KITTI-360 is one dataset family and one platform; separate sequences
do not establish independent rigs. Preserve the research plan's broader
comparison requirements. Freeze code and open a separate pre-registration PR
for review before reading new evaluation sensor contents.

## Historical report, 2026-10-15

The following is the original cloud access report. Its pending/blocked status
is historical; the local results above supersede it.

**Status: official archive access still blocked; no pre-registration.** On
2026-10-15, this managed environment could not reach the official archive host:
an HTTP HEAD request for a spent development sequence returned 403. After the
initial stop, the user requested continued development. A synthetic-tested
converter is in [PR #236](https://github.com/rsasaki0109/Calibrex/pull/236),
and an opt-in far-start margin plus spent-data replay tooling is in
[PR #237](https://github.com/rsasaki0109/Calibrex/pull/237). Actual converter
validation on 0000/0002 and the 55-drive comparison remain pending access.
This is an access report, not a new audit verdict; camera-LiDAR v2 remains
refuted (8/9). No evaluation has started.

## Distribution and documented layout

The official [login page](https://www.cvlibs.net/datasets/kitti-360/user_login.php)
requires an account to download data; an existing KITTI account can be used.
No account was registered, no interactive login was attempted, and no credential
was requested. The [download page](https://www.cvlibs.net/datasets/kitti-360/download.php)
lists these aggregate sizes (the site's units, not measured byte counts):

| Component | Published total | Official download URL |
| --- | --- | --- |
| Perspective images, train/validation | 128G | [download script ZIP](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_2d_perspective.zip) |
| Raw Velodyne scans | 119G | [download script ZIP](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_3d_velodyne.zip) |
| Calibration | 3K | [calibration ZIP](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip) |
| Vehicle poses | 8.9M | [poses ZIP](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip) |
| Synchronized OXTS | 37.3M | [OXTS ZIP](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses_oxts.zip) |

The official [format documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
describes the following extracted layout, with `SEQUENCE` expanded to
`2013_05_28_drive_NNNN_sync`:

```text
calibration/{perspective.txt,calib_cam_to_velo.txt,calib_cam_to_pose.txt,...}
data_2d_raw/SEQUENCE/image_{00,01}/{data_rect,data_rgb}/FFFFFFFFFF.png
data_2d_raw/SEQUENCE/image_{00,01}/timestamps.txt
data_3d_raw/SEQUENCE/velodyne_points/data/FFFFFFFFFF.bin
data_3d_raw/SEQUENCE/velodyne_points/timestamps.txt
data_poses/SEQUENCE/{poses.txt,cam0_to_world.txt}
data_poses/SEQUENCE/oxts/{data/FFFFFFFFFF.txt,timestamps.txt,dataformat.txt}
```

This is the documented layout, not a verified archive member list. The
repository's existing `tools/download_kitti360_selected_frames.py` uses
`https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/SEQUENCE_velodyne.zip`
for per-sequence Velodyne archives, and range-reads fisheye archives. That code
is prior implementation evidence, not a successful current access test or a
perspective-camera converter. The perspective script's per-sequence URLs and
archive grouping remain unverified.

## Sequence inventory and previous exposure

| Sequence ID | Eligibility for a future camera-LiDAR evaluation | Exact archive bytes / ETag / members |
| --- | --- | --- |
| `0000` | spent; development only | not obtained in this access check |
| `0002` | spent; development only | not obtained |
| `0003` | **spent; exclude** (prior development capture, below) | not obtained |
| `0004` | candidate, subject to the remaining exposure audit | not obtained |
| `0005` | candidate, subject to the remaining exposure audit | not obtained |
| `0006` | candidate, subject to the remaining exposure audit | not obtained |
| `0007` | candidate, subject to the remaining exposure audit | not obtained |
| `0009` | candidate, subject to the remaining exposure audit | not obtained |
| `0010` | candidate, subject to the remaining exposure audit | not obtained |

The handoff had omitted `0003`. The existing
[research plan](../research/camera_lidar_sota_plan.md) explicitly records a
materialized drive `0003` development capture: 25 center frames, a 550-member
Velodyne window selection, pose motion compensation, and correspondence
methods. The paragraph was introduced by commit
`71c6098bce575c9856104debeacc865b1fd1ca97`; it is already in the base commit of
this access check. A changed camera stream does not make the recording unread
again. No raw data or poses from `0003` were read in this session, and it was
not used for new development. The remaining six candidates are not yet a frozen
evaluation set: their exposure audit and metadata inventory are unfinished.

## HTTP check and Range status

The following request, at 08:31:22 UTC (17:31:22 UTC+09:00) on 2026-10-15,
used only the spent `0000` sequence and requested headers only:

```bash
curl --head --silent --show-error --fail --max-time 30 \
  'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync_velodyne.zip'
```

```text
HTTP/1.1 403 Forbidden
content-length: 16
content-type: text/plain
date: Thu, 15 Oct 2026 08:31:22 GMT
server: envoy
connection: close
curl: (22) The requested URL returned error: 403
```

The 16 bytes describe the denial response, **not the archive size**. No ETag,
archive size, or `Accept-Ranges` was obtained. The selected executor's
`/etc/codex/network-policy.json` has a restricted allowlist that excludes both
`s3.eu-central-1.amazonaws.com` and `www.cvlibs.net`. GitHub access worked after
enabling the command's supported network permission; the same permission did
not make the archive accessible. This establishes a block in this environment,
not that KITTI-360 is unavailable globally or that S3 itself requires a login.
The environment status tool did not establish an enforced-policy readiness
state, so the actual 403 is retained as the access evidence.

HTTP Range support is **unverified**. No Range request or central-directory
fetch followed the failed HEAD. Once access is configured, first verify size
and ETag on HEAD, then use exact ZIP end-record/central-directory ranges with
`If-Match`, require HTTP 206 and the exact `Content-Range`, and list members
without opening them. Avoid block-cached readers that also fetch neighboring
image, point-cloud, or pose payloads during a metadata-only inventory. Only
after the metadata inventory can disjoint nominal 600-frame windows be selected;
the registration must also freeze how actual 60-second boundaries are checked
after timestamps become readable. No timestamps were read to select windows here.

## Exact-range metadata inventory tool

`tools/kitti360_archive_metadata.py` is ready for the access recheck. It takes
an actual official ZIP URL obtained from the distribution documentation or
download scripts; it does not guess perspective archive URLs. For the already
documented spent sequence 0000 Velodyne URL:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python tools/kitti360_archive_metadata.py \
  --url 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync_velodyne.zip' \
  --output /tmp/kitti360-0000-velodyne-metadata.json
```

The tool first requires a successful HEAD with an archive size and strong ETag.
It then requests exactly 22 bytes for the ZIP end record, the fixed ZIP64 end
records when needed, and the exact central-directory range. It never opens a
member or requests a local header or its neighboring payload. Every GET uses
`If-Match`, and checks HTTP 206, exact `Content-Range`, exact length, unchanged
ETag and identity encoding **before reading the body**. Redirects are refused;
default proxy and TLS verification are preserved. A server that ignores Range
does not cause a full-archive read.

The ZIP parser reads a bounded memory view of these retained ranges at their
original offsets. It does not create an archive-sized temporary file, including
on Windows filesystems that allocate physical space for file truncation. The
integration check on 2026-10-18 retained the ZIP64 regression above 4 GiB and
all 20 metadata tests passed; no official sensor data was read for that check.

Comments, split ZIPs, extended/displaced ZIP64 end records, inconsistent counts,
duplicate names and directories over 64 MiB stop the inventory. The tool never
broadens a tail read to work around an unsupported format. This restriction is
deliberate: a broad ZIP tail can contain sensor payloads, especially for a small
archive. Existing raw-audit tooling and frozen audit verdicts are unchanged.

Successful JSON inventories contain member names, sizes, compression, CRC-32,
local-header offsets, sequence IDs as strings, URL, archive size/ETag, actual
Range verification and provenance: code commit, generator hash, UTC timestamp
and every range's offsets/hash. No existing output is overwritten. There is no
window selection, estimator call, sensor timestamp read or pre-registration.

Synthetic regressions verify ZIP/ZIP64 listings and sparse offsets above 4 GiB,
with every requested byte constrained to metadata. They also cover ignored
Range, mismatched headers, truncated responses, unsupported end records,
directory limits/counts and output preservation. **These tests do not establish
Range support or member layouts on the official host.** All candidate inventories
and 60 s interval selection remain pending actual metadata access.

The tool itself was attempted against the spent 0000 URL on 2026-10-15. It stopped
at HEAD with `Tunnel connection failed: 403 Forbidden`; no GET followed and no
inventory file was produced. This confirms the environment block, rather than
verifying archive format or Range support.

## Exposure and provenance

- Repository base: `3906078275cb334c8bc8c2dd18513692a5fc7750` (`origin/main`),
  inspected in a separate worktree on 2026-10-15.
- Instructions read in order: `AGENTS.md`, `docs/sota_handoff.md`,
  `docs/benchmarks/sota_leaderboard.md`, then the v2 check report.
- Official HTML documentation and download-link targets were inspected through
  the browser tool. Following the five ZIP links in the distribution table
  returned unsupported-ZIP-content errors and only exposed their target URLs;
  it did not expose image, point-cloud, calibration, OXTS or pose contents. This
  is not verification of the archives or their headers.
- The shell archive access was the HEAD request above; an earlier sandboxed
  attempt could not connect to the proxy. No candidate archive was downloaded,
  listed, extracted or passed to a method in this session.
- Exposure searches inspected repository prose/code and history, including
  `git log --all -S 'drive 0003' -- docs/research/camera_lidar_sota_plan.md`.
- This step adds only the metadata inventory tool, its synthetic tests and
  documentation. Converter and estimator proposals remain in their separate
  step PRs. No threshold, pre-registration, existing verdict or protected draft
  PR was changed. No evaluation run was started.

## Next choices

1. Resume camera-LiDAR v3 after the official distribution host is available
   through the environment configuration workflow (and a legitimate download
   route is established). Repeat metadata-only checks, exclude spent `0003`,
   develop the converter only on `0000`/`0002`, compare the estimator rules on
   all 55 spent raw drives, then open the pre-registration PR and stop for human
   approval. Do not freeze a claim from these incomplete access results.
2. Alternatively, pursue the handoff's first `ins-lidar` claim: gate the judged
   roll, pitch and clock offset plus coverage. Use only spent INS-LiDAR data for
   development, audit exposure before choosing new evaluation drives, and
   disclose v2's prior camera-LiDAR reads of Velodyne/OXTS. Dataset access and
   development metrics still need verification; no supported outcome is promised.

Camera-LiDAR development has resumed on synthetic inputs only; the real-data
work remains blocked. The alternative audit has not been started. Existing
audit verdicts and the leaderboard remain unchanged.
