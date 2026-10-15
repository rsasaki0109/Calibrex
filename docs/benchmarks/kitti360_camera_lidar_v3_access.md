# KITTI-360 camera-LiDAR v3: access check

**Status: blocked before converter development or pre-registration.** On
2026-10-15, this managed environment could not reach the official archive host:
an HTTP HEAD request for a spent development sequence returned 403. The user
requested stopping if KITTI-360 could not be acquired or converted. Steps 2--4
and the evaluation have therefore not started. This is an access report, not a
new audit verdict; camera-LiDAR v2 remains refuted (8/9).

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
- No converter, estimator, threshold, pre-registration, existing verdict or
  protected draft PR was changed. No evaluation run was started.

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

Neither alternative has been executed. Existing audit verdicts and the
leaderboard remain unchanged.
