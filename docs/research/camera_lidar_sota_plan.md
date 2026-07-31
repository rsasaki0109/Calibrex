# Camera-LiDAR SOTA research plan

This plan defines what Calibrex must implement and prove before making a
state-of-the-art claim for Camera-LiDAR extrinsic calibration. It was frozen
on 31 July 2026 after reviewing current papers and public implementations.

## Current position

Calibrex is not SOTA today. The fixed KITTI raw 0005 benchmark proved that the
native I2I solver could be made 1.719 times faster without changing its result,
but neither the scalar nor optimized path recovered any of the six prescribed
`+0.10 m` or `+10 deg` perturbations.

The first geometry-first rotation milestone is now complete on both Phase 1
datasets. The native D2D solver recovered every frozen `1, 2, 10 deg`
direction on KITTI raw 0018 and KITTI-360 drive 0000. At `20 deg`, it recovered
178/200 on KITTI raw and 184/200 on KITTI-360; both datasets have a low median
but a large p90 or p95, exposing a real direction-dependent capture limit.
The first full six-DoF cell is also complete on KITTI raw 0018. At the frozen
`(0.5 deg, 0.25 m)` perturbation, all 200 optimizations completed but only
84/200 met the strict `<0.5 deg / <0.20 m` hit definition. The zero
computational failures and 42% hit rate isolate an accuracy/capture limitation
rather than an execution failure. The remaining seven six-DoF cells,
independent datasets, and external learned baselines remain open. These are
therefore reproduction and falsification milestones, not a SOTA claim.

Phase 3 and Phase 4 now have schema-valid native refinement, paired ablation,
reference-error, trajectory, and audit boundaries, but their current tests are
synthetic. No U1--U5 or asynchronous real-sequence accuracy result has been
promoted into this current-position summary.

## Claim ladder

“SOTA” is not one leaderboard. Calibrex will keep the following claims
separate:

1. **Training-free targetless SOTA candidate** — no calibration ground-truth
   training; depth foundation models may be frozen external feature providers.
2. **Learned targetless SOTA candidate** — supervised calibration training is
   allowed, but train/test rigs, dates, and scenes must be declared.
3. **Spatiotemporal targetless SOTA candidate** — estimates extrinsics and time
   offsets from asynchronous sequences.
4. **Production target-based reference** — a separate category used for
   metrology and runtime comparison, never mixed into a targetless ranking.

The first credible Calibrex claim should be category 1. Category 2 is the
accuracy ceiling, category 3 is the long-term robotics result, and category 4
provides independent reference evidence.

## Evidence from current methods

| Method | Relevant result | Role in Calibrex |
| --- | --- | --- |
| Pandey-style I2I | Current Calibrex result: 0/6 recovery from large perturbations | Legacy native baseline |
| Borer et al. D2D MI | At `10 deg`, rotation-only hit rate is 96.5% on KITTI and 100% on KITTI-360; I2I is 0% and 0.5% | First native research target |
| Koide et al. direct registration | Multi-data error down to 0.010 m / 0.132 deg for Ouster+pinhole; 74–249 s for 15 pairs | MIT-licensed external classical baseline |
| UniCalib, WACV 2026 | KITTI Odometry: 0.550 cm / 0.044 deg at `5 deg, 0.10 m`; KITTI Raw: 2.368 cm / 0.084 deg at `10 deg, 0.25 m`; about 11 Hz | Current learned accuracy target and MIT external baseline |
| CLRNet, 2026 preprint | View-of-Delft rigid four-frame Camera-LiDAR: 0.3 cm / 0.03 deg after correcting up to `1 m, 20 deg`; full-range experiment: 0.2 cm / 0.04 deg | Current multi-frame learned watchlist; adapter only when code and usable terms are released |
| TLC-Calib, RA-L 2026 | Neural Gaussian scene representation; reports gains on KITTI-360, Waymo, and Fast-LIVO2; official code and processed KITTI-360/Fast-LIVO2 data released May 2026 under a non-commercial research license | External-process research baseline only; never import its restricted code into the core |
| Continuous-time multi-LiDAR/camera/IMU | Reports about 0.04 deg / 0.5 cm with common FoV and estimates time offset | Long-term spatiotemporal architecture |
| iKalibr, T-RO 2025 | Public targetless multi-sensor spatiotemporal framework | External system-level baseline |
| FAST-Calib | Target-based, below 6.5 mm registration residual and below 0.7 s | Separate production/metrology reference |

Primary sources:

- Borer et al., <https://arxiv.org/abs/2311.01905>
- Koide et al., <https://arxiv.org/abs/2302.05094>
- UniCalib, <https://openaccess.thecvf.com/content/WACV2026/html/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.html>
- UniCalib implementation, <https://github.com/han-15/UniCalib>
- CLRNet, <https://arxiv.org/abs/2603.15767>
- TLC-Calib, <https://arxiv.org/abs/2504.04597>
- TLC-Calib implementation, <https://github.com/SNU-VGILab/TLC-Calib>
- Continuous-time targetless calibration, <https://arxiv.org/abs/2501.02821>
- iKalibr, <https://github.com/Unsigned-Long/iKalibr>
- FAST-Calib, <https://arxiv.org/abs/2507.17210>

## Frozen benchmark suite

SOTA development must use two protocol families: exact paper reproduction and
an independent Calibrex falsification suite. Results from one family must not
be substituted for the other.

### A. Analytical D2D reproduction

Reproduce Borer et al. before extending it:

- KITTI raw `2011_09_30_drive_0018`, camera `image_02`.
- KITTI-360 `2013_05_28_drive_0000`, camera `image_03`.
- Exactly 25 frames sampled uniformly over each sequence.
- Rotation-only perturbations at `1, 2, 10, 20 deg`.
- Two hundred deterministic Fibonacci-sphere directions per error level.
- Six-DoF perturbations at `(0.5 deg, 25 cm)`, `(1 deg, 25 cm)`,
  `(0.5 deg, 50 cm)`, and `(1 deg, 50 cm)`.
- A hit requires rotation error below `0.5 deg`; for six DoF it additionally
  requires translation error below `20 cm`.

The paper's published hit rates are reproduction targets, not values to copy
into a Calibrex result.

### Phase 1 KITTI reproduction evidence (31 July 2026)

The first frozen KITTI `10 deg` gate has been executed end to end:

| Field | Observed value |
| --- | --- |
| Sequence / camera | `2011_09_30_drive_0018_sync` / `image_02` |
| Frame policy | 25 endpoint-inclusive uniform frames; nearest-half-up integer rounding |
| Feature provider | Official FeatDepth KITTI-raw checkpoint, raw inverse-depth/disparity output |
| Provider source | commit `550420b3fb51a027549716b74c6fbce41651d3a5`, MIT |
| Checkpoint | 661,073,985 bytes; SHA-256 `559f168c762d01af7c745b32ced8930a12423a0e9b62af791f7d894bd902af27` |
| Perturbations | 200 deterministic Fibonacci-sphere directions at `10 deg` |
| Hit definition | strict rotation error `< 0.5 deg`; fixed translation error `< 0.20 m` |
| Result | 200/200 hits, 0 failures, hit rate `1.000` |
| Rotation error | mean `0.2366 deg`; median `0.2337 deg`; maximum `0.4384 deg` |
| Runtime | 2,596.86 s wall time with four workers; 904,076 KiB measured process peak |
| Paper target | `0.965` hit rate |

The schema-valid benchmark digest is
`03824378e8c36e28a2ecc7d63141dbe5bc704a60f756a5ce5b333e22203c08ea`;
the frozen protocol digest is
`a41eb9d532f7c10ea5c8cdfc3b444e7ebd0526ed389dae3fb56adf4eb6845ec4`;
and the Bull's Eye SVG digest is
`e36707ce1524a4cb3825816a4e34cb14d80ac9c9b0005a457aed626c056a687d`.
All 200 optimizer traces were retained.

The remaining frozen KITTI raw rotation levels were then executed without
changing the solver or strict hit rule:

| Initial rotation | Hits | Mean | Median | p90 | p95 | Maximum | Wall / peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1 deg` | 200/200 | `0.2439 deg` | `0.2404 deg` | `0.3156 deg` | `0.3530 deg` | `0.4906 deg` | `1:08:29` / `979,436 KiB` |
| `2 deg` | 200/200 | `0.2434 deg` | `0.2358 deg` | `0.3205 deg` | `0.3533 deg` | `0.4651 deg` | `30:09.92` / `930,104 KiB` |
| `10 deg` | 200/200 | `0.2366 deg` | `0.2337 deg` | — | — | `0.4384 deg` | `43:16.86` / `904,076 KiB` |
| `20 deg` | 178/200 | `3.1941 deg` | `0.2425 deg` | `17.0067 deg` | `28.4920 deg` | `36.8213 deg` | `41:37.67` / `945,608 KiB` |

All 600 newly executed traces validate and match their problem and protocol
digests. All 200 `20 deg` solver executions completed; “178/200” is the strict
recovery count, not a computational-failure count. Its low median but large
tail shows that reporting only central tendency would hide the capture-basin
failures.

| Rotation | Protocol SHA-256 | Benchmark SHA-256 | Definition / Bull's Eye sidecar / SVG SHA-256 |
| ---: | --- | --- | --- |
| `1 deg` | `f4546eb35e97fdc5a040c0f215a90f9099be72bd419c8ca4986bab431faad866` | `102480b7139a5dee41bd46182f5bcc1a0bd91bfc1c0cd25cc5b9a4cc592e51b9` | `a8c18bf3753338d8ded71ba3a268ea81cc43a6623050aeef2def88408139df38` / `b2fb64ef01ef003bd9d6809893001b266fc83abed1854da62eab6ab6bc8da559` / `d1d53ef2a6566d0bd438d4db807c930a4ec50b86a45a81428cfc5bcca79e6400` |
| `2 deg` | `53615a0b1d7756beaf6e68663080552718e32e2efd94643c557fb5e6b19c1881` | `eec11698117a41d494ef38ea347cc1620c62c7ec5f06be17e70322bf08ea99fc` | `84c6ea3e77b183f68b33723efe2953514aac15fc4eb1e9bc0c355b3e27c48a86` / `e44ede39e48bd3b941b6ed1e6001008109f2022843fb4bc61929e968bdee2771` / `ff97012c6f21b201b471b034bfeacb00e3e8ab733d6e608e622f7c5da8d67ee2` |
| `20 deg` | `67f182c2c6154199ade83872bf40abada8016019a5c36f94335309c4c278a65a` | `22833d64e1c99747dfe05fa40cc7c03960a0ab7c48ee9ce2e44748cb7a47f882` | `1a70621b9e69873a9255271acf3ca98365cc6e067e58eff168d5820323b6cf34` / `25d1d6e5d4b6252e29ba74fe22c0e0c6c42b254de94392b4f4fcaac069173778` / `30c2cd35487ed0bc3387c8725722c18be083f4ed5d4f4e2b5cfecb3d7ac5b7eb` |

The shared problem SHA-256 is
`1fddd3e015e3a8b5a1356405a9296b247f6c2d80a040ea3fa5e7323572a14041`.

Important comparability limits remain explicit:

- The paper does not publish its exact 25 frame IDs, code, checkpoint digest,
  histogram-bin count, or BOBYQA settings. Calibrex freezes its own uniform
  rounding policy, 32 bins, and bounded SO(3) pattern search before execution.
- The official FeatDepth repository says that the selected checkpoint was
  trained on KITTI raw, but does not publish enough training IDs to rule out
  overlap with drive 0018. The provider artifact records overlap as
  `possible`. The test-split online-refined checkpoint was not used.
- Raw FeatDepth inverse depth/disparity is the histogram feature. Converting it
  nonlinearly to nominal metric depth materially changes fixed-width
  histograms and is not the frozen reproduction path.
- This result uses dataset-provided extrinsics rather than independent
  metrology. It does not satisfy the KITTI-360 or multi-dataset final gates.

The external provider remains outside the ROS-independent core:

```bash
PYTHONPATH=src /path/to/featdepth-env/bin/python \
  tools/run_featdepth_provider.py /path/to/2011_09_30_drive_0018_sync \
  --featdepth-repository /path/to/FeatDepth \
  --checkpoint /path/to/fm_depth.pth \
  --output-directory /path/to/depth \
  --artifact-output /path/to/featdepth-provider.json

calibrex camera-lidar build-kitti-problem \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --output /path/to/problem.yaml
calibrex camera-lidar freeze-rotation-protocol /path/to/problem.yaml \
  --output /path/to/protocol.yaml --perturbation-count 200 --rotation-deg 10
calibrex camera-lidar benchmark-rotation \
  /path/to/problem.yaml /path/to/protocol.yaml \
  --trace-dir /path/to/traces \
  --definition-output /path/to/definition.yaml \
  --output /path/to/benchmark.yaml \
  --workers 4 --resume --required-hit-rate 0.965
```

### KITTI-360 reproduction input lock

The second Phase 1 path uses the official KITTI-360
`2013_05_28_drive_0000_sync` sequence and right fisheye `image_03`.
The 25 endpoint-inclusive uniform frame IDs span `0000000000` through
`0000011517`. Its frozen inputs are:

| Field | Locked value |
| --- | --- |
| MiDaS model | `dpt_beit_large_512`, official MiDaS tag `v3_1` |
| MiDaS source | commit `1645b7e1675301fdfac03640738fe5a6531e17d6`, MIT |
| Checkpoint | 1,581,966,003 bytes; SHA-256 `9e9e900747e9e8b3112df716979219836a27716277b3d0dc53889cbba8b82328` |
| Image processing | Official aspect-preserving minimal resize, multiple of 32, bicubic return to the 1400×1400 source grid |
| Camera model | Official KITTI-360 MEI parameters, including radial-tangential coefficients |
| Motion compensation | Official azimuth-scaled scan uncurl model from `kitti360Scripts` commit `32f6d64eef27c32b52c542b4e95be8af9c9c6444` |
| Pose coverage | Ego motion applied to 21 selected scans; four official identity fallbacks where an adjacent pose is unavailable |
| Depth feature | Raw MiDaS relative inverse depth; nonpositive bicubic overshoot is rejected, with no metric conversion or online refinement |

The official KITTI-360 projection retains only the positive signed camera
range. Accepting the rear optical hemisphere is a severe error for the
side-facing fisheye: in the one-direction smoke test it biased the recovered
rotation to `2.004 deg`. Enforcing the official forward-hemisphere rule reduced
the same result to `0.250 deg`, below the paper's strict `0.5 deg` hit
threshold.

Comparability remains bounded. Borer et al. report a double-sphere fit but do
not publish its parameters. KITTI-360 publishes an MEI calibration instead.
Calibrex therefore uses the exact public MEI model and labels this as a
reproduction with a declared camera-model substitution, not a bit-exact
re-execution of unpublished intrinsics. The provider also declares possible
training/evaluation overlap because the MiDaS training mixture includes
KITTI-family data. The disposable Python 3.12 environment applies a recorded
`default_factory` compatibility-only patch to MiDaS's pinned `timm==0.6.12`;
model operations and weights are unchanged.

The external provider and motion adapter remain outside the ROS-independent
core:

```bash
PYTHONPATH=src /path/to/midas-env/bin/python \
  tools/run_midas_kitti360_provider.py /path/to/2013_05_28_drive_0000_sync \
  --calibration-root /path/to/calibration \
  --midas-repository /path/to/MiDaS \
  --checkpoint /path/to/dpt_beit_large_512.pt \
  --output-directory /path/to/midas-depth \
  --artifact-output /path/to/midas-provider.yaml

PYTHONPATH=src python tools/motion_compensate_kitti360.py \
  /path/to/2013_05_28_drive_0000_sync /path/to/midas-provider.yaml \
  --calibration-root /path/to/calibration \
  --poses /path/to/poses.txt \
  --output-directory /path/to/compensated-scans \
  --manifest-output /path/to/compensated-scans/manifest.yaml

calibrex camera-lidar build-kitti360-problem \
  /path/to/2013_05_28_drive_0000_sync /path/to/midas-provider.yaml \
  --calibration-root /path/to/calibration \
  --lidar-directory /path/to/compensated-scans \
  --lidar-manifest /path/to/compensated-scans/manifest.yaml \
  --output /path/to/problem.yaml

calibrex camera-lidar freeze-rotation-protocol /path/to/problem.yaml \
  --output /path/to/protocol.yaml --perturbation-count 200 --rotation-deg 10
calibrex camera-lidar benchmark-rotation \
  /path/to/problem.yaml /path/to/protocol.yaml \
  --trace-dir /path/to/traces \
  --definition-output /path/to/definition.yaml \
  --output /path/to/benchmark.yaml \
  --workers 2 --resume --required-hit-rate 1.0
```

### Phase 1 KITTI-360 reproduction evidence (31 July 2026)

The frozen KITTI-360 `10 deg` gate completed without threshold or code changes
during execution:

| Field | Observed value |
| --- | --- |
| Sequence / camera | `2013_05_28_drive_0000_sync` / `image_03` |
| Frame policy | 25 endpoint-inclusive uniform frames |
| Perturbations | 200 deterministic Fibonacci-sphere directions at `10 deg` |
| Hit definition | strict rotation error `< 0.5 deg`; fixed translation error `< 0.20 m` |
| Result | 200/200 hits, 0 failures, hit rate `1.000` |
| Rotation error | mean `0.3812 deg`; median `0.3924 deg`; p90 `0.4437 deg`; p95 `0.4538 deg`; maximum `0.4676 deg` |
| Per-trial runtime | mean `131.45 s`; median `121.27 s`; maximum `294.23 s` |
| Full run | `3:16:24` wall time with two workers; `2,025,128 KiB` measured process peak |
| Paper target | `1.000` hit rate |

All 200 trace artifacts validate, pin the same problem/protocol digests, and
retain the original hit decision. Frozen artifact digests are:

| Artifact | SHA-256 |
| --- | --- |
| Problem | `6bf2cc157de930f58e891457564dbd522ecd5b30e037ef22044a9cbc3e8390de` |
| Protocol | `916b63970c16e9d170aa6deb68650ae6ac8f584a448253b29a85e738c8e2917f` |
| Raw benchmark definition | `eb6e44af8980cdcaa3903268670d56e10f8eed51f18692acb8da4eca57070439` |
| Aggregated benchmark | `80c099a476ba6da133af18568201f4d59b3bddf17f45168dc128620a423af238` |
| Bull's Eye sidecar | `e1a3424a79ccd994054b55ea07f348746dbc993d84c452b92a50a86bbd6cf2fe` |
| Bull's Eye SVG | `49207ba6d5ffdf44ce95070b38e8936371cac2004d6cd80dfc61cc698c4a9977` |

The remaining frozen KITTI-360 rotation levels were then run without changing
the solver or strict hit rule:

| Initial rotation | Hits | Mean | Median | p90 | p95 | Maximum | Wall / peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1 deg` | 200/200 | `0.3768 deg` | `0.3872 deg` | `0.4400 deg` | `0.4526 deg` | `0.4665 deg` | `1:37:56` / `2,023,108 KiB` |
| `2 deg` | 200/200 | `0.3834 deg` | `0.3920 deg` | `0.4427 deg` | `0.4473 deg` | `0.4751 deg` | `2:07:35` / `2,043,124 KiB` |
| `10 deg` | 200/200 | `0.3812 deg` | `0.3924 deg` | `0.4437 deg` | `0.4538 deg` | `0.4676 deg` | `3:16:24` / `2,025,128 KiB` |
| `20 deg` | 184/200 | `2.9900 deg` | `0.3882 deg` | `0.4533 deg` | `28.4409 deg` | `45.4505 deg` | `2:40:34` / `2,082,896 KiB` |

All 600 newly executed traces validate and match their problem and protocol
digests. All 200 `20 deg` solver executions completed; its 16 misses are
capture failures rather than computational failures. As on KITTI raw, the
median alone would conceal a large failure tail.

| Rotation | Protocol SHA-256 | Benchmark SHA-256 | Definition / Bull's Eye sidecar / SVG SHA-256 |
| ---: | --- | --- | --- |
| `1 deg` | `c5368b28e42026a9b068d5250cce9f986333aef8159a4569bf8fd114056af3ad` | `9a111dcb28f0077939f36603cbb6da6da2626f44b80675cf8397332235f40470` | `91b8aba354722770515e7bfd714597218f460acdda5f251fceff0cf0c2a8d25e` / `fec05af81ed338cab0303672863d9a7145f3ee9c00fa72f8275179dab0c45661` / `ca55d2480ab991e889cdefa01c982fe77609498dfae486719d0405220807f394` |
| `2 deg` | `ea4f74b1988ec12960369ad08ef509cb137cd159fd21d7bf45674bb5cad00645` | `2a07146b99f889ee1a24e73336980a6a1206cbe73c58e6e83c295db30165bc6c` | `06df5fdde202883c58ef766810ce88e36add24d245039dbf27113e3e39a1d9db` / `da0c1c006ff3c7709767bc2f750796524e16715aa40fa9bcfe3f692140818add` / `08555135487a59f426d57bd43dc82ab86c32edb3ef3da77d0cd0f4b7df460d7c` |
| `20 deg` | `7c9cb75c8a46759c1e587a3a9b88a14b78baf2ec087e408f7b77b68cd1ea0f1f` | `e79d14b18b856e4ffb6a96bb70bc53820bcedaa17d0947d03af9c18695f5d726` | `bcc317265636a355770d2a967656d9c00f136f6ddc4a9ee996a0b3eaaa05baf8` / `ec582e24352ce6e6f52004345e8ad2e1c14fabb7699f2a4ee98729205cd73eca` / `bb4aaa4cbf5f01df46a4149d9f007ae2073b638824094679f24c2778972cd7df` |

This clears the complete prespecified Phase 1 rotation-only matrix on both
datasets. It does not clear the six-DoF, independent-rig, or learned accuracy
gates.

### A2D2 cross-family execution smoke

The small official A2D2 public pair was added as a deliberately limited third
dataset-family execution check. Frames 60 and 61 from
`20180810_150607` use the official undistorted front-left images, calibration,
and view-filtered LiDAR NPZ members. The official tutorial states that the
points are already mapped into the camera view, so Calibrex back-projects the
distributed pixel row/column/depth into a z-forward camera frame and declares
identity only as a pre-registration reference. This cannot provide independent
extrinsic accuracy.

The pinned MiDaS v3.1 provider and five-direction `10 deg` smoke produced:

| Field | Value |
|---|---:|
| Dataset family | A2D2 |
| Frames | 2 |
| Perturbations | 5 deterministic Fibonacci directions at `10 deg` |
| Completed solver runs | 5/5 |
| Hits (`rotation error < 0.5 deg`) | 0/5 |
| Final rotation errors | `1.016, 1.102, 24.766, 1.143, 1.183 deg` |
| Wall time / peak RSS | `40.68 s / 180388 KiB` |

The outcome is a cross-family **FAIL**, not a threshold-tuning opportunity.
Two frames do not represent the 25-frame Borer protocol, the source point
cloud is pre-registered, and one direction entered a grossly wrong optimum.
The artifacts remain useful for detecting portability and objective-topology
regressions:

| Artifact | SHA-256 |
|---|---|
| MiDaS provider | `a18ad94ba17c1ede7cac668ce41609caa0a023563b64cf3c287e55686d26b55b` |
| Generated LiDAR manifest | `e0a79a049f06543064ce8f64f9a4bd9677ecfba11d3ef5f74e5df7c2483941b0` |
| Problem | `cdb7b1864235149a2859537f829ae0454d1e300abc3271d30d028746862ce447` |
| Protocol | `7f525397a88a6ab0a6514c20bf86f0f6e6dea7ba5a2c50ee12a8ab90da199929` |
| Benchmark | `ba4a5b65b23990ff129b442c87434b7e3f36818fe4b10792e5b266fb40465396` |
| Bull's Eye sidecar | `267b6791662403578d4d01e81b61b19d20d7353dfa5d59c08b355f4375dc1554` |
| Bull's Eye SVG | `57323bdafaf451efc342e01022715c660e3299f85e30a3465d4b983c863da77a` |

### B. Learned-method reproduction

Reproduce the five UniCalib configurations without changing their splits:

| Protocol | Training | Test | Perturbation |
| --- | --- | --- | --- |
| U1 | KITTI Odometry 01–21 | 00 | `5 deg, 0.10 m` |
| U2 | KITTI Odometry 01–21 | 00 | `10 deg, 0.25 m` |
| U3 | KITTI Raw 2011-09-26 | 2011-09-30 | `10 deg, 0.25 m` |
| U4 | None beyond U3 model | KITTI-360 Test-SLAM | `10 deg, 0.25 m` |
| U5 | None beyond U3 model | Waymo testing subset | `10 deg, 0.25 m` |

Every learned run must lock the training sample inventory, checkpoint digest,
depth-model name/version/weight digest, code commit, CUDA stack, seed, and
train-data isolation declaration.

When code and weights become available, add the CLRNet View-of-Delft protocol
as a distinct benchmark family. Preserve its single-frame, iterative
large-error, four-frame rigid, and full-range settings separately. In
particular, do not compare its sequence-level median result directly with a
single-frame mean from another paper.

### C. Independent Calibrex falsification

The independent suite prevents benchmark-specific overfitting:

- Keep the existing digest-locked KITTI raw 0005 test as a regression shard.
- Add at least three dates or rigs from KITTI Raw, KITTI-360, Waymo, and A2D2.
- Split by sequence/date/rig, never by neighboring frames alone.
- Prespecify signed SE(3) perturbations at:
  - rotation: `1, 2, 5, 10, 20 deg`;
  - translation: `0.05, 0.10, 0.25, 0.50 m`;
  - coupled six-DoF errors using deterministic sphere/Sobol directions.
- Add scene strata for static/dynamic, day/night, texture, weather, FoV, LiDAR
  scan pattern, and projected-point support.
- Retain failures in every denominator.
- Freeze the suite before comparing new model variants.

### D. Absolute and downstream evidence

Public dataset calibration is not absolute metrology. A final SOTA submission
also needs:

- one target or surveyed-rig dataset whose reference uncertainty is declared;
- repeated remount/capture sessions;
- a downstream held-out metric such as depth-edge alignment or point-to-image
  reprojection that was not used by the solver;
- sensitivity to camera intrinsics, time offset, rolling shutter, dynamic
  objects, and occlusion.

FAST-Calib or another target-based method may generate an independent
reference, but it remains in a separate result category.

## Metrics and statistical gates

Every method reports:

- SE(3) translation distance and quaternion geodesic rotation error;
- per-axis errors only as secondary diagnostics;
- hit rate and area under the success-versus-perturbation curve;
- median, mean, p90, p95, maximum, and failure rate;
- runtime, peak memory, preprocessing time, and hardware;
- held-out objective and downstream projection metrics;
- per-stratum metrics and worst-stratum result;
- deterministic paired bootstrap confidence intervals;
- empirical uncertainty coverage when the method predicts uncertainty.

The following rules apply:

- Test data cannot select checkpoints, thresholds, depth providers, or
  optimizer parameters.
- A mean-only win is insufficient.
- A failed baseline reproduction cannot support a SOTA claim.
- A method must improve or match every primary safety metric; trading a lower
  mean for a higher catastrophic-failure rate is a rejection.
- Report dataset-reference error and independent downstream evidence
  separately.

## Architecture

### Core contracts

Add versioned, ROS-independent contracts:

- `DepthMapObservation`: metric/relative depth, validity, uncertainty,
  intrinsics, image transform, capture time, and source frame.
- `DepthProviderArtifact`: provider/model/version/checkpoint/license/input and
  output digests, scale convention, preprocessing, command, and environment.
- `CameraLidarCalibrationProblem`: observations, initial transform, bounds,
  frame/time conventions, and split IDs.
- `CalibrationCandidateTrace`: every seed, objective evaluation, accepted
  update, stopping reason, and output transform.
- `CameraLidarBenchmarkProtocol`: perturbation generator, exact samples,
  thresholds, train/validation/test isolation, and metric definitions.

All public APIs require type hints. Configs, traces, and results remain
schema-valid. Generated results include source, input, config, model, and
output provenance.

### External boundaries

- UniCalib and Koide enter through external adapters with pinned containers or
  environments.
- CLRNet remains a watchlist method until its promised code, weights, and
  non-commercial-use terms can be inspected. If usable, it stays external.
- iKalibr enters as an external spatiotemporal baseline.
- GPL implementations such as MLCC, Livox Camera Calib, and FAST-Calib must
  remain subprocess/container baselines; no GPL code enters `src/calibrex`.
- Learned depth stays an optional provider. The native D2D solver consumes a
  depth artifact and does not import PyTorch, CUDA, ROS, or model code.

## Research phases and go/no-go gates

### Phase 0 — Benchmark and baseline lock (weeks 1–2)

Deliver:

- generic schema-versioned external calibration run artifact;
- dataset locks and protocol files for D2D A/B and UniCalib U1–U5;
- Koide and UniCalib adapters;
- immutable perturbation manifests and paired aggregation;
- baseline reproduction report with failures and provenance.

Gate:

- All artifacts validate and rerun from digests.
- Published baseline results are reproduced within a prespecified tolerance,
  or discrepancies are explained and the affected comparison is marked
  non-comparable.

### Phase 1 — Native rotation-only D2D MI (weeks 3–5)

Deliver:

- solver-neutral depth-map contract;
- frozen MoGe-2/Depth Anything/FeatDepth provider artifacts;
- LiDAR depth projection with z-buffer visibility;
- per-frame D2D MI averaged across frames;
- bounded derivative-free SO(3) optimization;
- exact Borer perturbation protocol and Bull's Eye plots.

Gate:

- At `10 deg`, achieve at least `96.5%` KITTI and `100%` KITTI-360 hit rate,
  with no test-set tuning.
- If hit rate is below `90%` on either dataset, stop feature expansion and
  diagnose depth scale, visibility, FoV, and objective topology.

### Phase 2 — Robust six-DoF analytical solver (weeks 6–8)

Deliver:

- deterministic global rotation seeds and bounded translation search;
- multi-resolution depth, robust histogram/MI estimation, occlusion masks, and
  dynamic-region downweighting;
- local SE(3) refinement and training-only endpoint safety gate;
- capture-range AUC and failure diagnosis.

Implementation checkpoint:

- The first frozen real KITTI-360 six-DoF smoke completed from a paired
  `(0.5 deg, 0.5 m)` perturbation. The bounded native solver converged in
  `165.90 s` and returned `0.2505 deg / 2.08e-17 m`, satisfying the strict
  `<0.5 deg / <0.20 m` hit definition.
- The one-trial benchmark, definition, and full candidate trace are
  schema-valid. Their SHA-256 values are respectively
  `665fd6aa17ff6933e94cfb960358aa0cdc2ebaf5f2b223d629cb70b0066264f0`,
  `8fa5ee01b70a6236637114d35626312da2af3d7453b58f2d1f5c1725a42a455a`,
  and
  `ef090ae8a642675a548096cafb606ebc747635e733e3bbf65009eaa6c617b78c`.
  The frozen problem/protocol digests are
  `f08ea1d8c98966db678216deece5d25fb3740d60708f6f0a5e56a751aa33cc31`
  and
  `c62e20a304245e4ac63d1e064b2eae8106540358d46f22d9c829c8cb587ed2c4`.
- This establishes executable real-data six-DoF recovery, not the Phase 2
  hit-rate gate. The full 200-direction KITTI/KITTI-360 matrices and the
  remaining perturbation pairs are still required.
- The first full 200-direction cell subsequently completed on KITTI raw 0018
  at `(0.5 deg, 0.25 m)`. All 200 solver trials converged without a
  computational failure; 84/200 passed the strict hit definition (42%,
  bootstrap mean 95% CI `[35.5%, 49.0%]`). Native output errors were:
  rotation mean/median/p90/p95/max `0.5230/0.5239/0.6812/0.7228/0.8234 deg`
  and translation `0.1447/0.1421/0.1681/0.1803/0.2389 m`. Runtime
  mean/median/p90/p95/max was `106.54/102.19/142.13/154.81/191.00 s`;
  the four-worker run took `1:29:19` wall time and peaked at `995112 KiB`
  RSS. This cell falsifies a robust-recovery claim for the current native
  six-DoF search even at the smaller translation perturbation; it does not
  represent a computational failure.
- The schema-valid problem/protocol/definition/benchmark SHA-256 values for
  that full cell are respectively
  `452bc71fb9a5337414c81457ff8a4d705c6d681bb4d0e715cce095aa4bcda27f`,
  `c35bb1c226e002cf87b1ab604ac759fbd2250f184e49c3a6b362586bd496b709`,
  `9477f613ef7b3518ca707df99ca27fbc1478a6bc6e29060fce35b73d64139f51`,
  and
  `8dbc26f921468480470700fd309a8a3361333f71feab002786425e2bac1a9750`.
  All 200 trace schemas, trial identities, problem/protocol identities, and
  benchmark source digests were independently checked.

Gate:

- Match or exceed D2D's published six-DoF hit rates: `88%` on KITTI and
  `76.5%` on KITTI-360 at `(0.5 deg, 50 cm)`.
- Do not regress rotation-only performance or the existing KITTI 0005 runtime
  regression.
- If translation remains weak while rotation succeeds, pivot translation to
  explicit 2D–3D correspondence/PnP rather than adding more MI heuristics.

### Phase 3 — Probabilistic depth correspondence (weeks 9–12)

Deliver:

- optional shared-depth encoder adapter;
- correspondence mean, covariance/outlier probability, reliability map;
- PnP-RANSAC or robust differentiable pose estimation;
- analytical D2D initialization followed by learned refinement;
- checkpoint and training-corpus provenance.

Implementation checkpoint:

- `slac.probabilistic_correspondence/v0.1` now freezes provider/checkpoint and
  training-corpus lineage together with each 2D--3D mean, positive-definite
  image covariance, outlier probability, and reliability score. It records
  the dataset license separately from the provider license; paired benchmark
  provenance sets `data_verified: false` when the dataset license or source
  digest inventory is missing.
- `slac.probabilistic_pnp_result/v0.1` records the input artifact digest,
  provider identity, robust-estimator options, OpenCV version/license, pose,
  RANSAC support, covariance-normalized inliers, and reprojection diagnostics.
- `calibrex camera-lidar refine-probabilistic-pnp` provides the first optional
  Apache-2.0 OpenCV adapter. `--initial-problem` passes a schema-valid D2D
  problem's declared initial pose as the iterative PnP seed and pins that
  problem's SHA-256 in the result; it is not represented as a solved D2D
  output. The core remains ROS/training-framework/OpenCV independent and
  accepts provider output only through the validated artifact.
- `ProbabilisticCameraLidarRefiner` consumes all provider frames behind that
  contract, starts from the declared D2D transform, fits one shared bounded
  SE(3), and reports disjoint frame holdout. Covariance, outlier probability,
  and reliability are individually frozen ablation switches rather than
  implicit model behavior.
- `refine-probabilistic-multiframe --initial-trace TRACE` uses the output pose
  of a schema-valid D2D candidate trace only after its problem digest and
  camera/LiDAR frame pair match the supplied problem. The result records the
  trace ID, SHA-256, solver status, and original hit decision. Without this
  option the source is explicitly
  `problem_initial_transform`, which is useful for controls but is not called
  a solved-D2D initializer.
- `calibrex camera-lidar benchmark-probabilistic-ablation` executes the full
  estimator plus `without_covariance`, `without_outlier_probability`, and
  `without_reliability` on identical frozen frame splits. Individual results,
  raw benchmark definition, failure-aware aggregates, and paired bootstrap
  intervals are all retained. Its optional `--initial-trace` pins the same
  actual D2D output across every method and split. Real U1--U5 provider output
  is still required before these ablations support a paper claim.
- Each refinement result records the problem's digest-pinned reference
  transform plus initial/final Euclidean translation and
  quaternion-geodesic rotation errors. The paired ablation aggregates these
  reference errors together with held-out reprojection, so U1--U5 cm/deg
  thresholds and their paired confidence intervals can flow directly into
  the SOTA audit rather than being inferred from a pixel objective.
- The current adapter is a typed, provenance-complete evaluation boundary; it
  is not yet evidence for the U1--U5 accuracy or runtime gates. D2D-seeded
  multi-frame refinement and paired synthetic ablations are implemented; a
  real shared-depth provider, frozen U1--U5 executions, and paired real-data
  confidence intervals remain required.

Gate:

- On U1, beat the best-per-metric envelope: `0.550 cm / 0.042 deg`.
- On U2, beat `0.913 cm / 0.053 deg`.
- On U3, beat `2.368 cm / 0.084 deg`.
- The paired confidence interval must support the improvement, and failure
  rate may not increase.
- Cross-domain U4/U5 must improve over the reproduced UniCalib baseline before
  claiming general SOTA. Otherwise claim only in-domain or training-free SOTA.
- Match or exceed UniCalib's roughly 91 ms end-to-end runtime on comparable
  hardware, or establish a better accuracy/runtime Pareto point.
- When CLRNet becomes reproducible, add a separate View-of-Delft gate. A
  multi-frame claim must beat `0.3 cm / 0.03 deg` under its rigid
  `1 m, 20 deg` setting without hiding per-frame failures behind only a
  sequence median.

### Phase 4 — Continuous-time multi-frame refinement (weeks 13–16)

Deliver:

- continuous-time trajectory adapter and native factor boundary;
- per-point LiDAR time, camera exposure/readout time, and scalar clock offset;
- motion compensation, rolling-shutter support, and joint extrinsic/time
  refinement;
- observability and excitation diagnostics.

Implementation checkpoint:

- The existing capture-time evidence path already decodes per-point firing
  times, applies constant-twist deskew, estimates an injected scalar clock
  offset when geometric references exist, and rejects static motion.
- `ContinuousTimeCameraLidarSolver` now joins that timing model to
  covariance-weighted image/LiDAR correspondences and jointly refines a
  bounded SE(3) correction plus scalar clock offset. Training and holdout
  captures are disjoint, every objective candidate is retained, and the
  solver refuses to update when image motion per second is below the declared
  excitation threshold.
- `slac.continuous_time_camera_lidar_problem/v0.1` and
  `slac.continuous_time_camera_lidar_result/v0.1` freeze all timestamps,
  twists, optional piecewise SE(3) trajectory knots, transforms, uncertainty,
  optimizer settings, split evidence, and source digests. The
  `piecewise_se3_linear_slerp/v0.1` adapter interpolates translation and
  shortest-arc rotation in a ROS-independent core, rejects extrapolation, and
  records the chosen trajectory model in every result. Existing
  `slac.trajectory/v0.1` outputs can be attached without ROS dependencies via
  `calibrex camera-lidar attach-continuous-trajectory PROBLEM TRAJECTORY
  --output ADAPTED_PROBLEM`; the adapter verifies the body frame and pins both
  source digests. Before optimization, the solver verifies that the trajectory
  covers every LiDAR firing, rolling-shutter row exposure, and both ends of
  the configured clock-offset search. Problems are executable with
  `calibrex camera-lidar refine-continuous-time PROBLEM --output RESULT`.
- `calibrex camera-lidar benchmark-continuous-time-ablation` runs the full
  model against fixed-clock, no-per-point-time, no-rolling-shutter, and
  no-covariance variants on identical capture splits. Camera row coordinates,
  readout duration, and readout direction determine each rolling-shutter
  exposure time before LiDAR deskew. The command emits individual result
  artifacts plus the raw and aggregated paired benchmark. When the problem
  declares a frozen reference extrinsic and clock offset, every result and
  ablation also reports quaternion-geodesic rotation error, translation
  distance, and absolute clock-offset error; asynchronous real sequences
  remain the required external-validity gate. Dataset license is distinct
  from implementation provenance, and an undeclared license makes the paired
  benchmark `data_verified: false`.
- The native factor now supports rolling-shutter row times and a non-constant
  supplied trajectory, with constant twist retained as an explicit fallback.
  Trajectory estimation itself remains outside the calibration core.
  Schema-valid real-data run artifacts and proof of spatial improvement on
  asynchronous held-out sequences remain open, so the phase gate is not yet
  claimed.

Gate:

- Injected time offsets and extrinsics are recovered on held-out sequences.
- Static or weakly excited segments are rejected rather than silently updated.
- Spatial accuracy improves over Phase 3 on asynchronous real data.

### Phase 5 — Paper-quality audit (weeks 17–18)

Deliver:

- ablations for depth provider, visibility, initialization, uncertainty,
  dynamic masks, frame count, and temporal model;
- exact container/checkpoint/data manifests;
- cross-rig and cross-dataset results;
- failure gallery and limitations;
- archived benchmark artifacts and one-command reproduction.

Audit implementation checkpoint:

- `slac.camera_lidar_sota_audit_protocol/v0.1` freezes the declared category,
  every required benchmark method/metric/statistic/threshold, artifact path
  and digest, minimum dataset-family coverage, and independent-rig IDs.
- `calibrex camera-lidar audit-sota PROTOCOL --output AUDIT` verifies every
  digest and schema, evaluates the numerical gates, and emits
  `supported`, `refuted`, or `incomplete`. The result schema rejects a
  `supported` verdict unless all required gates and coverage minima are
  achieved.
- Frozen numerical requirements can address a benchmark metric's mean,
  median, p90, p95, maximum, or method failure rate. Benchmark artifacts now
  retain p90/p95 explicitly, preventing a favorable average from satisfying a
  claim whose tail-error gate fails. Requirements can also address the lower
  or upper paired-bootstrap improvement confidence bound for an explicit
  reference/candidate pair, making the Phase 3 confidence-interval gate
  machine-enforceable. Runtime and peak-memory mean/p95 gates use the same
  frozen benchmark and digest path. A schema-valid artifact is still
  contradicted as claim evidence when its provenance declares
  `data_verified: false`. Result-model validation requires `refuted` to have
  a contradicted required gate and `incomplete` to have unresolved
  evidence/coverage, in addition to the existing strict `supported` check.
  Only achieved required requirements count toward dataset-family and
  independent-rig coverage; optional smoke tests cannot satisfy claim scope.
- The protocol is an audit mechanism, not evidence creation. Missing learned,
  six-DoF, asynchronous real-data, or independent-rig runs remain incomplete;
  failed frozen gates remain contradicted and cannot be relabeled.

Final claim gate:

- Win the declared category on at least two dataset families and one
  independent rig.
- No primary metric, failure rate, or worst-stratum result is materially worse.
- All claims map to schema-valid artifacts and can be independently rerun.

## First implementation issues

The first three PR-sized issues are:

1. **Generic external-run artifact + UniCalib adapter.**
   This follows the existing development roadmap and establishes a real SOTA
   baseline before native algorithm work.
2. **Depth-map/provider schemas + deterministic D2D evaluator.**
   This creates the GPL/ROS/CUDA-safe boundary and verifies objective topology
   without optimization.
3. **Rotation-only bounded D2D optimizer + exact paper protocol.**
   This is the first go/no-go experiment and must land before six-DoF or
   learned refinement.

## Expected outcome

The likely publishable Calibrex contribution is not “another calibration
network.” It is a hybrid system:

1. training-free D2D global capture;
2. uncertainty-aware depth correspondence and PnP refinement;
3. optional continuous-time spatiotemporal refinement;
4. unusually strong provenance, falsification, failure retention, and
   cross-dataset reproducibility.

If Phase 1 reaches the published D2D capture range, the project has a credible
analytical-SOTA path. If Phase 3 also clears the UniCalib gates across held-out
rigs, Calibrex can pursue a general targetless SOTA claim.
