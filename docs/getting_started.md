# Install and first check

Calibrex has three commands that chain: `calibrex estimate` (no calibration yet),
`calibrex check` (is the deployed one still right?) and `calibrex drift` (did it change between
recordings?). This page installs it and runs the first check; [the workflow
page](tutorials/workflow.md) shows all three with real outputs.

Calibrex is alpha research software with a ROS-independent core. The supported
no-source install is the versioned wheel attached to the GitHub release.

## Install

```bash
python -m pip install "https://github.com/rsasaki0109/Calibrex/releases/download/v0.6.4/calibrex-0.6.4-py3-none-any.whl"
```

## See a check before you have a bag

```bash
calibrex demo check          # new in 0.6.5
```

It writes a small simulated IMU + 4D radar bag and checks it twice, in a few seconds:
the true mounting passes, and the same mounting turned 4 deg in pitch fails on pitch. The
data is simulated; it shows what the command prints and how to read it.

## Check a recording

Point `calibrex check` at a bag that records `/tf_static` and the sensor
topics (see [your bag format](#your-bag-format)). It reads the deployed transforms, checks every sensor pair it can find,
and writes a self-contained HTML report.

```bash
calibrex check my_bag/ --html check.html
```

Each pair gets a per-axis verdict. A `pass` covers only the axes the data can
judge: pairs with partial coverage name the axes they could not check, and a
pair the data cannot judge reads `inconclusive` rather than `pass`.

On a first run, look before you wait:

```bash
calibrex check my_bag/ --plan                                   # seconds: which pairs can be checked, and why not the others
calibrex check my_bag/ --max-duration-s 120 --html check.html   # the first 120 s
```

LiDAR and camera pairs are slow the first time, because the estimators register every scan
or track every frame: 8.5 min for 40 s of Hesai LiDAR + IMU and 28 min for 180 s of Livox
MID360 with the lever arm, on a shared CPU
([measurements](tutorials/calibrex_check.md#runtime-after-the-speed-up)). The estimate is
cached, so checking another calibration against the same recording takes about 3 s. When
the bag has no `/tf_static`, pass the calibration with `--tf` (URDF, Kalibr, FAST-LIO
config or `frames.yaml`); `--write-frames-template FILE` writes a starter `frames.yaml`
naming the bag's sensor frames for you to fill in.

- The `next steps` block at the end of the output says what to try for each pair that was
  skipped or `inconclusive`.
- The camera pairs `camera-imu` and `camera-focal` need OpenCV: `pip install
  'calibrex[opencv]'` after the wheel. `--plan` says so when it is missing.

## Your bag format

The bag is read as it is, with no conversion and no ROS installation; the format is detected from
the file: rosbag2 (`.db3`), ROS 2 MCAP (chunks `none`, `zstd`, `lz4`) and ROS 1 `.bag` (chunks
`none`, `bz2`, `lz4`). `calibrex check my_bag/ --plan` shows which sensor pairs the bag can be
checked for without running an estimator ([details](tutorials/calibrex_check.md#supported-input-formats)).

**Autoware bags** record raw Velodyne packets; `tools/velodyne_scan_to_pointcloud2.py` (in the repository) writes a
derived bag with one point cloud per sensor, after which `lidar-lidar` gives real verdicts
([the Autoware section](tutorials/calibrex_check.md#autoware-all-sensors-bag1)).

## No calibration yet?

Without a calibration, estimate one from the bag, then check it on another recording:

```bash
calibrex estimate my_bag/ --output est/ --html est.html     # frames.yaml, static transforms, URDF
calibrex check other_bag/ --tf est/frames.yaml --html check.html
calibrex drift day1/ day2/ day3/ --output drift/            # did it change over time?
```

Axes the data did not observe are marked `NOT MEASURED`, never written as measured.

## Next steps

- [The workflow: estimate, check, drift](tutorials/workflow.md): the three commands end to end,
  with real outputs and the browser page.
- [Check a deployed calibration](tutorials/calibrex_check.md): the full tutorial,
  with the KITTI yaw-injection demo.
- [Check a bag in the browser](app/check.html): drop a rosbag2 and plan, check or estimate
  with no install; the files never leave the page
  ([how it works](tutorials/calibrex_check.md#plan-and-run-a-bag-in-the-browser)).
- [Browser calibration](tutorials/browser_calibration.md): calibrate an IMU against
  a trajectory without installing anything; data stays in the page.
- [Calibration CI](tutorials/calibration_ci.md): gate a pipeline on calibration evidence.
- [GPU acceleration](tutorials/gpu_acceleration.md): accelerate repeated Camera–LiDAR
  six-DoF estimation, save a standard result and compare with the CPU backends.
- [SOTA leaderboard](benchmarks/sota_leaderboard.md): what is, and is not, claimed.
- [Calibrate your own data](tutorials/your_own_data.md) and
  [public datasets](tutorials/public_datasets.md).
