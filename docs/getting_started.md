# Install and first check

Calibrex has three commands that chain: `calibrex estimate` (no calibration yet),
`calibrex check` (is the deployed one still right?) and `calibrex drift` (did it change between
recordings?). This page installs it and runs the first check; [the workflow
page](tutorials/workflow.md) shows all three with real outputs.

Calibrex is alpha research software with a ROS-independent core. The supported
no-source install is the versioned wheel attached to the GitHub release.

## Install

```bash
python -m pip install "https://github.com/rsasaki0109/Calibrex/releases/download/v0.5.1/calibrex-0.5.1-py3-none-any.whl"
```

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

## Your bag format

The bag is read as it is, with no conversion and no ROS installation; the format is detected from
the file: rosbag2 (`.db3`), ROS 2 MCAP (chunks `none`, `zstd`, `lz4`) and ROS 1 `.bag` (chunks
`none`, `bz2`, `lz4`). `calibrex check my_bag/ --plan` shows which sensor pairs the bag can be
checked for without running an estimator ([details](tutorials/calibrex_check.md#supported-input-formats)).
The ROS 1 and MCAP-lz4 readers, camera-LiDAR (rotation only) and the Autoware Velodyne decoder are on
`main` and ship with the release after v0.5.1; to use them now, install from the repository:
`python -m pip install "git+https://github.com/rsasaki0109/Calibrex.git"`.

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
- [SOTA leaderboard](benchmarks/sota_leaderboard.md): what is, and is not, claimed.
- [Calibrate your own data](tutorials/your_own_data.md) and
  [public datasets](tutorials/public_datasets.md).
