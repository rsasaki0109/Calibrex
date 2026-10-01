# calibrex check: yaw injected into the deployed tf (KITTI raw dev drives)

Cells are the pair verdict and `|candidate error| / tolerance` per judged axis. The estimates (and so the closure) do not depend on the candidate.

| deployed tf | overall | lidar-vehicle | lidar-wheel_odometry | ins-lidar | imu-vehicle | closure |
|---|---|---|---|---|---|---|
| vendor (KITTI calibration) | **inconclusive** | **pass** (partial)<br>pitch 0.50/0.50, yaw 0.31/0.50 deg | **pass** (partial)<br>pitch 0.50/0.50, yaw 0.31/0.50 deg | **pass** (partial)<br>roll 0.11/0.50, pitch 0.06/0.50 deg | **inconclusive** | pass<br>pitch -0.00, yaw +0.00 deg |
| velo_link yaw +1 deg | **fail** | **fail** (partial)<br>pitch 0.51/0.50, yaw 1.31/0.50 deg | **fail** (partial)<br>pitch 0.50/0.50, yaw 1.31/0.50 deg | **pass** (partial)<br>roll 0.11/0.50, pitch 0.06/0.50 deg | **inconclusive** | pass<br>pitch -0.00, yaw +0.00 deg |
| velo_link yaw +3 deg | **fail** | **fail** (partial)<br>pitch 0.52/0.50, yaw 3.31/0.50 deg | **fail** (partial)<br>pitch 0.52/0.50, yaw 3.31/0.50 deg | **pass** (partial)<br>roll 0.11/0.50, pitch 0.06/0.50 deg | **inconclusive** | pass<br>pitch -0.00, yaw +0.00 deg |
