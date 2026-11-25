# lidar-lidar: brief for the next agent (Codex)

This brief is meant to be handed over as is, and it holds everything needed to start
without asking. Work on the branch `claude/nifty-hypatia-6nsmas`, which has every commit
listed under "Where things stand", or on a branch cut from it. Do not open a PR, and do not
post anything to an external service (a GitHub issue to GrandTour, for example), unless the
user explicitly asks for it.

## Read first

1. `AGENTS.md`: the project rules, which include provenance, schema stability and that
   calibration changes need tests or evaluation.
2. `docs/sota_handoff.md`: pre-registration, spent data, and how verdicts are handled.
   Its "Spent data" table already lists GrandTour.
3. `docs/benchmarks/grand_tour_lidar_lidar_dev.md`: the development note.
4. `docs/benchmarks/grand_tour_lidar_lidar.md`: the v1 audit (refuted) and its post hoc
   analysis.
5. `docs/tutorials/unitree_go2_lidar_lidar.md`: the end goal, calibrating the Go2 L1
   against an added LiDAR.

## Where things stand (commit `52075034`)

**Implemented**

- Per-point placement on the reference trajectory, in
  `src/calibrex/evaluation/lidar_lidar_map.py` and
  `src/calibrex/solvers/lidar_lidar_map_solver.py`:
    - `--target-motion reference_trajectory`;
    - target scan accumulation (`--target-accumulate-scans`);
    - clock offset estimation (`--target-time-offset`, `--time-offset-search`).
- `--preset unitree-go2-l1` for the Go2.
- `--map-half-window-scans` sets the map window. 0 registers each target scan against the
  nearest reference scan alone, so the reference odometry cancels.
- `--window-sensitivity HALF_WINDOW` refits on the same train blocks with a different map
  window. Each DoF's shift goes into `std_systematic`, and `std_reported` must cover it.
- GrandTour harness tools:
    - `tools/grand_tour_zarr_to_rosbag2.py`: Zarr to rosbag2. `--sparse` emulates an
      L1-like target.
    - `tools/grand_tour_same_time_alignment.py`: odometry-free check on same-time scans.
    - `tools/grand_tour_lidar_lidar_evaluate.py`: runs the evaluation once.
    - `tools/build_grand_tour_lidar_lidar_audit.py`: the frozen scorer and audit builder.

**v1 audit result: refuted, 2/6** (`docs/assets/grand_tour_lidar_lidar/evaluation/`)

- **Passed:**
    - same-time alignment within 35 mm on every mission; the dataset's tf misses by
      89–225 mm;
    - the sparse target stays within 0.25 deg of the dense estimate in rotation.
- **Failed:**
    - cross-mission consistency: 1.52 against a bound of 1.0;
    - dense pass on every mission: 11-14 warned;
    - sparse pass on every mission: pitch unobservable on 2 of 4 missions;
    - sparse translation to dense: 2.4 cm against a bound of 1 cm.

**Post hoc** (`docs/assets/grand_tour_lidar_lidar/post_hoc/`)

- **Translation scatter.** The cause is odometry drift over the ±5-scan map. With map
  window 0, all five missions pass and the z scatter drops from 1.6 cm to 0.6 cm. The
  systematic term covers the translation scatter.
- **11-04.** Its yaw differs by 0.3 deg, and the odometry-free estimate reproduces it. A
  real change of the mount is likely.
- **Unresolved: rotation scatter.** Excluding 11-04, roll and yaw still scatter by about
  0.1 deg between missions. The reported std is only 0.01–0.04 deg and does not cover it.
  This is the main topic of Task A.

**Data**

- Source: Hugging Face `leggedrobotics/grand_tour_dataset`, MIT license, revision
  `af6b6dd6f91696f8ec8c98134e195514834911b2`.
- Rig: ANYmal D with the Boxi payload. Hesai XT-32 (reference), Livox MID360 (target,
  mounted upside-down), Velodyne VLP-16.
- **Spent missions.** Use these for development only, never for evaluation:
    - 2024-11-02-17-43-10 (development);
    - 2024-11-03-07-52-45, 2024-11-04-13-07-13, 2024-11-11-12-42-47 and 2024-11-14-13-45-37
      (v1 evaluation).
- **Unread.** No method has read the sensor data of any other mission. Choose a v2
  evaluation set from them by a rule that uses metadata only. The tf metadata of 13
  missions has been read, which is fine.

## Task A: find the cause of the ~0.1 deg rotation scatter (spent data only)

Goal: explain the rotation scatter between missions. If it can be fixed, fix it. If it
cannot, decide on an honest uncertainty that covers it.

Use the five spent missions. Test the hypotheses below one at a time, and record a number
for each, whether it confirms or rejects the hypothesis.

1. **A small Hesai-Livox clock offset.** A few ms of offset times the yaw rate turns into
   a yaw bias. Run each mission with `--target-motion reference_trajectory
   --time-offset-search 0.02 0.002`, and plot each mission's offset against the yaw
   difference to the mean.
   - The full-mission offset search builds one candidate at a time. It holds about one
     dense mission in memory.
2. **Scene geometry.** A scene whose plane normals are biased makes some rotation axes
   weak.
   - Per 10 s block, compute the normal distribution: the eigenvalues of the normals'
     covariance.
   - Compare it with the per-block estimates. The block estimates come from the
     jackknife; save them, or add an option to export them.
   - Check whether the missions' rotation offsets correlate with their geometry.
3. **Systematic error in the Hesai intrinsics.** Ring elevations or per-ring range bias
   would make the result depend on which rings dominate. Look for a ring dependence by
   refitting with:
     - one ring at a time, or one group of rings at a time;
     - points restricted to one range band at a time.
4. **Deskew.** Compare four runs on the same mission: `rigid_nearest` against
   `reference_trajectory` with K=1, each with and without `deskew: none`. The Livox and
   Hesai sweeps are both about 0.1 s long.
5. **A real mount change between days.** This needs a rotation estimate that does not
   depend on the LiDARs:
    - option 1: a rotation-only hand-eye between the Livox's built-in IMU
      (`livox_imu.tar`, small) and the Hesai odometry;
    - option 2: one between `livox_imu` and another IMU (`stim320_imu` or `cpt7_imu`).
   The repository has imu-lidar rotation tools, starting with `calibrex.check`. If the
   IMU-based rotation scatters by mission in the same way, the change is real.
   - Reading the IMUs of a spent mission is allowed. Record it in the spent-data table.

**Deliverable:** `docs/research/lidar_lidar_rotation_scatter.md`.

- One line per hypothesis, with the number behind it.
- The conclusion: the cause, the fix, or a rotation floor that covers the scatter.
- Label everything post hoc on spent data.
- If you fix the code, add tests: synthetic data, as in
  `tests/unit/test_lidar_lidar_trajectory.py`.

## Task B: v2 pre-registration and one evaluation on unread missions

Do this only after Task A has decided what the method and its uncertainty should be.

1. **Fix the method.** The candidate:
    - dense: `--map-half-window-scans 0`, plus `--window-sensitivity 5` as a check in
      the other direction (or the method Task A settles on);
    - sparse: the `unitree-go2-l1`-equivalent settings.
2. **Choose the claim.** The candidates:
    - `std_reported` covers the cross-mission scatter (a coverage claim);
    - consistency;
    - same-time alignment;
    - for sparse: pass, or an honest inconclusive. An honest per-axis claim, which gates
      only the DoFs the method judges plus a coverage gate, is the pattern that brought
      camera-imu to supported. See `docs/sota_handoff.md`.
3. **Thresholds.** Derive them from the five spent missions with the rules in
   `docs/sota_handoff.md`:
    - an error gate is 1.5 × the worst development value, rounded up;
    - a coverage floor is the development rate ÷ 1.5, rounded down.
4. **Choose the evaluation set.** Use metadata only:
    - list each mission's file sizes and LFS oids (= SHA-256) with
      `curl https://huggingface.co/api/datasets/leggedrobotics/grand_tour_dataset/tree/<rev>/<mission>/data`;
    - keep to unread missions, from dates other than v1's, about 4–6 of them;
    - pin the SHA-256 of each mission's three files: `hesai_points.tar`,
      `livox_points.tar` and `tf.tar`.
5. **Freeze.** Follow the v1 order:
    - commit the code; that is the `frozen_code_commit`;
    - in a separate commit holding only docs, write the pre-registration (the claim,
      missions, pins, method flags, gates, thresholds and the rerun policy);
    - push.
   Templates: `docs/benchmarks/grand_tour_lidar_lidar_preregistration.yaml` and
   `tests/unit/test_grand_tour_lidar_lidar_audit.py`.
   - The v1 scorer in `tools/build_grand_tour_lidar_lidar_audit.py` is frozen. Create v2
     as a new file and change nothing in v1.
6. **Run once.**
    - First confirm that `git diff <frozen> HEAD -- src tools` is empty.
    - Then run with `nohup`.
    - A run takes 1–1.5 hours: about 15 min of runtime per GB of data.
    - Rerun only under the pre-registered policy.
7. **Publish.**
    - Score, then build the audit.
    - Update the leaderboard. Reuse the audit list in the `provenance.command` of
      `docs/assets/sota_leaderboard.json` and add v2 to it.
    - Update the README, `docs/index.md`, `docs/sota_handoff.md` (standings and spent
      data) and `CHANGELOG.md`.
    - A refuted verdict is published the same way. Never change a verdict.

## Practical notes (pitfalls already hit)

- **Environment.**
    - `uv sync --extra dev`, then `UV_FROZEN=1 uv run ...`.
    - `uv run` can rewrite `uv.lock`. Run `git checkout uv.lock` before committing.
    - Zarr is not a package dependency: `uv run --with "zarr<3" python tools/...`.
- **Formatting.** Run `ruff format` only on the files you changed. On
  `src/calibrex/cli/main.py` it rewrites hundreds of unrelated lines.
- **Docs.** `mkdocs build --strict` fails with the latest mkdocs-material. Pin it:
  `uv run --with "mkdocs-material>=9.5,<9.7" --with "mkdocs-include-markdown-plugin>=6.0"
  mkdocs build --strict`.
- **Schemas.** After changing a schema, run `calibrex schema all --output-dir schemas`,
  then `python tools/check_schema_compat.py`. Keep changes to added optional fields.
- **Point timestamps.** `time` is absolute seconds for the Hesai (`absolute_s`), absolute
  ns as float64 for the Livox (`absolute_ns`), and an offset for the Velodyne
  (`offset_s`).
- **Reference.** The dataset's static tf does not align the released clouds. Use it only
  as an initial value, never as an accuracy reference. This is the conclusion of the
  development note.
- **Disk.** A converted dense bag is about 2× the size of its tars. Delete bags when a run
  ends.
- **Memory.** The map, and the per-point poses in `reference_trajectory`, are large. Do
  not hold many sample sets at once.
- **Commits.** End every commit message with the attribution lines the user's environment
  requires. Never put a model name in a commit.
