"""Bird's-eye motion-platform README hero GIF helpers for slac."""

from __future__ import annotations

import math
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from slac.core.geometry import SE3
from slac.core.io import read_mapping
from slac.data.odometry_track import OdometryPoseSample, OdometryTrack
from slac.data.rosbag2 import (
    ODOMETRY_TYPE,
    POINTCLOUD2_TYPE,
    decode_odometry,
    decode_pointcloud2,
    iter_messages,
)
from slac.pipelines.online import (
    OnlineCalibrationRunOptions,
    OnlineGateThresholds,
    run_online_calibration,
)

if TYPE_CHECKING:
    pass

INDOOR02_KISSICP_BAG_DIR = Path("data/public/tiers_lidars_dataset/indoor02_rosbag2_kissicp")
INDOOR02_KISSICP_CONFIG = Path(
    "examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_config.yaml"
)
INDOOR02_KISSICP_STORAGE = INDOOR02_KISSICP_BAG_DIR / "indoor02_rosbag2_kissicp.db3"
INDOOR02_VELO_TOPIC = "/velodyne_points"
INDOOR02_OUSTER_TOPIC = "/os_cloud_nodee/points"
INDOOR02_ODOM_TOPIC = "/odom"

MOTION_HERO_FRAME_COUNT = 50
MOTION_HERO_FPS = 10
MOTION_HERO_HOLD_FRAMES = 6
MOTION_HERO_GATE_M = 0.40
MOTION_HERO_MIN_GAP_PX = 8
MOTION_HERO_WIDTH = 960
MOTION_HERO_HEIGHT = 540
MOTION_BOUNDS_MARGIN_M = 4.0
MOTION_RANGE_CROP_M = 20.0
MOTION_GATE_CELL_PX = 8
MOTION_GATE_GAP_PX = 2
MOTION_TEXT_MIN_FOREGROUND_PIXELS = 48
MOTION_FONT_FAMILY = "Noto Sans"
MOTION_REVIEW_FRAME_DIR = Path(
    "/tmp/claude-1000/-home-sasaki-workspace-Calibrex/"
    "a7615978-7430-47f7-be9c-dee6615fc440/scratchpad/herogif-review"
)

INDOOR02_MAP_MAX_SOURCE_POINTS = 8400
INDOOR02_MAP_MAX_REPLAY_DURATION_S = 60.0
INDOOR02_MAP_SUBSAMPLE_SEED = "scan_index"
INDOOR02_GIF_MAX_TARGET_MESSAGES = 36
INDOOR02_GIF_MAX_TARGET_POINTS = 1500
INDOOR02_GIF_MAX_REPLAY_DURATION_S = 60.0
INDOOR02_GIF_BATCH_SIZE = 1500
INDOOR02_GIF_HOLDOUT_RATIO = 0.2
INDOOR02_GIF_ROLLING_WINDOW = 2000
INDOOR02_GIF_MIN_RANK = 6
INDOOR02_GIF_MAX_HOLDOUT_RMSE_M = 0.40
INDOOR02_GIF_MAX_ROLLING_REGRESSION_M = 0.15

MOTION_MAIN_TOP = 72
MOTION_MAIN_BOTTOM = 405
MOTION_HUD_TOP = 405
MOTION_HUD_BOTTOM = 524
MOTION_MARGIN = 16

Point2 = tuple[int, int]
Point3 = tuple[float, float, float]
Color = tuple[int, int, int]

MAP_GREY: Color = (120, 138, 162)
MAP_BRIGHT: Color = (190, 205, 224)
CORRECTION_VECTOR: Color = (251, 191, 36)
CURRENT_SCAN: Color = (34, 211, 238)
TRAJECTORY: Color = (34, 211, 238)
TARGET_ORANGE: Color = (245, 158, 11)
BG: Color = (10, 15, 27)
PANEL: Color = (18, 27, 43)
GRID: Color = (55, 68, 90)
TEXT_DIM: Color = (148, 163, 184)
GOOD: Color = (34, 197, 94)
WARNING: Color = (245, 158, 11)
FAIL: Color = (251, 113, 133)


@dataclass(frozen=True)
class LayoutRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class MotionHeroBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class MotionHeroScan:
    timestamp_ns: int
    points_world: tuple[Point3, ...]


@dataclass(frozen=True)
class MotionHeroTargetBatch:
    timestamp_ns: int
    points_sensor: tuple[Point3, ...]


@dataclass(frozen=True)
class MotionHeroFrameState:
    scan_index: int
    batch_index: int
    holdout_rmse_history: tuple[float | None, ...]
    gate_statuses: tuple[str, ...]
    accepted_batch_count: int
    batch_count: int
    current_holdout_rmse_m: float | None
    batch_transform: SE3


@dataclass(frozen=True)
class MotionHeroScene:
    bag_dir: Path
    timeline_path: Path
    odom_track: OdometryTrack
    scans: tuple[MotionHeroScan, ...]
    target_batches: tuple[MotionHeroTargetBatch, ...]
    trajectory_xy_yaw: tuple[tuple[float, float, float], ...]
    bounds: MotionHeroBounds
    batch_transforms: tuple[SE3, ...]
    holdout_rmses: tuple[float | None, ...]
    gate_statuses: tuple[str, ...]
    accepted_batch_count: int
    batch_count: int
    frame_states: tuple[MotionHeroFrameState, ...]
    initial_batch_transform: SE3
    gate_thresholds: dict[str, float | int]
    final_gate_status: str
    rejected_batch_count: int
    inconclusive_batch_count: int
    trajectory_bbox_m: dict[str, float]
    view_window_m: dict[str, float]
    scan_count: int
    scans_per_animation_frame: float


def indoor02_kissicp_bag_available() -> bool:
    return INDOOR02_KISSICP_STORAGE.is_file()


def indoor02_map_replay_budgets(*, scan_count: int | None = None) -> dict[str, object]:
    budgets: dict[str, object] = {
        "max_source_points": INDOOR02_MAP_MAX_SOURCE_POINTS,
        "max_replay_duration_s": INDOOR02_MAP_MAX_REPLAY_DURATION_S,
        "subsample_seed": INDOOR02_MAP_SUBSAMPLE_SEED,
    }
    if scan_count is not None:
        budgets["max_source_messages"] = scan_count
    return budgets


def indoor02_calibration_replay_budgets() -> dict[str, object]:
    return {
        "max_target_messages": INDOOR02_GIF_MAX_TARGET_MESSAGES,
        "max_target_points": INDOOR02_GIF_MAX_TARGET_POINTS,
        "max_replay_duration_s": INDOOR02_GIF_MAX_REPLAY_DURATION_S,
        "batch_size": INDOOR02_GIF_BATCH_SIZE,
        "holdout_ratio": INDOOR02_GIF_HOLDOUT_RATIO,
        "rolling_window": INDOOR02_GIF_ROLLING_WINDOW,
    }


def indoor02_gif_replay_budgets(*, scan_count: int | None = None) -> dict[str, object]:
    return {
        "map_view": indoor02_map_replay_budgets(scan_count=scan_count),
        "calibration_run": indoor02_calibration_replay_budgets(),
    }


def motion_hero_generation_command(output: Path) -> str:
    return (
        "python3 tools/generate_calibration_evidence_gif.py "
        f"--source tiers-indoor02-kissicp --visual motion "
        f"--output {output} --frames {MOTION_HERO_FRAME_COUNT}"
    )


def run_indoor02_motion_hero_pipeline(
    *,
    frames: int,
    timeline_path: Path | None = None,
) -> MotionHeroScene:
    from slac.core.online_timeline import OnlineCalibrationTimelineArtifact

    if not indoor02_kissicp_bag_available():
        raise SystemExit(
            f"{INDOOR02_KISSICP_STORAGE} is required for the motion hero GIF."
        )

    gate_thresholds = OnlineGateThresholds(
        min_rank=INDOOR02_GIF_MIN_RANK,
        max_holdout_rmse_m=INDOOR02_GIF_MAX_HOLDOUT_RMSE_M,
        max_rolling_regression_m=INDOOR02_GIF_MAX_ROLLING_REGRESSION_M,
    )
    if timeline_path is None:
        with tempfile.TemporaryDirectory(prefix="slac_motion_hero_online_") as tmp_name:
            output_dir = Path(tmp_name) / "outputs"
            result = run_online_calibration(
                INDOOR02_KISSICP_CONFIG,
                OnlineCalibrationRunOptions(
                    output_dir=output_dir,
                    batch_size=INDOOR02_GIF_BATCH_SIZE,
                    rolling_window=INDOOR02_GIF_ROLLING_WINDOW,
                    holdout_ratio=INDOOR02_GIF_HOLDOUT_RATIO,
                    gate_thresholds=gate_thresholds,
                ),
            )
            if result is None:
                raise SystemExit("motion hero requires a real online calibration run")
            timeline_path = Path(result.run.provenance["online_timeline_path"])
            timeline = OnlineCalibrationTimelineArtifact.model_validate(read_mapping(timeline_path))
    else:
        timeline = OnlineCalibrationTimelineArtifact.model_validate(read_mapping(timeline_path))

    if not timeline.batches:
        raise SystemExit("online calibration produced no timeline batches for the motion hero")

    scans, target_batches, trajectory, odom_track = load_indoor02_motion_bag_data()
    trajectory_bbox_m = trajectory_bbox(trajectory)
    bounds = compute_motion_bounds(trajectory)
    view_window_m = {
        "x_min": bounds.x_min,
        "x_max": bounds.x_max,
        "y_min": bounds.y_min,
        "y_max": bounds.y_max,
    }
    initial_batch_transform = _load_initial_extrinsic_from_config(INDOOR02_KISSICP_CONFIG)
    scans = _crop_motion_scans(scans, bounds, odom_track)
    batch_transforms = tuple(_transform_result_to_se3(batch.estimate) for batch in timeline.batches)
    holdout_rmses = tuple(batch.batch_holdout_rmse_m for batch in timeline.batches)
    gate_statuses = tuple(batch.gate_status for batch in timeline.batches)
    scan_count = len(scans)
    anim_frames = max(1, frames - MOTION_HERO_HOLD_FRAMES)
    scans_per_animation_frame = scan_count / max(1, anim_frames)
    frame_states = build_motion_hero_frame_states(
        batch_count=len(timeline.batches),
        accepted_batch_count=timeline.accepted_batch_count,
        holdout_rmses=holdout_rmses,
        gate_statuses=gate_statuses,
        batch_transforms=batch_transforms,
        scan_count=scan_count,
        frames=frames,
    )
    return MotionHeroScene(
        bag_dir=INDOOR02_KISSICP_BAG_DIR,
        timeline_path=timeline_path,
        odom_track=odom_track,
        scans=scans,
        target_batches=target_batches,
        trajectory_xy_yaw=trajectory,
        bounds=bounds,
        batch_transforms=batch_transforms,
        holdout_rmses=holdout_rmses,
        gate_statuses=gate_statuses,
        accepted_batch_count=timeline.accepted_batch_count,
        batch_count=len(timeline.batches),
        frame_states=frame_states,
        initial_batch_transform=initial_batch_transform,
        gate_thresholds={
            "min_rank": gate_thresholds.min_rank,
            "max_holdout_rmse_m": gate_thresholds.max_holdout_rmse_m,
            "max_rolling_regression_m": gate_thresholds.max_rolling_regression_m,
        },
        final_gate_status=timeline.final_gate_status,
        rejected_batch_count=timeline.rejected_batch_count,
        inconclusive_batch_count=timeline.inconclusive_batch_count,
        trajectory_bbox_m=trajectory_bbox_m,
        view_window_m=view_window_m,
        scan_count=scan_count,
        scans_per_animation_frame=scans_per_animation_frame,
    )


def load_indoor02_motion_bag_data() -> tuple[
    tuple[MotionHeroScan, ...],
    tuple[MotionHeroTargetBatch, ...],
    tuple[tuple[float, float, float], ...],
    OdometryTrack,
]:
    bag_dir = INDOOR02_KISSICP_BAG_DIR
    topics = {INDOOR02_VELO_TOPIC, INDOOR02_OUSTER_TOPIC, INDOOR02_ODOM_TOPIC}
    odom_samples: list[OdometryPoseSample] = []
    velo_messages: list[tuple[int, object]] = []
    ouster_messages: list[tuple[int, object]] = []
    replay_start_ns: int | None = None
    target_messages = 0

    for connection, timestamp_ns, data in iter_messages(bag_dir, topics=topics):
        if connection.topic == INDOOR02_ODOM_TOPIC and connection.message_type == ODOMETRY_TYPE:
            message = decode_odometry(connection.topic, timestamp_ns, data)
            pose = SE3(message.position, message.orientation_xyzw)
            odom_samples.append(OdometryPoseSample(timestamp_ns, pose))
            continue
        if connection.message_type != POINTCLOUD2_TYPE:
            continue
        if connection.topic == INDOOR02_VELO_TOPIC:
            if replay_start_ns is None:
                replay_start_ns = timestamp_ns
            if INDOOR02_MAP_MAX_REPLAY_DURATION_S is not None:
                elapsed_ns = timestamp_ns - replay_start_ns
                if elapsed_ns > int(INDOOR02_MAP_MAX_REPLAY_DURATION_S * 1_000_000_000):
                    continue
            message = decode_pointcloud2(connection.topic, timestamp_ns, data)
            velo_messages.append((timestamp_ns, message))
            continue
        if connection.topic != INDOOR02_OUSTER_TOPIC:
            continue
        if replay_start_ns is None:
            replay_start_ns = timestamp_ns
        if INDOOR02_GIF_MAX_REPLAY_DURATION_S is not None:
            elapsed_ns = timestamp_ns - replay_start_ns
            if elapsed_ns > int(INDOOR02_GIF_MAX_REPLAY_DURATION_S * 1_000_000_000):
                continue
        if target_messages >= INDOOR02_GIF_MAX_TARGET_MESSAGES:
            continue
        message = decode_pointcloud2(connection.topic, timestamp_ns, data)
        ouster_messages.append((timestamp_ns, message))
        target_messages += 1

    if not odom_samples or not velo_messages:
        raise SystemExit("Indoor02 kissicp bag missing odometry or Velodyne scans")

    odom_track = OdometryTrack(odom_samples)
    scans: list[MotionHeroScan] = []
    points_budget = INDOOR02_MAP_MAX_SOURCE_POINTS
    for scan_index, (timestamp_ns, message) in enumerate(velo_messages):
        pose, _, _ = odom_track.interpolate(timestamp_ns)
        remaining = max(1, points_budget // max(1, len(velo_messages) - scan_index))
        world_points = _transform_xyz_subsample(message.xyz, pose, remaining, seed=scan_index)
        scans.append(MotionHeroScan(timestamp_ns, world_points))
    trajectory = _trajectory_from_scans(scans, odom_track)

    target_batches: list[MotionHeroTargetBatch] = []
    per_batch = max(1, INDOOR02_GIF_MAX_TARGET_POINTS // max(1, len(ouster_messages)))
    for batch_index, (timestamp_ns, message) in enumerate(ouster_messages):
        sensor_points = _subsample_xyz(message.xyz, per_batch, seed=1000 + batch_index)
        target_batches.append(MotionHeroTargetBatch(timestamp_ns, sensor_points))

    return tuple(scans), tuple(target_batches), trajectory, odom_track


def build_motion_hero_frame_states(
    *,
    batch_count: int,
    accepted_batch_count: int,
    holdout_rmses: tuple[float | None, ...],
    gate_statuses: tuple[str, ...],
    batch_transforms: tuple[SE3, ...],
    scan_count: int,
    frames: int,
) -> tuple[MotionHeroFrameState, ...]:
    anim_frames = max(1, frames - MOTION_HERO_HOLD_FRAMES)
    states: list[MotionHeroFrameState] = []
    for frame_index in range(frames):
        progress = (
            1.0
            if frame_index >= anim_frames
            else frame_index / max(1, anim_frames - 1)
        )
        batch_index = min(batch_count - 1, round(progress * max(1, batch_count - 1)))
        # Map view replays the full-sequence odometry path; HUD batches stay on the
        # bounded online-calibration timeline. Both advance on the same animation clock.
        scan_index = min(scan_count - 1, round(progress * max(1, scan_count - 1)))
        visible = batch_index + 1
        visible_gate_statuses = gate_statuses[:visible]
        accepted_so_far = sum(1 for status in visible_gate_statuses if status == "pass")
        states.append(
            MotionHeroFrameState(
                scan_index=scan_index,
                batch_index=batch_index,
                holdout_rmse_history=holdout_rmses[:visible],
                gate_statuses=visible_gate_statuses,
                accepted_batch_count=accepted_so_far,
                batch_count=batch_count,
                current_holdout_rmse_m=holdout_rmses[batch_index],
                batch_transform=batch_transforms[batch_index],
            )
        )
    return tuple(states)


def trajectory_bbox(
    trajectory: tuple[tuple[float, float, float], ...],
) -> dict[str, float]:
    if not trajectory:
        raise RuntimeError("motion hero requires a non-empty trajectory for bbox")
    xs = [x for x, _y, _yaw in trajectory]
    ys = [y for _x, y, _yaw in trajectory]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }


def compute_motion_bounds(
    trajectory: tuple[tuple[float, float, float], ...],
    *,
    margin_m: float = MOTION_BOUNDS_MARGIN_M,
) -> MotionHeroBounds:
    if not trajectory:
        raise RuntimeError("motion hero requires a non-empty trajectory for bounds")
    xs = [x for x, _y, _yaw in trajectory]
    ys = [y for _x, y, _yaw in trajectory]
    return MotionHeroBounds(
        x_min=min(xs) - margin_m,
        x_max=max(xs) + margin_m,
        y_min=min(ys) - margin_m,
        y_max=max(ys) + margin_m,
    )


def _point_in_bounds(x: float, y: float, bounds: MotionHeroBounds) -> bool:
    return bounds.x_min <= x <= bounds.x_max and bounds.y_min <= y <= bounds.y_max


def _crop_motion_scans(
    scans: tuple[MotionHeroScan, ...],
    bounds: MotionHeroBounds,
    odom_track: OdometryTrack,
) -> tuple[MotionHeroScan, ...]:
    cropped: list[MotionHeroScan] = []
    for scan in scans:
        pose, _, _ = odom_track.interpolate(scan.timestamp_ns)
        rig_x, rig_y = pose.translation_m[0], pose.translation_m[1]
        kept: list[Point3] = []
        for x, y, z in scan.points_world:
            if not _point_in_bounds(x, y, bounds):
                continue
            if math.hypot(x - rig_x, y - rig_y) > MOTION_RANGE_CROP_M:
                continue
            kept.append((x, y, z))
        cropped.append(MotionHeroScan(scan.timestamp_ns, tuple(kept)))
    return tuple(cropped)


def _load_initial_extrinsic_from_config(config_path: Path) -> SE3:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    initial = config["frames"]["ouster_os1"]["transform"]["initial"]
    return SE3.from_lists(initial["translation"], initial["rotation_quat_xyzw"])


def draw_motion_hero_frame(
    image: bytearray,
    *,
    scene: MotionHeroScene,
    frame_state: MotionHeroFrameState,
    width: int,
    height: int,
    draw_pixel,
    draw_helpers: object,
) -> None:
    fill_rect = draw_helpers.fill_rect
    line = draw_helpers.line
    circle = draw_helpers.circle
    thick_line = draw_helpers.thick_line

    fill_rect(image, 0, 0, width, height, BG)
    main_height = MOTION_MAIN_BOTTOM - MOTION_MAIN_TOP
    main = LayoutRect(MOTION_MARGIN, MOTION_MAIN_TOP, width - 2 * MOTION_MARGIN, main_height)
    fill_rect(image, main.x, main.y, main.width, main.height, PANEL)
    line(image, main.x, main.y, main.right, main.y, GRID, alpha=0.85)
    line(image, main.x, main.bottom, main.right, main.bottom, GRID, alpha=0.85)
    line(image, main.x, main.y, main.x, main.bottom, GRID, alpha=0.85)
    line(image, main.right, main.y, main.right, main.bottom, GRID, alpha=0.85)

    project = _make_projector(main, scene.bounds)
    _draw_equal_grid(image, main, scene.bounds, project, line, fill_rect)

    for scan_index, scan in enumerate(scene.scans[: frame_state.scan_index + 1]):
        age = frame_state.scan_index - scan_index
        is_current = scan_index == frame_state.scan_index
        if is_current:
            color = CURRENT_SCAN
            alpha = 1.0
            radius = 3
        else:
            fade = max(0.45, 1.0 - age / max(1, frame_state.scan_index))
            color = mix(MAP_GREY, MAP_BRIGHT, fade)
            alpha = 0.55 + 0.35 * fade
            radius = 2
        for point_index, point in enumerate(scan.points_world):
            if not is_current and point_index % 2:
                continue
            px, py = project(point[0], point[1])
            circle(image, px, py, radius, color, alpha=alpha)

    traj = scene.trajectory_xy_yaw[: frame_state.scan_index + 1]
    for index in range(1, len(traj)):
        x0, y0, _ = traj[index - 1]
        x1, y1, _ = traj[index]
        thick_line(
            image,
            *project(x0, y0),
            *project(x1, y1),
            TRAJECTORY,
            thickness=3,
            alpha=1.0,
        )
    if traj:
        x, y, yaw = traj[-1]
        px, py = project(x, y)
        _draw_heading_triangle(image, px, py, yaw, draw_pixel)

    if frame_state.batch_index < len(scene.target_batches):
        batch = scene.target_batches[frame_state.batch_index]
        pose, _, _ = scene.odom_track.interpolate(batch.timestamp_ns)
        t_world_current = pose.compose(frame_state.batch_transform)
        t_world_initial = pose.compose(scene.initial_batch_transform)
        centroid = _batch_centroid_sensor(batch.points_sensor)
        initial_world = t_world_initial.transform_point(centroid)
        current_world = t_world_current.transform_point(centroid)
        if (
            _point_in_bounds(*initial_world[:2], scene.bounds)
            and _point_in_bounds(*current_world[:2], scene.bounds)
        ):
            p0 = project(initial_world[0], initial_world[1])
            p1 = project(current_world[0], current_world[1])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 3:
                thick_line(
                    image,
                    p0[0],
                    p0[1],
                    p1[0],
                    p1[1],
                    CORRECTION_VECTOR,
                    thickness=2,
                    alpha=0.95,
                )
                circle(image, p0[0], p0[1], 3, CORRECTION_VECTOR, alpha=0.9)
        rig_x, rig_y = pose.translation_m[0], pose.translation_m[1]
        for point_index, point in enumerate(batch.points_sensor):
            if point_index % 2:
                continue
            wx, wy, _wz = t_world_current.transform_point(point)
            if not _point_in_bounds(wx, wy, scene.bounds):
                continue
            if math.hypot(wx - rig_x, wy - rig_y) > MOTION_RANGE_CROP_M:
                continue
            px, py = project(wx, wy)
            circle(image, px, py, 3, TARGET_ORANGE, alpha=0.95)

    _draw_scale_bar(image, main, scene.bounds, project, line, fill_rect)
    _draw_motion_hud(image, frame_state, width, draw_helpers)


def motion_hero_layout(width: int) -> dict[str, LayoutRect]:
    hud_height = MOTION_HUD_BOTTOM - MOTION_HUD_TOP
    spark_w = round(width * 0.58)
    gate_w = width - 2 * MOTION_MARGIN - spark_w - MOTION_HERO_MIN_GAP_PX
    spark = LayoutRect(MOTION_MARGIN, MOTION_HUD_TOP + 12, spark_w, hud_height - 36)
    gate = LayoutRect(
        spark.right + MOTION_HERO_MIN_GAP_PX,
        MOTION_HUD_TOP + 12,
        gate_w,
        hud_height - 36,
    )
    title1 = LayoutRect(MOTION_MARGIN, 14, 920, 22)
    title2 = LayoutRect(MOTION_MARGIN, 44, 920, 18)
    footnote = LayoutRect(width - 620, MOTION_HUD_BOTTOM - 8, 604, 14)
    rmse_label = LayoutRect(spark.x + spark.width - 92, spark.y + 4, 84, 16)
    counter = LayoutRect(gate.x, gate.bottom - 14, gate.width, 14)
    spark_label = LayoutRect(spark.x, spark.y - 2, 180, 16)
    gate_label = LayoutRect(gate.x, gate.y - 2, 180, 16)
    _verify_layout_gaps(title1, title2, spark, gate, footnote)
    return {
        "title1": title1,
        "title2": title2,
        "spark": spark,
        "gate": gate,
        "rmse_label": rmse_label,
        "counter": counter,
        "footnote": footnote,
        "spark_label": spark_label,
        "gate_label": gate_label,
    }


def _verify_layout_gaps(*rects: LayoutRect) -> None:
    for left_index, left in enumerate(rects):
        for right in rects[left_index + 1 :]:
            overlap_x = left.x < right.right and right.x < left.right
            overlap_y = left.y < right.bottom and right.y < left.bottom
            if overlap_x and overlap_y:
                raise RuntimeError(f"motion hero layout overlap: {left} vs {right}")
            if overlap_x:
                gap_y = max(left.y - right.bottom, right.y - left.bottom)
                if 0 <= gap_y < MOTION_HERO_MIN_GAP_PX:
                    raise RuntimeError(
                        f"motion hero layout vertical gap too small: {left} vs {right}"
                    )
            if overlap_y:
                gap_x = max(left.x - right.right, right.x - left.right)
                if 0 <= gap_x < MOTION_HERO_MIN_GAP_PX:
                    raise RuntimeError(
                        f"motion hero layout horizontal gap too small: {left} vs {right}"
                    )


def resolve_motion_font() -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", MOTION_FONT_FAMILY],
        text=True,
    ).strip()
    if not font_file:
        raise RuntimeError(f"motion hero font resolution failed for {MOTION_FONT_FAMILY!r}")
    path = Path(font_file)
    if not path.is_file():
        raise FileNotFoundError(f"motion hero font file missing: {font_file}")
    return str(path)


def build_motion_frame_text_filter(
    frame_state: MotionHeroFrameState,
    *,
    width: int = MOTION_HERO_WIDTH,
) -> str:
    font_file = resolve_motion_font()
    layout = motion_hero_layout(width)
    labels: list[tuple[str, int, int, int, str]] = [
        (
            "slac — simultaneous localization and calibration",
            layout["title1"].x,
            layout["title1"].y,
            20,
            "E5E7EB",
        ),
        (
            (
                "TIERS Indoor02 (real data): Velodyne world map + KISS-ICP odometry; "
                "Ouster extrinsic converges online"
            ),
            layout["title2"].x,
            layout["title2"].y,
            13,
            "CBD5E1",
        ),
        (
            (
                "map: full-sequence odometry replay; "
                "gates/RMSE: real bounded slac calibrate --online run"
            ),
            layout["footnote"].x,
            layout["footnote"].y,
            11,
            "94A3B8",
        ),
        (
            "holdout RMSE / batch",
            layout["spark_label"].x,
            layout["spark_label"].y,
            13,
            "CBD5E1",
        ),
        (
            "gate verdict / batch",
            layout["gate_label"].x,
            layout["gate_label"].y,
            13,
            "CBD5E1",
        ),
        (
            f"{frame_state.accepted_batch_count}/{frame_state.batch_count} batches adopted",
            layout["counter"].x,
            layout["counter"].y,
            12,
            "CBD5E1",
        ),
    ]
    if frame_state.current_holdout_rmse_m is not None:
        labels.append(
            (
                f"{frame_state.current_holdout_rmse_m:.2f} m",
                layout["rmse_label"].x,
                layout["rmse_label"].y,
                12,
                "E5E7EB",
            )
        )
    return ",".join(_drawtext(font_file, *label) for label in labels)


def build_motion_text_filter(*, frame_state: object | None = None) -> str:
    if frame_state is None:
        raise RuntimeError("motion hero text filter requires a frame state")
    return build_motion_frame_text_filter(frame_state)


def bake_motion_frame_text(
    ppm_path: Path,
    frame_state: MotionHeroFrameState,
    *,
    width: int = MOTION_HERO_WIDTH,
    height: int = MOTION_HERO_HEIGHT,
) -> None:
    overlay = build_motion_frame_text_filter(frame_state, width=width)
    tmp_path = ppm_path.with_name(ppm_path.stem + "_text.ppm")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(ppm_path),
            "-vf",
            overlay,
            "-frames:v",
            "1",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"motion hero ffmpeg drawtext failed for {ppm_path.name}: {result.stderr.strip()}"
        )
    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"motion hero text bake produced no output for {ppm_path.name}")
    tmp_path.replace(ppm_path)
    pixels, read_width, read_height = read_ppm_pixels(ppm_path)
    if read_width != width or read_height != height:
        raise RuntimeError(
            f"motion hero text bake size mismatch for {ppm_path.name}: "
            f"{read_width}x{read_height} != {width}x{height}"
        )
    verify_motion_text_rendered(pixels, read_width, read_height, frame_state)


def read_ppm_pixels(path: Path) -> tuple[bytearray, int, int]:
    header = path.read_bytes()[:64]
    if not header.startswith(b"P6"):
        raise RuntimeError(f"unsupported ppm format in {path}")
    first_newline = header.find(b"\n")
    second_newline = header.find(b"\n", first_newline + 1)
    if min(first_newline, second_newline) < 0:
        raise RuntimeError(f"invalid ppm header in {path}")
    width_str, height_str = header[first_newline + 1 : second_newline].decode("ascii").split()
    width = int(width_str)
    height = int(height_str)
    expected = width * height * 3
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    pixels = bytearray(result.stdout)
    if len(pixels) != expected:
        raise RuntimeError(
            f"ppm pixel payload size mismatch in {path}: {len(pixels)} != {expected}"
        )
    return pixels, width, height


def _is_background_pixel(red: int, green: int, blue: int) -> bool:
    background_colors = (BG, PANEL, (13, 20, 34))
    return any(
        abs(red - color[0]) + abs(green - color[1]) + abs(blue - color[2]) <= 18
        for color in background_colors
    )


def _pixel_differs_from_background(red: int, green: int, blue: int) -> bool:
    return not _is_background_pixel(red, green, blue)


def count_foreground_pixels_in_rect(
    pixels: bytearray,
    width: int,
    *,
    rect: LayoutRect,
) -> int:
    count = 0
    x_end = min(width, rect.right)
    y_end = rect.bottom
    for y in range(rect.y, y_end):
        row = y * width * 3
        for x in range(rect.x, x_end):
            index = row + x * 3
            if _pixel_differs_from_background(pixels[index], pixels[index + 1], pixels[index + 2]):
                count += 1
    return count


def verify_motion_text_rendered(
    pixels: bytearray,
    width: int,
    height: int,
    frame_state: MotionHeroFrameState,
) -> None:
    layout = motion_hero_layout(width)
    required = {
        "title1": layout["title1"],
        "title2": layout["title2"],
        "footnote": layout["footnote"],
        "spark_label": layout["spark_label"],
        "gate_label": layout["gate_label"],
        "counter": layout["counter"],
    }
    if frame_state.current_holdout_rmse_m is not None:
        required["rmse_label"] = layout["rmse_label"]
    for name, rect in required.items():
        if rect.right > width or rect.bottom > height:
            raise RuntimeError(f"motion hero text rect {name} exceeds frame bounds")
        foreground = count_foreground_pixels_in_rect(pixels, width, rect=rect)
        if foreground < MOTION_TEXT_MIN_FOREGROUND_PIXELS:
            raise RuntimeError(
                f"motion hero text not rendered in {name}: "
                f"{foreground} foreground pixels < {MOTION_TEXT_MIN_FOREGROUND_PIXELS}"
            )


def write_motion_review_frames(
    scene: MotionHeroScene,
    *,
    output_dir: Path,
    draw_frame_fn,
    bake_text_fn,
) -> list[Path]:
    indices = [0, 12, 28, MOTION_HERO_FRAME_COUNT - 1]
    names = ["first", "early-convergence", "mid", "final"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in zip(indices, names, strict=True):
        ppm_path = output_dir / f"{name}.ppm"
        draw_frame_fn(scene, index, ppm_path)
        frame_state = scene.frame_states[index]
        bake_text_fn(ppm_path, frame_state)
        png_path = output_dir / f"{name}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(ppm_path),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(png_path),
            ],
            check=True,
        )
        ppm_path.unlink(missing_ok=True)
        paths.append(png_path)
    return paths


def _draw_motion_hud(
    image: bytearray,
    frame_state: MotionHeroFrameState,
    width: int,
    draw_helpers: object,
) -> None:
    layout = motion_hero_layout(width)
    spark = layout["spark"]
    gate = layout["gate"]
    fill_rect = draw_helpers.fill_rect
    line = draw_helpers.line
    circle = draw_helpers.circle
    rect = draw_helpers.rect

    fill_rect(image, spark.x, spark.y, spark.width, spark.height, (13, 20, 34))
    rect(image, spark.x, spark.y, spark.width, spark.height, GRID, alpha=0.9)
    values = [value for value in frame_state.holdout_rmse_history if value is not None]
    if values:
        y_min = min(min(values), MOTION_HERO_GATE_M) * 0.95
        y_max = max(max(values), MOTION_HERO_GATE_M) * 1.05
        chart = LayoutRect(spark.x + 8, spark.y + 22, spark.width - 16, spark.height - 30)
        gate_ratio = (MOTION_HERO_GATE_M - y_min) / (y_max - y_min)
        gate_y = chart.y + round((1.0 - gate_ratio) * chart.height)
        line(image, chart.x, gate_y, chart.right, gate_y, GOOD, alpha=0.55)
        for grid_index in range(1, 4):
            gy = chart.y + grid_index * chart.height // 4
            line(image, chart.x, gy, chart.right, gy, GRID, alpha=0.35)
        prev: Point2 | None = None
        for index, value in enumerate(frame_state.holdout_rmse_history):
            if value is None:
                continue
            ratio = (value - y_min) / max(1e-9, y_max - y_min)
            history_len = max(1, len(frame_state.holdout_rmse_history) - 1)
            px = chart.x + round(index / history_len * chart.width)
            py = chart.bottom - round(ratio * chart.height)
            if prev is not None:
                line(image, prev[0], prev[1], px, py, CURRENT_SCAN, alpha=0.9)
            prev = (px, py)
        if prev is not None:
            fill_height = chart.bottom - prev[1]
            fill_width = prev[0] - chart.x + 1
            fill_rect(
                image,
                chart.x,
                prev[1],
                fill_width,
                fill_height,
                CURRENT_SCAN,
                alpha=0.12,
            )
            circle(image, prev[0], prev[1], 4, CURRENT_SCAN, alpha=1.0)

    strip_height = gate.height - 18
    fill_rect(image, gate.x, gate.y, gate.width, strip_height, (13, 20, 34))
    rect(image, gate.x, gate.y, gate.width, strip_height, GRID, alpha=0.9)
    cell = MOTION_GATE_CELL_PX
    gap = MOTION_GATE_GAP_PX
    cols = max(1, (gate.width + gap) // (cell + gap))
    rows = max(1, math.ceil(len(frame_state.gate_statuses) / cols))
    if len(frame_state.gate_statuses) > cols and rows < 2:
        rows = 2
        cols = max(1, math.ceil(len(frame_state.gate_statuses) / rows))
    grid_width = cols * cell + max(0, cols - 1) * gap
    grid_height = rows * cell + max(0, rows - 1) * gap
    start_x = gate.x + max(0, (gate.width - grid_width) // 2)
    start_y = gate.y + max(0, (strip_height - grid_height) // 2)
    for index, status in enumerate(frame_state.gate_statuses):
        row = index // cols
        col = index % cols
        cx = start_x + col * (cell + gap)
        cy = start_y + row * (cell + gap)
        if status == "pass":
            color = GOOD
        elif status == "fail":
            color = FAIL
        else:
            color = WARNING
        fill_rect(image, cx, cy, cell, cell, color, alpha=0.98)


def _draw_equal_grid(image, main, bounds, project, line, fill_rect) -> None:
    span = max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min)
    step = 2.0
    if span > 20:
        step = 5.0
    elif span > 10:
        step = 2.0
    x = math.floor(bounds.x_min / step) * step
    while x <= bounds.x_max:
        p0 = project(x, bounds.y_min)
        p1 = project(x, bounds.y_max)
        line(image, p0[0], p0[1], p1[0], p1[1], GRID, alpha=0.22)
        x += step
    y = math.floor(bounds.y_min / step) * step
    while y <= bounds.y_max:
        p0 = project(bounds.x_min, y)
        p1 = project(bounds.x_max, y)
        line(image, p0[0], p0[1], p1[0], p1[1], GRID, alpha=0.22)
        y += step


def _draw_scale_bar(image, main, bounds, project, line, fill_rect) -> None:
    bar_m = 2.0
    x0 = bounds.x_min + 1.0
    y0 = bounds.y_min + 1.0
    p0 = project(x0, y0)
    p1 = project(x0 + bar_m, y0)
    thick = 3
    fill_rect(image, p0[0], p0[1] - thick // 2, max(1, p1[0] - p0[0]), thick, TEXT_DIM, alpha=0.95)
    line(image, p0[0], p0[1], p1[0], p1[1], TEXT_DIM, alpha=0.95)


def _draw_heading_triangle(image: bytearray, px: int, py: int, yaw: float, draw_pixel) -> None:
    tip_x = px + round(math.cos(yaw) * 14)
    tip_y = py - round(math.sin(yaw) * 14)
    base_center_x = px - round(math.cos(yaw) * 4)
    base_center_y = py + round(math.sin(yaw) * 4)
    left_x = base_center_x + round(math.cos(yaw + math.pi / 2) * 7)
    left_y = base_center_y - round(math.sin(yaw + math.pi / 2) * 7)
    right_x = base_center_x + round(math.cos(yaw - math.pi / 2) * 7)
    right_y = base_center_y - round(math.sin(yaw - math.pi / 2) * 7)
    triangle = ((tip_x, tip_y), (left_x, left_y), (right_x, right_y))
    min_y = min(point[1] for point in triangle)
    max_y = max(point[1] for point in triangle)
    for y in range(min_y, max_y + 1):
        intersections: list[float] = []
        for index in range(3):
            x0, y0 = triangle[index]
            x1, y1 = triangle[(index + 1) % 3]
            if y0 == y1:
                continue
            if (y >= min(y0, y1)) and (y < max(y0, y1)):
                intersections.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        if len(intersections) < 2:
            continue
        intersections.sort()
        x_start = round(intersections[0])
        x_end = round(intersections[-1])
        for x in range(x_start, x_end + 1):
            draw_pixel(image, x, y, TRAJECTORY, alpha=1.0)


def _batch_centroid_sensor(points: tuple[Point3, ...]) -> Point3:
    if not points:
        return (0.0, 0.0, 0.0)
    sx = sum(point[0] for point in points)
    sy = sum(point[1] for point in points)
    sz = sum(point[2] for point in points)
    count = float(len(points))
    return (sx / count, sy / count, sz / count)


def _make_projector(main: LayoutRect, bounds: MotionHeroBounds):
    span = max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min, 1e-6)

    def project(x: float, y: float) -> Point2:
        nx = (x - bounds.x_min) / span
        ny = (y - bounds.y_min) / span
        px = main.x + round(nx * (main.width - 8)) + 4
        py = main.bottom - round(ny * (main.height - 8)) - 4
        return px, py

    return project


def _trajectory_from_scans(
    scans: list[MotionHeroScan],
    odom_track: OdometryTrack,
) -> tuple[tuple[float, float, float], ...]:
    trajectory: list[tuple[float, float, float]] = []
    for scan in scans:
        pose, _, _ = odom_track.interpolate(scan.timestamp_ns)
        yaw = _yaw_from_quat(pose.rotation_quat_xyzw)
        trajectory.append((pose.translation_m[0], pose.translation_m[1], yaw))
    return tuple(trajectory)


def _yaw_from_quat(quat: tuple[float, float, float, float]) -> float:
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _transform_xyz_subsample(xyz, pose: SE3, max_points: int, *, seed: int) -> tuple[Point3, ...]:
    points = _subsample_xyz(xyz, max_points, seed=seed)
    return tuple(pose.transform_point(point) for point in points)


def _subsample_xyz(xyz, max_points: int, *, seed: int) -> tuple[Point3, ...]:
    count = len(xyz)
    if count <= max_points:
        return tuple((float(row[0]), float(row[1]), float(row[2])) for row in xyz)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(count), max_points))
    return tuple((float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])) for i in indices)


def _transform_result_to_se3(transform_result: object) -> SE3:
    return SE3.from_lists(
        transform_result.translation_m,
        transform_result.rotation_quat_xyzw,
    )


def mix(left: Color, right: Color, ratio: float) -> Color:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(round(left[index] + (right[index] - left[index]) * ratio) for index in range(3))


def _drawtext(font_file: str, text: str, x: int, y: int, size: int, color: str) -> str:
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("%", "\\%")
    )
    escaped_font = font_file.replace("\\", "\\\\").replace(":", "\\:")
    return (
        "drawtext="
        f"fontfile='{escaped_font}':"
        f"text='{escaped_text}':"
        f"x={x}:y={y}:fontsize={size}:fontcolor=0x{color}"
    )
