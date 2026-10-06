"""Validate the translation excitation predictions against the estimators' empirical std.

Two steps, so the slow LiDAR odometry runs once per recording:

``collect``  runs the odometry once and pickles the odometry windows (and the IMU
             stream or the GNSS track) to a work file.
``analyze``  evaluates time prefixes of the pickled windows (what ``--max-duration-s``
             / ``--gnss-max-duration-s`` would analyse) and random subsets of them,
             and writes one JSON with, per prefix and axis: the predicted (analytic)
             std, the jackknife std, the excitation diagnosis, and, per subset size,
             the spread of the subset estimates, the empirical std that the
             predicted std is judged against.

Examples (one heavy job at a time; the bags are not shipped)::

    python tools/validate_translation_observability.py collect imu-lidar \\
        --bag ros2/stadtgarten_seq2 --lidar-topic /livox/points --imu-topic /livox/imu \\
        --rotation cached_rotation.yaml --max-seconds 600 --work seq2_imu.pkl
    python tools/validate_translation_observability.py analyze imu-lidar \\
        --work seq2_imu.pkl --durations 60,120,240,480,600 --subsets 6,12,24 --out seq2_imu.json

    python tools/validate_translation_observability.py collect gnss-lidar \\
        --bag ros2/stadtgarten_seq2 --rtk rtk_slam_eval/data/stadtgarten_seq2/rtk.txt \\
        --max-seconds 900 --work seq2_gnss.pkl
    python tools/validate_translation_observability.py analyze gnss-lidar \\
        --work seq2_gnss.pkl --durations 120,240,480,900 --subsets 5,10,19 --out seq2_gnss.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("x", "y", "z")


def _prefix(windows: Sequence[Any], start: float, seconds: float) -> list[Any]:
    return [window for window in windows if window.times_s[-1] <= start + seconds]


def _subset_stats(
    items: Sequence[Any],
    sizes: Sequence[int],
    draws: int,
    fit: Callable[[Sequence[Any]], tuple[list[float], list[float]] | None],
    seconds_per_item: float,
) -> list[dict[str, Any]]:
    """Spread of the estimate over random subsets of ``m`` of the ``N`` windows.

    Sampling without replacement from ``N`` windows understates the spread of
    independent draws by ``sqrt(1 - m / N)``, which is divided out.
    """

    rng = np.random.default_rng(0)
    result = []
    for size in sizes:
        if size >= len(items):
            continue
        estimates, analytic = [], []
        for _ in range(draws):
            chosen = sorted(rng.choice(len(items), size=size, replace=False).tolist())
            solved = fit([items[index] for index in chosen])
            if solved is None:
                continue
            estimates.append(solved[0])
            analytic.append(solved[1])
        correction = 1.0 / math.sqrt(1.0 - size / len(items))
        array = np.array(estimates)
        result.append(
            {
                "windows": size,
                "of": len(items),
                "seconds": size * seconds_per_item,
                "draws": len(estimates),
                "empirical_std_m": (np.std(array, axis=0, ddof=1) * correction).tolist(),
                "predicted_std_m": np.mean(np.array(analytic), axis=0).tolist(),
                "mean_value_m": array.mean(axis=0).tolist(),
            }
        )
        print(f"subset {size}/{len(items)} windows done", file=sys.stderr)
    return result


def _axis_row(name: str, value: float, analytic: float, jackknife: float | None, exc: Any) -> dict:
    row: dict[str, Any] = {
        "axis": name,
        "value_m": value,
        "std_analytic_m": analytic,
        "std_jackknife_m": jackknife,
    }
    if exc is not None:
        row.update(
            cause=exc.cause,
            efficiency=exc.information_efficiency,
            needed_duration_s=exc.needed_duration_s,
            recording_s=exc.recording_s,
            rotation_rms_deg=exc.rotation_rms_deg,
            perpendicular_rms_deg=exc.perpendicular_rotation_rms_deg,
            missing=list(exc.missing_rotation_axes),
        )
    return row


# ----------------------------------------------------------------------- imu-lidar


def collect_imu(args: argparse.Namespace) -> None:
    from scipy.spatial.transform import Rotation

    from calibrex.check.estimators import detect_point_time
    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
    from calibrex.data.livox_ros2 import LivoxStreamProfile, load_livox_imu
    from calibrex.evaluation.imu_lidar_rotation import ImuLidarRunOptions, collect_livox_windows
    from calibrex.evaluation.imu_lidar_translation import ImuLidarTranslationOptions
    from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, gyro_rotation_model

    bag = Path(args.bag)
    spec = None if args.deskew == "none" else detect_point_time(bag, args.lidar_topic)
    name, encoding = spec if spec else (None, "offset_s")
    profile = LivoxStreamProfile(
        name=f"ros2-pointcloud2:{args.lidar_topic}+imu:{args.imu_topic}",
        point_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        point_time_field=name,
        point_time_encoding=encoding,  # type: ignore[arg-type]
        acceleration_unit=args.acceleration_unit,
    )
    rotation = load_imu_lidar_rotation(args.rotation)
    matrix = Rotation.from_quat(rotation.rotation_quat_xyzw).as_matrix()
    bias = np.asarray(rotation.gyro_bias_rps, dtype=np.float64)
    samples = load_livox_imu(bag, profile)
    gyro = GyroSeries(samples.times_s, samples.gyro_rps)
    rigid = spec is None
    opts = ImuLidarTranslationOptions(deskew="none" if rigid else "gyro")
    run_options = ImuLidarRunOptions(
        windowing=opts.windowing, coverage_margin_s=opts.coverage_margin_s
    )
    _, windows = collect_livox_windows(
        [bag],
        profile,
        run_options,
        rotation_model=None
        if rigid
        else gyro_rotation_model(gyro, matrix, bias, rotation.time_offset_s),
        max_seconds=args.max_seconds,
    )
    payload = {
        "windows": windows,
        "imu": (samples.times_s, samples.gyro_rps, samples.accel_mps2),
        "rotation": matrix,
        "gyro_bias": bias,
        "time_offset_s": rotation.time_offset_s,
        "deskew": "none" if rigid else "gyro",
    }
    Path(args.work).write_bytes(pickle.dumps(payload))
    print(f"{len(windows)} windows -> {args.work}")


def analyze_imu(args: argparse.Namespace) -> None:
    from calibrex.evaluation.imu_lidar_translation import (
        ImuLidarTranslationOptions,
        evaluate_imu_lidar_translation,
    )
    from calibrex.solvers.imu_lidar_translation_solver import (
        ImuPreintegrator,
        solve_translation,
        translation_std_m,
        window_systems,
    )

    data = pickle.loads(Path(args.work).read_bytes())
    windows = data["windows"]
    times, gyro, accel = data["imu"]
    imu = ImuPreintegrator(times, gyro, accel, data["gyro_bias"])
    start = min(float(w.times_s[0]) for w in windows)
    end = max(float(w.times_s[-1]) for w in windows)
    opts = ImuLidarTranslationOptions(deskew=data["deskew"])
    out: dict[str, Any] = {"start_s": start, "end_s": end, "prefixes": [], "subsets": []}
    for seconds in args.durations:
        subset = _prefix(windows, start, seconds)
        evaluation = evaluate_imu_lidar_translation(
            imu, subset, data["rotation"], data["time_offset_s"], opts
        )
        rows = [
            _axis_row(r.name, r.value, r.std_analytic, r.std_jackknife, r.excitation)
            | {"std_sensitivity_m": r.std_segment_sensitivity, "std_reported_m": r.std_reported}
            for r in evaluation.records
        ]
        out["prefixes"].append(
            {
                "seconds": seconds,
                "windows": len(subset),
                "axes": rows,
                "policy": evaluation.policy_status,
            }
        )
        print(f"prefix {seconds:g} s: {len(subset)} windows", file=sys.stderr)
    all_systems = window_systems(windows, imu, data["rotation"], data["time_offset_s"], opts.solver)

    def fit_imu(systems: Sequence[Any]) -> tuple[list[float], list[float]] | None:
        fit = solve_translation(systems, opts.solver)
        if fit.translation_m is None:
            return None
        return fit.translation_m.tolist(), translation_std_m(fit).tolist()

    span = float(np.mean([w.times_s[-1] - w.times_s[0] for w in windows]))
    out["subsets"] = _subset_stats(all_systems, args.subsets, args.draws, fit_imu, span)
    Path(args.out).write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------- gnss-lidar


def collect_gnss(args: argparse.Namespace) -> None:
    from calibrex.data.rtk_slam import iter_livox_scans, load_rtk_track
    from calibrex.evaluation.gnss_lidar_lever_arm import (
        GnssLidarRunOptions,
        _within_seconds,
        collect_windows,
    )
    from calibrex.solvers.gnss_lever_arm_solver import GnssTrackModel

    track = load_rtk_track(args.rtk)
    model = GnssTrackModel(track.times_s, track.enu_m, track.sigma_m)
    opts = GnssLidarRunOptions()
    segmenter = collect_windows(
        model,
        _within_seconds(iter_livox_scans(args.bag, args.lidar_topic), args.max_seconds),
        opts,
        prefix=f"{Path(args.bag).name}/",
    )
    payload = {"windows": segmenter.windows, "track": model}
    Path(args.work).write_bytes(pickle.dumps(payload))
    print(f"{len(segmenter.windows)} windows -> {args.work}")


def analyze_gnss(args: argparse.Namespace) -> None:
    from calibrex.evaluation.gnss_lidar_lever_arm import (
        GnssLidarRunOptions,
        evaluate_gnss_lidar_lever_arm,
    )
    from calibrex.solvers.gnss_lever_arm_solver import GnssLeverArmSolver

    data = pickle.loads(Path(args.work).read_bytes())
    windows, track = data["windows"], data["track"]
    start = min(float(w.times_s[0]) for w in windows)
    end = max(float(w.times_s[-1]) for w in windows)
    opts = GnssLidarRunOptions()
    solver = GnssLeverArmSolver()
    out: dict[str, Any] = {"start_s": start, "end_s": end, "prefixes": [], "subsets": []}
    for seconds in args.durations:
        subset = _prefix(windows, start, seconds)
        evaluation = evaluate_gnss_lidar_lever_arm(track, subset, opts)
        rows = [
            _axis_row(r.name, r.value, r.std_analytic, r.std_jackknife, r.excitation)
            | {"std_reported_m": r.std_reported}
            for r in evaluation.records
            if r.name in AXES
        ]
        out["prefixes"].append(
            {
                "seconds": seconds,
                "windows": len(subset),
                "axes": rows,
                "policy": evaluation.policy_status,
            }
        )
        print(f"prefix {seconds:g} s: {len(subset)} windows", file=sys.stderr)

    def fit_gnss(members: Sequence[Any]) -> tuple[list[float], list[float]] | None:
        fit = solver.solve(track, members, opts.solver)
        if fit.status != "converged" or fit.lever_arm_m is None:
            return None
        return fit.lever_arm_m.tolist(), [fit.dof(name).std for name in AXES]  # type: ignore[arg-type]

    span = float(np.mean([w.times_s[-1] - w.times_s[0] for w in windows]))
    out["subsets"] = _subset_stats(windows, args.subsets, args.draws, fit_gnss, span)
    Path(args.out).write_text(json.dumps(out, indent=1))


def _power_fit(seconds: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    """``values ~ a * seconds ** p`` by least squares in log-log (``a``, ``p``)."""

    slope, intercept = np.polyfit(np.log(seconds), np.log(values), 1)
    return float(np.exp(intercept)), float(slope)


def report(args: argparse.Namespace) -> None:
    """Markdown tables from ``analyze`` outputs given as ``LABEL=FILE``."""

    for item in args.inputs:
        label, _, path = item.partition("=")
        data = json.loads(Path(path).read_text())
        prefixes = data["prefixes"]
        subsets = [s for s in data["subsets"] if s["draws"] > 1]
        print(f"### {label}\n")
        print("Predicted (analytic) against empirical std (mm), random window subsets:\n")
        print(
            "| Windows | Data (s) | Axis | Predicted (analytic, mean) "
            "| Empirical (subset spread) | Empirical / predicted |"
        )
        print("| ---: | ---: | --- | ---: | ---: | ---: |")
        for subset in subsets:
            for index, axis in enumerate(AXES):
                empirical = 1000.0 * subset["empirical_std_m"][index]
                predicted = 1000.0 * subset["predicted_std_m"][index]
                print(
                    f"| {subset['windows']} of {subset['of']} | {subset['seconds']:.0f} | {axis} "
                    f"| {predicted:.1f} | {empirical:.1f} | {empirical / predicted:.1f} |"
                )
        start = next(p for p in prefixes if p["seconds"] >= args.predict_from)
        print(
            f"\nStd predicted from the {start['seconds']:g} s prefix "
            "(larger of analytic and jackknife, scaled by sqrt(T0 / T)) against the empirical "
            "std at the longer subset sizes (mm); fitted exponent of the empirical std in time "
            "(-0.5 for the same motion):\n"
        )
        print("| Axis | Std at T0 | Data (s) | Predicted | Empirical | Predicted / empirical |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        fits = {}
        for index, axis in enumerate(AXES):
            row = start["axes"][index]
            scaling = max(row["std_analytic_m"], row["std_jackknife_m"] or 0.0)
            fits[axis] = _power_fit(
                [s["seconds"] for s in subsets], [s["empirical_std_m"][index] for s in subsets]
            )
            for subset in subsets:
                if subset["seconds"] <= row["recording_s"]:
                    continue
                predicted = scaling * math.sqrt(row["recording_s"] / subset["seconds"])
                empirical = subset["empirical_std_m"][index]
                print(
                    f"| {axis} | {1000.0 * scaling:.1f} | {subset['seconds']:.0f} "
                    f"| {1000.0 * predicted:.1f} | {1000.0 * empirical:.1f} "
                    f"| {predicted / empirical:.2f} |"
                )
        print(
            "\nThe same prediction against the estimator run on the longer prefixes "
            "(the extended recording; larger of analytic and jackknife std, mm):\n"
        )
        print("| Axis | Prefix (s) | Data (s) | Predicted | Actual | Predicted / actual |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for index, axis in enumerate(AXES):
            row = start["axes"][index]
            scaling = max(row["std_analytic_m"], row["std_jackknife_m"] or 0.0)
            seen = set()
            for prefix in prefixes:
                later = prefix["axes"][index]
                if later["recording_s"] <= 1.2 * row["recording_s"]:
                    continue
                if round(later["recording_s"]) in seen:
                    continue
                seen.add(round(later["recording_s"]))
                actual = max(later["std_analytic_m"], later["std_jackknife_m"] or 0.0)
                predicted = scaling * math.sqrt(row["recording_s"] / later["recording_s"])
                print(
                    f"| {axis} | {prefix['seconds']:g} | {later['recording_s']:.0f} "
                    f"| {1000.0 * predicted:.1f} | {1000.0 * actual:.1f} "
                    f"| {predicted / actual:.2f} |"
                )
        print(
            "\nRecording needed to reach a std bound (s of usable data): predicted from the "
            f"{start['seconds']:g} s prefix against where the fitted empirical std crosses it:\n"
        )
        print(
            "| Axis | Fitted exponent | Bound (mm) | Predicted needed | Observed (fit) "
            "| Predicted / observed |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for index, axis in enumerate(AXES):
            row = start["axes"][index]
            scaling = max(row["std_analytic_m"], row["std_jackknife_m"] or 0.0)
            amplitude, exponent = fits[axis]
            for bound in args.bounds:
                predicted = row["recording_s"] * (scaling / bound) ** 2
                observed = (bound / amplitude) ** (1.0 / exponent) if exponent < 0 else math.inf
                ratio = "-" if not math.isfinite(observed) else f"{predicted / observed:.2f}"
                print(
                    f"| {axis} | {exponent:.2f} | {1000.0 * bound:g} | {predicted:.0f} "
                    f"| {observed:.0f} | {ratio} |"
                )
        print("\nDiagnosis per prefix (cause, needed recording, reported std, share kept):\n")
        print("| Prefix (s) | Axis | Cause | Needed (s) | Reported std (mm) | Efficiency |")
        print("| ---: | --- | --- | ---: | ---: | ---: |")
        for prefix in prefixes:
            for row in prefix["axes"]:
                needed = row.get("needed_duration_s")
                print(
                    f"| {prefix['seconds']:g} | {row['axis']} | {row.get('cause', '-')} "
                    f"| {'-' if needed is None else f'{needed:.0f}'} "
                    f"| {1000.0 * row['std_reported_m']:.1f} "
                    f"| {row.get('efficiency') or 0.0:.2f} |"
                )
        print()


def _ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def _floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("collect", "analyze"):
        step_parser = sub.add_parser(step)
        pair = step_parser.add_subparsers(dest="pair", required=True)
        for name in ("imu-lidar", "gnss-lidar"):
            p = pair.add_parser(name)
            p.add_argument("--work", required=True)
            if step == "collect":
                p.add_argument("--bag", required=True)
                p.add_argument("--lidar-topic", default="/livox/points")
                p.add_argument("--max-seconds", type=float, default=None)
                if name == "imu-lidar":
                    p.add_argument("--imu-topic", default="/livox/imu")
                    p.add_argument("--rotation", required=True)
                    p.add_argument("--deskew", choices=("auto", "none"), default="auto")
                    p.add_argument("--acceleration-unit", default="mps2")
                else:
                    p.add_argument("--rtk", required=True)
            else:
                p.add_argument("--durations", type=_floats, required=True)
                p.add_argument("--subsets", type=_ints, default=[])
                p.add_argument("--draws", type=int, default=60)
                p.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("inputs", nargs="+", help="LABEL=analyze_output.json")
    rep.add_argument("--bounds", type=_floats, default=[0.01, 0.005])
    rep.add_argument("--predict-from", type=float, default=60.0, help="prefix seconds")
    args = parser.parse_args()
    if args.step == "report":
        report(args)
        return
    handler = {
        ("collect", "imu-lidar"): collect_imu,
        ("analyze", "imu-lidar"): analyze_imu,
        ("collect", "gnss-lidar"): collect_gnss,
        ("analyze", "gnss-lidar"): analyze_gnss,
    }[(args.step, args.pair)]
    handler(args)


if __name__ == "__main__":
    main()
