"""Source-bound fresh-process PNG and complete A2D2 calibration comparison.

Both variants use the same installed Pillow/NumPy and public input bytes.
Normal CLI timing includes imports, decoding, solving, diagnostics, saving,
report generation and bundle verification. Pixel digests are collected in
separate untimed workers. No candidate/iteration/diagnostic budget is reduced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402

CASES = ("levinson", "pandey")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def comparable(result: dict, output: str) -> dict:
    """Normalize only declared output paths and run identity/time/source metadata."""
    result = json.loads(json.dumps(result))
    run = result["run"]
    run["id"] = "<RUN>"
    run["created_at"] = "<TIME>"
    run["git_commit"] = "<COMMIT>"
    for key in ("generated_at", "git_commit"):
        run["provenance"][key] = "<" + key.upper() + ">"

    def paths(value):
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        if isinstance(value, str):
            return value.replace(output, "<OUTPUT>")
        return value

    return paths(result)


def worker(protocol_path: Path, variant: str, case: str, folder: Path) -> None:
    protocol = json.loads(protocol_path.read_bytes())
    if sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]:
        raise ValueError("launcher changed")
    source = protocol_path.parent / ("sources-" + variant)
    from calibrex.core import provenance

    # Capsules have no .git directory. Identify their measured revision instead
    # of accidentally recording the enclosing live checkout's current HEAD.
    provenance.git_commit = lambda: protocol["commits"][variant]
    work = {"compiled_decodes": 0, "fallback_decodes": 0}
    if variant == "current":
        from calibrex.data import pillow_png_adapter

        original = pillow_png_adapter._decode_luminance_rows

        def observed(*args):
            result = original(*args)
            work["compiled_decodes" if result is not None else "fallback_decodes"] += 1
            return result

        pillow_png_adapter._decode_luminance_rows = observed
    if case == "pixels":
        import numpy as np

        from calibrex.data.kitti import read_png_luminance

        pixels = {}
        for name in protocol["png_inputs"]:
            image = read_png_luminance(Path(protocol["dataset"]) / name)
            if image is None:
                raise ValueError("PNG failed to decode")
            values = np.asarray(image.rows, dtype="<f8")
            pixels[name] = {
                "width": image.width,
                "height": image.height,
                "sha256_float64_le": hashlib.sha256(values.tobytes()).hexdigest(),
            }
        write(folder / "pixels.json", pixels)
        status = 0
    else:
        from calibrex.cli.main import main

        status = main(
            [
                "calibrate",
                str(protocol_path.parent / (case + ".yaml")),
                "--output-dir",
                str(folder / "artifacts"),
                "--json",
            ]
        )
    imports = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.copy().items()
        if name == "calibrex" or name.startswith("calibrex.")
    }
    write(folder / "imports.json", {"source": str(source), "modules": imports})
    write(folder / "work.json", work)
    raise SystemExit(status)


def _input_hashes(dataset: Path) -> dict[str, str]:
    return {p.name: sha(p) for p in sorted(dataset.iterdir()) if p.is_file()}


def _check_imports(root: Path, variant: str, folder: Path, *, frozen: bool) -> None:
    receipt = json.loads((folder / "imports.json").read_bytes())
    path_type = PureWindowsPath if PureWindowsPath(receipt["source"]).drive else PurePosixPath
    source = path_type(receipt["source"])
    if source.name != "sources-" + variant:
        raise ValueError("import variant differs")
    manifest = json.loads((root / (source.name + "-manifest.json")).read_bytes())
    for name in receipt["modules"].values():
        relative = path_type(name).relative_to(source / "src").as_posix()
        if relative not in manifest or (frozen and sha(Path(name)) != manifest[relative]):
            raise ValueError("worker imported outside its frozen source")
    work = json.loads((folder / "work.json").read_bytes())
    if variant == "current" and (work["compiled_decodes"] < 2 or work["fallback_decodes"]):
        raise ValueError("compiled PNG decoder was not used for both public frames")
    if variant == "prior" and work != {"compiled_decodes": 0, "fallback_decodes": 0}:
        raise ValueError("baseline work counters differ")


def summarize(root: Path, *, frozen: bool = False) -> dict:
    """Validate every full result and recompute timing from all attempts."""
    import yaml
    from jsonschema import validate

    from calibrex.core.validation import validate_file

    protocol = json.loads((root / "protocol.json").read_bytes())
    validate(protocol, json.loads((root / "protocol.schema.json").read_bytes()))
    records = json.loads((root / "attempts.json").read_bytes())
    expected = [
        (case, repeat, variant)
        for case in CASES
        for repeat in range(protocol["repetitions"])
        for variant in (("prior", "current") if repeat % 2 == 0 else ("current", "prior"))
    ]
    if [(r["case"], r["repeat"], r["variant"]) for r in records] != expected or any(
        r["exit_code"] or r["seconds"] <= 0 for r in records
    ):
        raise ValueError("incomplete, reordered or failed attempts")
    outputs = {case: [] for case in CASES}
    for record in records:
        folder = root / f"{record['case']}-{record['repeat']:02d}-{record['variant']}"
        _check_imports(root, record["variant"], folder, frozen=frozen)
        if (folder / "stderr.txt").read_text():
            raise ValueError("unexpected worker stderr")
        path = folder / "artifacts/result.yaml"
        if not validate_file(path).valid:
            raise ValueError("schema-invalid calibration result")
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
        if result["run"]["git_commit"] != protocol["commits"][record["variant"]]:
            raise ValueError("result source metadata differs")
        config_hash = protocol["provenance"]["configs"][record["case"]]
        if result["run"]["config_sha256"] != config_hash:
            raise ValueError("result config binding differs")
        raw_inputs = result["run"]["provenance"]["raw_input_files"]
        if {
            (
                PureWindowsPath(p["path"])
                if PureWindowsPath(p["path"]).drive
                else PurePosixPath(p["path"])
            ).name: p["sha256"]
            for p in raw_inputs
        } != protocol["provenance"]["inputs"]:
            raise ValueError("result public input binding differs")
        display = json.loads((folder / "stdout.txt").read_bytes())
        path_type = PureWindowsPath if PureWindowsPath(display["result"]).drive else PurePosixPath
        output = str(path_type(display["result"]).parent)
        outputs[record["case"]].append(comparable(result, output))
        verification = json.loads((folder / "artifacts/verification.json").read_bytes())
        if not verification["valid"] or verification["issue_count"]:
            raise ValueError("calibration output bundle did not verify")
    summaries = {}
    for case in CASES:
        if any(value != outputs[case][0] for value in outputs[case]):
            raise ValueError("complete calibration result differs: " + case)
        medians = {
            v: statistics.median(
                r["seconds"] for r in records if r["case"] == case and r["variant"] == v
            )
            for v in ("prior", "current")
        }
        summaries[case] = {
            "median_seconds": medians,
            "speedup": medians["prior"] / medians["current"],
            "complete_results_exact": True,
        }
    pixels = []
    for variant in ("prior", "current"):
        folder = root / ("pixels-" + variant)
        _check_imports(root, variant, folder, frozen=frozen)
        pixels.append(json.loads((folder / "pixels.json").read_bytes()))
    if pixels[0] != pixels[1] or set(pixels[0]) != set(protocol["png_inputs"]):
        raise ValueError("decoded pixels differ")
    return {
        "workers": len(records),
        "cases": summaries,
        "pixels_exact": True,
        "provenance": {
            "protocol_sha256": sha(root / "protocol.json"),
            "attempts_sha256": sha(root / "attempts.json"),
        },
    }


def audit(root: Path) -> dict:
    """Audit a published archive on CPU, without the raw data or Pillow."""
    index = json.loads((root / "index.json").read_bytes())
    actual = {
        p.relative_to(root).as_posix(): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "index.json"
    }
    if actual != index["files"]:
        raise ValueError("archive inventory or bytes differ")
    protocol = json.loads((root / "protocol.json").read_bytes())
    if sha(root / "protocol.schema.json") != protocol["provenance"]["schema_sha256"]:
        raise ValueError("protocol schema binding differs")
    manifests = {}
    for variant in ("prior", "current"):
        name = "sources-" + variant
        manifest = json.loads((root / (name + "-manifest.json")).read_bytes())
        if sha(root / (name + "-manifest.json")) != protocol["provenance"]["manifests"][variant]:
            raise ValueError("source manifest binding differs")
        actual = {}
        with tarfile.open(root / (name + ".tar.gz")) as capsule:
            for member in capsule.getmembers():
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:2] != ("src", "calibrex")
                    or path.suffix != ".py"
                    or path.relative_to("src").as_posix() in actual
                ):
                    raise ValueError("unexpected capsule member")
                stream = capsule.extractfile(member)
                assert stream is not None
                actual[path.relative_to("src").as_posix()] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
        if actual != manifest:
            raise ValueError("source capsule differs")
        manifests[variant] = manifest
    changed = {
        name
        for name in manifests["prior"].keys() | manifests["current"].keys()
        if manifests["prior"].get(name) != manifests["current"].get(name)
    }
    if changed != {"calibrex/data/kitti.py", "calibrex/data/pillow_png_adapter.py"}:
        raise ValueError("runtime changes extend beyond PNG decoding")
    for name, field in (
        ("launcher.py.txt", "launcher_sha256"),
        ("freezer.py.txt", "freezer_sha256"),
    ):
        if sha(root / name) != protocol["provenance"][field]:
            raise ValueError("launcher or freezer binding differs")
    for case in CASES:
        if sha(root / (case + ".yaml")) != protocol["provenance"]["configs"][case]:
            raise ValueError("config bytes differ")
    summary = summarize(root)
    if summary != json.loads((root / "summary.json").read_bytes()):
        raise ValueError("reported summary differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=("prior", "current"))
    parser.add_argument("--case", choices=(*CASES, "pixels"))
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.case, args.output_dir)
        return
    if args.audit:
        print(json.dumps(audit(args.audit), indent=2))
        return
    if not all((args.baseline_commit, args.dataset, args.output_dir)) or args.repetitions < 2:
        parser.error(
            "baseline commit, dataset, output directory and at least two pairs are required"
        )
    import yaml

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve()
    inputs = _input_hashes(dataset)
    commits = {
        v: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    manifests = {}
    for variant, commit in commits.items():
        frozen = output / ("sources-" + variant)
        _freeze(frozen, commit)
        shutil.copyfile(frozen / "manifest.json", output / (frozen.name + "-manifest.json"))
        manifests[variant] = sha(frozen / "manifest.json")
        with tarfile.open(output / (frozen.name + ".tar.gz"), "w:gz") as capsule:
            for path in sorted((frozen / "src").rglob("*.py")):
                capsule.add(path, arcname=path.relative_to(frozen).as_posix())
    configs = {}
    base = ROOT / "examples/public_datasets/a2d2_pandey_mutual_information"
    for case, name in (("levinson", "levinson_online_config.yaml"), ("pandey", "config.yaml")):
        payload = yaml.safe_load((base / name).read_text(encoding="utf-8"))
        payload["dataset"]["path"] = str(dataset)
        payload["project"]["output_dir"] = "unused-benchmark-default"
        path = output / (case + ".yaml")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        configs[case] = sha(path)
    helper = ROOT / "tools/benchmark_bspline_lidar_imu.py"
    schema = ROOT / "tools/schemas/png_luminance_benchmark.schema.json"
    for source, name in (
        (Path(__file__), "launcher.py.txt"),
        (helper, "freezer.py.txt"),
        (schema, "protocol.schema.json"),
    ):
        shutil.copyfile(source, output / name)
    protocol = {
        "schema_version": "slac.png_luminance_benchmark/v0.1",
        "commits": commits,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "host_unreserved": True,
        "blas_threads": 1,
        "dataset": str(dataset),
        "png_inputs": sorted(name for name in inputs if name.endswith(".png")),
        "metadata_commit_injection": True,
        "provenance": {
            "inputs": inputs,
            "configs": configs,
            "manifests": manifests,
            "launcher_sha256": sha(Path(__file__)),
            "freezer_sha256": sha(helper),
            "schema_sha256": sha(schema),
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "Pillow", "scipy", "pydantic", "PyYAML")
            },
        },
    }
    write(output / "protocol.json", protocol)
    from jsonschema import validate

    validate(protocol, json.loads(schema.read_bytes()))
    records = []
    schedule = [
        (case, repeat, variant)
        for case in CASES
        for repeat in range(args.repetitions)
        for variant in (("prior", "current") if repeat % 2 == 0 else ("current", "prior"))
    ]
    schedule += [("pixels", None, variant) for variant in ("prior", "current")]
    for case, repeat, variant in schedule:
        frozen = output / ("sources-" + variant)
        _verify_sources(frozen)
        if (
            inputs != _input_hashes(dataset)
            or sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]
            or sha(helper) != protocol["provenance"]["freezer_sha256"]
            or sha(schema) != protocol["provenance"]["schema_sha256"]
            or any(sha(output / (name + ".yaml")) != digest for name, digest in configs.items())
        ):
            raise ValueError("frozen benchmark input, source or protocol changed")
        folder = output / (
            f"{case}-{repeat:02d}-{variant}" if repeat is not None else "pixels-" + variant
        )
        folder.mkdir()
        env = os.environ.copy()
        env.update(
            dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1")
        )
        env.update(PYTHONPATH=str(frozen / "src"), PYTHONDONTWRITEBYTECODE="1")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-protocol",
            str(output / "protocol.json"),
            "--variant",
            variant,
            "--case",
            case,
            "--output-dir",
            str(folder),
        ]
        with (
            (folder / "stdout.txt").open("w") as stdout,
            (folder / "stderr.txt").open("w") as stderr,
        ):
            start = time.perf_counter()
            done = subprocess.run(
                command, cwd=frozen, env=env, stdout=stdout, stderr=stderr, check=False
            )
            seconds = time.perf_counter() - start
        if repeat is not None:
            records.append(
                {
                    "case": case,
                    "repeat": repeat,
                    "variant": variant,
                    "seconds": seconds,
                    "exit_code": done.returncode,
                }
            )
            write(output / "attempts.json", records)
            print(json.dumps(records[-1]), flush=True)
        if done.returncode:
            raise RuntimeError(f"worker failed; all attempts retained at {folder}")
        _verify_sources(frozen)
    write(output / "summary.json", summarize(output, frozen=True))
    print((output / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
