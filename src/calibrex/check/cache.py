"""On-disk cache of ``calibrex check`` estimator artifacts.

The IMU-LiDAR and camera-IMU estimators do not use the candidate calibration:
their estimate, standard deviations and controls are a function of the bag and
the estimator options only. The candidate enters afterwards, as the reference
the artifact is compared with (``reference_value`` / ``error_to_reference``)
and, in the check, as the point the estimate is judged against. So the
artifact is cached under a key of the bag digest, the estimator name, every
estimator option and the calibrex version and a content hash of the estimation source, and the
candidate-dependent reference fields are rewritten for the current candidate on
every hit. Re-checking the same bag with another candidate (or other verdict
thresholds) then costs seconds instead of the estimator's full run.

Estimators that start from the candidate (``lidar-lidar``) are *not* cached.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from calibrex import __version__
from calibrex.core.progress import emit_stage

CACHE_FORMAT = 2  # 2: content-hash code fingerprint (1 keyed on repository state)


class CacheableArtifact(Protocol):
    """A schema artifact that can be saved to YAML."""

    def save(self, path: str | Path) -> None: ...


ArtifactT = TypeVar("ArtifactT", bound=CacheableArtifact)


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/calibrex/check`` (``~/.cache/calibrex/check`` by default)."""

    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "calibrex" / "check"


#: Source files (relative to the package) whose bytes are *not* part of the code
#: fingerprint. A prefix ending in ``/`` excludes a directory. Each entry was
#: checked to be unreachable from the code that produces a cached estimator
#: artifact (``check/estimators.py`` and what it calls) or to only present
#: results. When in doubt a module stays out of this list: a spurious miss costs
#: a recompute, a spurious hit returns a stale estimate.
FINGERPRINT_EXCLUDED: tuple[str, ...] = (
    "cli/",  # argument parsing and printing; calls the library, never imported by it
    "visualization/",  # HTML/SVG reports rendered from finished artifacts
    "check/hints.py",  # next-step and refusal text only
    "check/progress.py",  # terminal progress display (observes events)
    "core/progress.py",  # progress hooks only observe; results are bit-identical on/off
    "check/estimate.py",  # builds exports from cached artifacts; never produces them
    "check/drift.py",  # compares finished per-bag estimates; never produces one
    "core/calibration_drift.py",  # the drift artifact schema
    "data/rosbag2_imu_rotate.py",  # test-bag writer (known-bad control); not read by estimators
    "init_templates.py",  # ``calibrex init`` template text
)

_FINGERPRINT_MEMO: dict[Path, str] = {}


def _is_excluded(relative: str) -> bool:
    return any(
        relative.startswith(entry) if entry.endswith("/") else relative == entry
        for entry in FINGERPRINT_EXCLUDED
    )


def code_fingerprint(root: Path | None = None) -> str:
    """The calibrex version plus a content hash of the estimation source.

    The hash covers the relative path and bytes of every ``*.py`` under
    ``root`` (the installed ``calibrex`` package by default) except
    :data:`FINGERPRINT_EXCLUDED`. It does not depend on version control, so
    commits and edits that leave the estimation code alone keep cached
    estimates valid, and an installed wheel fingerprints the same way as a
    checkout. Memoised per process and root.
    """

    package_dir = (root if root is not None else Path(__file__).resolve().parents[1]).resolve()
    memo = _FINGERPRINT_MEMO.get(package_dir)
    if memo is not None:
        return memo
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix()
        if _is_excluded(relative) or "__pycache__" in path.parts:
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    result = f"{__version__}+src-{digest.hexdigest()[:16]}"
    _FINGERPRINT_MEMO[package_dir] = result
    return result


def _json_default(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def canonical_json(material: Mapping[str, Any]) -> str:
    """Stable JSON of the key material (sorted keys, dataclasses and arrays expanded)."""

    return json.dumps(material, sort_keys=True, default=_json_default, separators=(",", ":"))


@dataclass
class EstimatorCache:
    """A directory of cached estimator artifacts, one YAML and one key manifest each."""

    directory: Path
    code: str = field(default_factory=code_fingerprint)

    def key(self, bag_sha256: str, estimator: str, options: Mapping[str, Any]) -> str:
        """Key of one estimator artifact; ``options`` must hold every setting that changes it."""

        material = {
            "format": CACHE_FORMAT,
            "bag_sha256": bag_sha256,
            "estimator": estimator,
            "options": dict(options),
            "code": self.code,
        }
        return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.directory / f"{key}.yaml", self.directory / f"{key}.json"

    def load(self, key: str, loader: Callable[[Path], ArtifactT]) -> ArtifactT | None:
        """The cached artifact, or ``None`` when absent or unreadable."""

        artifact_path, manifest_path = self._paths(key)
        if not (artifact_path.is_file() and manifest_path.is_file()):
            return None
        try:
            return loader(artifact_path)
        except Exception:  # a corrupt or outdated entry is a miss, not an error
            return None

    def store(
        self,
        key: str,
        artifact: CacheableArtifact,
        bag_sha256: str,
        estimator: str,
        options: Mapping[str, Any],
    ) -> None:
        """Write an artifact atomically; failures to write the cache are not errors."""

        artifact_path, manifest_path = self._paths(key)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.directory) as scratch:
                staged = Path(scratch) / artifact_path.name
                artifact.save(staged)
                manifest = Path(scratch) / manifest_path.name
                manifest.write_text(
                    json.dumps(
                        {
                            "format": CACHE_FORMAT,
                            "key": key,
                            "bag_sha256": bag_sha256,
                            "estimator": estimator,
                            "options": json.loads(canonical_json(options)),
                            "code": self.code,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(staged, artifact_path)
                os.replace(manifest, manifest_path)
        except OSError:
            return


def cached_artifact(
    cache: EstimatorCache | None,
    bag_sha256: str,
    estimator: str,
    options: Mapping[str, Any],
    *,
    loader: Callable[[Path], ArtifactT],
    compute: Callable[[], ArtifactT],
    rebase: Callable[[ArtifactT], ArtifactT],
) -> tuple[ArtifactT, bool]:
    """Return ``(artifact, from_cache)``.

    ``compute`` runs the estimator (on a miss, or without a cache). On a hit
    ``rebase`` rewrites the candidate-dependent reference fields of the cached
    artifact for the current candidate.
    """

    if cache is None:
        return compute(), False
    key = cache.key(bag_sha256, estimator, options)
    hit = cache.load(key, loader)
    if hit is not None:
        emit_stage(f"reusing cached {estimator} estimate")
        return rebase(hit), True
    artifact = compute()
    cache.store(key, artifact, bag_sha256, estimator, options)
    return artifact, False
