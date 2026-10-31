"""KITTI-360 v3 development batch: exact member ranges, spent-only guards, reference chain."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.test_kitti360_to_rosbag2 import SEQUENCE as SEQ
from tests.unit.test_kitti360_to_rosbag2 import _fixture, _oxts_fixture
from tools import kitti360_camera_lidar_v3_batch as batch
from tools import kitti_camera_lidar_v3_replay as replay

from calibrex.check.tf_sources import load_bag_tf_static
from calibrex.data.kitti360_to_rosbag2 import convert_kitti360_to_rosbag2

SEGMENT = {"id": "0000-s00", "sequence": "0000", "start_frame": 0, "frames": 3}


def _archive(members: dict[str, bytes], *, stored: set[str] = frozenset()) -> tuple[bytes, dict]:
    """A real ZIP and the inventory ``kitti360_archive_metadata.inventory`` would report."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            method = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            archive.writestr(name, content, compress_type=method)
    data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        directory = archive.start_dir
    listed = {
        "url": "https://example.invalid/a.zip",
        "size_bytes": len(data),
        "etag": '"e"',
        "members": [
            {
                "name": item.filename,
                "size_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
                "compression": item.compress_type,
                "crc32": f"{item.CRC:08x}",
                "local_header_offset": item.header_offset,
                "is_directory": item.is_dir(),
            }
            for item in infos
        ],
        "provenance": {
            "ranges": [
                {"role": "central_directory", "start": directory, "end": len(data) - 23},
            ]
        },
    }
    return data, listed


def _work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, list]:
    """Inventories of the six archives one 0000 segment reads, served from memory."""

    frames = range(3)
    contents = {
        "calibration": {
            f"calibration/{name}": name.encode()
            for name in ("perspective.txt", "calib_cam_to_pose.txt", "calib_cam_to_velo.txt")
        },
        "perspective-timestamps": {f"{SEQ}/image_00/timestamps.txt": b"t0\n"},
        "velodyne-timestamps": {f"{SEQ}/velodyne_points/timestamps.txt": b"t1\n"},
        "oxts": {
            f"data_poses/{SEQ}/oxts/timestamps.txt": b"t2\n",
            **{f"data_poses/{SEQ}/oxts/data/{i:010d}.txt": b"o%d" % i for i in range(5)},
        },
        "0000-camera00": {
            # A neighbour outside the segment sits between two wanted members.
            **{f"{SEQ}/image_00/data_rect/{i:010d}.png": b"png%d" % i * 50 for i in frames},
            f"{SEQ}/image_01/data_rect/0000000000.png": b"other",
        },
        "0000-velodyne": {
            f"{SEQ}/velodyne_points/data/{i:010d}.bin": bytes([i]) * 64 for i in frames
        },
    }
    reorder = contents["0000-camera00"]
    keys = list(reorder)
    contents["0000-camera00"] = {k: reorder[k] for k in [keys[0], keys[3], keys[1], keys[2]]}
    archives, bindings = {}, {}
    (tmp_path / "inventory").mkdir()
    for name, members in contents.items():
        data, listed = _archive(members, stored={f"{SEQ}/velodyne_points/data/0000000001.bin"})
        listed["url"] = f"https://example.invalid/{name}.zip"
        archives[listed["url"]] = data
        bindings[name] = {k: listed[k] for k in ("url", "size_bytes", "etag")}
        (tmp_path / "inventory" / f"{name}.json").write_text(json.dumps(listed))
    monkeypatch.setattr(batch, "_bindings", lambda: bindings)
    requests: list = []

    def get(url, etag, size, start, end):
        requests.append((url, start, end))
        assert etag == '"e"' and size == len(archives[url])
        return archives[url][start : end + 1]

    return tmp_path, {"contents": contents, "get": get, "archives": archives}, requests


def test_acquire_reads_exact_member_spans_and_verifies_each_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, served, requests = _work(tmp_path, monkeypatch)
    data = tmp_path / "raw"
    receipt = batch.acquire(work, SEGMENT, data, get=served["get"])
    prefix = dict.fromkeys(("perspective-timestamps", "0000-camera00"), "data_2d_raw/")
    prefix |= dict.fromkeys(("velodyne-timestamps", "0000-velodyne"), "data_3d_raw/")
    outside = {
        f"{SEQ}/image_01/data_rect/0000000000.png",
        *(f"data_poses/{SEQ}/oxts/data/{i:010d}.txt" for i in (3, 4)),
    }
    for name, members in served["contents"].items():
        for member, content in members.items():
            target = data / (prefix.get(name, "") + member)
            if member in outside:
                assert not target.exists()
            else:
                assert target.read_bytes() == content
    # The image archive's neighbour is never inside a requested range.
    url = "https://example.invalid/0000-camera00.zip"
    with zipfile.ZipFile(io.BytesIO(served["archives"][url])) as archive:
        neighbour = archive.getinfo(f"{SEQ}/image_01/data_rect/0000000000.png").header_offset
    camera_requests = [(start, end) for u, start, end in requests if u == url]
    assert len(camera_requests) == 2
    assert all(not start <= neighbour <= end for start, end in camera_requests)
    assert receipt["fetched_bytes"] == sum(end - start + 1 for _, start, end in requests)
    assert receipt["archives"]["0000-camera00"]["members"] == 3


def test_acquire_rejects_a_corrupt_member_and_missing_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, served, _ = _work(tmp_path, monkeypatch)

    def corrupt(url, etag, size, start, end):
        payload = bytearray(served["get"](url, etag, size, start, end))
        if "velodyne.zip" in url:
            payload[-1] ^= 0xFF
        return bytes(payload)

    with pytest.raises(ValueError, match=r"CRC-32|deflate|span"):
        batch.acquire(work, SEGMENT, tmp_path / "raw", get=corrupt)
    longer = SEGMENT | {"frames": 6}
    with pytest.raises(ValueError, match="absent"):
        batch.acquire(work, longer, tmp_path / "raw2", get=served["get"])
    with pytest.raises(ValueError, match="fetch cap"):
        batch.acquire(work, SEGMENT, tmp_path / "raw3", get=served["get"], cap_bytes=10)


def test_only_spent_sequences_and_their_fixed_segments_are_accepted() -> None:
    fixed = batch.segments()
    assert len(fixed) == 22 and {s["sequence"] for s in fixed.values()} == {"0000", "0002"}
    for candidate in ("0003", "0004", "0009"):
        with pytest.raises(ValueError, match="spent development sequence"):
            batch.archive_names(candidate)
    assert set(replay.spent_kitti360_segments()) == set(fixed)
    with pytest.raises(ValueError, match="not a spent"):
        replay._require_spent("0004-s00")


def test_inventory_requires_every_metadata_range_to_match_its_binding(tmp_path: Path) -> None:
    bindings = batch._bindings()

    def faithful(url: str) -> dict:
        (pin,) = [item for item in bindings.values() if item["url"] == url]
        return {
            "url": url,
            "size_bytes": pin["size_bytes"],
            "etag": pin["etag"],
            "members": [{}] * pin["member_count"],
            "provenance": {"ranges": pin["provenance"]["ranges"]},
        }

    written = batch.inventory(tmp_path / "ok", read=faithful)
    assert set(written) == {
        "calibration",
        "oxts",
        "perspective-timestamps",
        "velodyne-timestamps",
        "0000-camera00",
        "0000-velodyne",
        "0002-camera00",
        "0002-velodyne",
    }

    def tampered(url: str) -> dict:
        listed = faithful(url)
        ranges = [dict(item) for item in listed["provenance"]["ranges"]]
        ranges[-1]["sha256"] = "0" * 64
        return listed | {"provenance": {"ranges": ranges}}

    with pytest.raises(ValueError, match="SHA-256 bindings"):
        batch.inventory(tmp_path / "bad", read=tampered)
    with pytest.raises(ValueError, match="ETag"):
        batch.inventory(tmp_path / "etag", read=lambda url: faithful(url) | {"etag": '"x"'})


def test_failed_preflight_is_retained_and_a_retry_is_a_new_attempt(tmp_path: Path) -> None:
    report = batch.run_segment("0000-s00", tmp_path, free=lambda: 0)
    assert report.status == "FAIL" and report.stage == "preflight"
    assert "insufficient free disk" in (report.error or "")
    assert report.provenance["evaluation_candidate_bytes_read"] == 0
    second = batch.run_segment("0000-s00", tmp_path, free=lambda: 0)
    assert second.attempt == 2
    assert json.loads((tmp_path / "0000-s00/attempt_1/report.json").read_text())["attempt"] == 1


def test_get_retries_short_bodies_but_never_http_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = batch._tool("kitti360_archive_metadata")
    bodies = [b"abc", b"abcd"]

    class Response:
        status = 206

        def __init__(self, body: bytes) -> None:
            self.body = body
            self.headers = {"Content-Range": "bytes 0-3/10", "ETag": '"e"'}

        def geturl(self) -> str:
            return "https://h/u.zip"

        def read(self, _n: int) -> bytes:
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr(metadata, "urlopen", lambda request, timeout: Response(bodies.pop(0)))
    monkeypatch.setattr(batch.time, "sleep", lambda _s: None)
    assert batch._get("https://h/u.zip", '"e"', 10, 0, 3) == b"abcd"
    monkeypatch.setattr(
        metadata,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            batch.urllib.error.HTTPError("https://h/u.zip", 412, "Precondition Failed", {}, None)
        ),
    )
    with pytest.raises(batch.HttpRejected, match="412"):
        batch._get("https://h/u.zip", '"e"', 10, 0, 3)


def test_kitti360_reference_is_the_serialized_tf_chain_the_check_reads(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _fixture(root)
    _oxts_fixture(root)
    bag = tmp_path / "bag"
    result = convert_kitti360_to_rosbag2(
        root, bag, sequence=SEQ, camera="image_00", motion_source="oxts", max_frames=600
    )
    conversion = json.loads(result.provenance_path.read_text())
    reference = replay._reference(conversion, "0000-s00")
    edges = {(e.parent, e.child): e.transform for e in load_bag_tf_static(bag).edges}

    def matrix(transform) -> np.ndarray:
        value = np.eye(4)
        value[:3, :3] = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
        value[:3, 3] = transform.translation_m
        return value

    chain = np.linalg.inv(matrix(edges[("imu_link", "cam0_optical")])) @ matrix(
        edges[("imu_link", "velo_link")]
    )
    np.testing.assert_allclose(reference, chain, rtol=0, atol=1e-12)
    for wrong in ("0000-s01", "0002-s00"):
        with pytest.raises(ValueError, match="must match this spent segment"):
            replay._reference(conversion, wrong)
    legacy = json.loads(json.dumps(conversion))
    del legacy["camera"]["t_pose_velodyne"]
    with pytest.raises(ValueError, match="t_pose_velodyne"):
        replay._reference(legacy, "0000-s00")


def _frozen_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """A git repository holding a pre-registration that pins the committed segments."""

    import subprocess

    import yaml

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "docs/benchmarks").mkdir(parents=True)
    (root / "src/a.py").write_text("x = 1\n")
    (root / "tools/b.py").write_text("y = 1\n")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "code")
    frozen = git("rev-parse", "HEAD")
    bindings = batch._bindings()
    plan = {
        "frozen_code_commit": frozen,
        "recordings": {
            "exposure": {"owner_confirmation": "confirmed by the owner"},
            "evaluation": json.loads(batch.SEGMENTS.read_text())["evaluation"],
            "archives": {
                name: {k: bindings[name][k] for k in ("size_bytes", "etag")}
                for sequence in batch.CANDIDATES
                for name in batch.archive_names(sequence, evaluation=True)
            },
        },
    }
    path = root / "docs/benchmarks/kitti360_camera_lidar_v3_preregistration.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False))
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "PREREGISTRATION", path)
    return path, {"git": git, "plan": plan}


def test_evaluation_unlocks_only_from_the_committed_frozen_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    path, repo = _frozen_repo(tmp_path, monkeypatch)
    evaluation = batch.verify_preregistration(path)
    assert len(evaluation) == 42
    assert {s["sequence"] for s in evaluation.values()} == set(batch.CANDIDATES)
    with pytest.raises(ValueError, match="repository's v3 pre-registration"):
        batch.verify_preregistration(tmp_path / "other.yaml")
    (repo_root := path.parents[2] / "tools/b.py").write_text("y = 2\n")
    with pytest.raises(ValueError, match="differ from frozen_code_commit"):
        batch.verify_preregistration(path)
    repo["git"]("commit", "-qam", "change code after the freeze")
    with pytest.raises(ValueError, match="differ from frozen_code_commit"):
        batch.verify_preregistration(path)
    repo["git"]("revert", "--no-edit", "HEAD")
    assert repo_root.read_text() == "y = 1\n"
    assert len(batch.verify_preregistration(path)) == 42
    unconfirmed = json.loads(json.dumps(repo["plan"]))
    unconfirmed["recordings"]["exposure"]["owner_confirmation"] = " "
    path.write_text(yaml.safe_dump(unconfirmed, sort_keys=False))
    with pytest.raises(ValueError, match="differs from its committed version"):
        batch.verify_preregistration(path)
    repo["git"]("commit", "-qam", "drop the confirmation")
    with pytest.raises(ValueError, match="exposure confirmation"):
        batch.verify_preregistration(path)


def test_candidates_never_run_without_the_verified_evaluation_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "PREREGISTRATION", tmp_path / "frozen.yaml")
    batch.PREREGISTRATION.write_text("frozen\n")
    candidate = json.loads(batch.SEGMENTS.read_text())["evaluation"][0]
    with pytest.raises(ValueError, match="spent development segments"):
        batch.run(tmp_path, [candidate["id"]])
    with pytest.raises(KeyError):
        batch.run_segment(candidate["id"], tmp_path, free=lambda: 0)
    assert batch.archive_names("0004", evaluation=True)[-1] == "0004-velodyne"
    with pytest.raises(ValueError, match="pre-registered candidate"):
        batch.archive_names("0000", evaluation=True)
    report = batch.run_segment(
        candidate["id"], tmp_path, free=lambda: 0, evaluation={candidate["id"]: candidate}
    )
    assert report.status == "FAIL" and report.provenance["phase"] == "evaluation"
    assert report.provenance["preregistration_sha256"] == hashlib.sha256(b"frozen\n").hexdigest()
    with pytest.raises(ValueError, match="not a spent"):
        replay._require_spent(candidate["id"])
    try:
        replay.authorize_evaluation({candidate["id"]: candidate})
        replay._require_spent(candidate["id"])
    finally:
        replay.authorize_evaluation({})
    with pytest.raises(ValueError, match="not a spent"):
        replay._require_spent(candidate["id"])
