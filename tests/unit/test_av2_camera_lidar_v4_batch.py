import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.test_av2_camera_lidar_v4_split import _lister
from tools import av2_camera_lidar_v4_batch as batch
from tools import av2_camera_lidar_v4_split as split
from tools import kitti_camera_lidar_v3_replay as replay


def test_only_the_fixed_validation_development_logs_run(tmp_path: Path) -> None:
    logs = batch.development_logs()
    assert len(logs) == 30 and all(item["split"] == "val" for item in logs.values())
    assert set(replay.spent_av2_logs()) == set(logs)
    evaluation = json.loads(batch.SPLIT.read_text())["evaluation"][0]["log"]
    with pytest.raises(ValueError, match="development logs"):
        batch.run(tmp_path, [evaluation])
    with pytest.raises(ValueError, match="not a spent"):
        replay._require_spent(evaluation)


def test_members_must_match_the_committed_pin() -> None:
    pin = split.pin("val", "00000000-v", _lister())
    listed = batch.members(pin, _lister())
    assert len(listed) == 3 + 2 + 3
    with pytest.raises(ValueError, match="differ from the committed pin"):
        batch.members(pin, _lister(drop="2.feather"))


def test_failed_preflight_is_retained_and_a_retry_is_a_new_attempt(tmp_path: Path) -> None:
    log = next(iter(batch.development_logs()))
    first = batch.run_log(log, tmp_path, free=lambda: 0)
    assert first.status == "FAIL" and first.stage == "preflight"
    assert first.provenance["evaluation_bytes_read"] == 0
    second = batch.run_log(log, tmp_path, free=lambda: 0)
    assert second.attempt == 2 and (tmp_path / log / "attempt_1" / "report.json").is_file()


def test_av2_reference_is_the_serialized_tf_chain(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from tests.unit.test_av2_to_rosbag2 import _log, _matrix

    from calibrex.check.tf_sources import load_bag_tf_static
    from calibrex.data.av2_to_rosbag2 import convert_av2_to_rosbag2

    bag = tmp_path / "bag"
    result = convert_av2_to_rosbag2(_log(tmp_path), bag)
    conversion = json.loads(result.provenance_path.read_text())
    reference = replay._av2_reference(conversion, "0000aaaa-log")
    edges = {(e.parent, e.child): e.transform for e in load_bag_tf_static(bag).edges}
    chain = np.linalg.inv(_matrix(edges[("imu_link", "ring_front_center_optical")])) @ _matrix(
        edges[("imu_link", "velo_link")]
    )
    np.testing.assert_allclose(reference, chain, atol=1e-12)
    with pytest.raises(ValueError, match="must match this development log"):
        replay._av2_reference(conversion, "other-log")
    turned = replay._check_candidate
    turn = Rotation.from_rotvec(np.radians(3.0) * np.eye(3)[2]).as_matrix()
    candidate = reference.copy()
    candidate[:3, :] = turn @ reference[:3, :]
    turned(candidate, reference, "inject_z+3")
