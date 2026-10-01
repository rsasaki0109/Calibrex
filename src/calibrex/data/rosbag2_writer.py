"""ROS-free rosbag2 writer: a ``metadata.yaml`` and one sqlite3 ``.db3`` file.

The output is read by :mod:`calibrex.data.rosbag2`. Payloads are CDR bytes made
by :mod:`calibrex.data.ros_cdr_writer`. Latched topics such as ``/tf_static``
are recorded with a transient-local QoS profile, as ``ros2 bag record`` does.

The ``topics`` and ``messages`` tables and ``metadata.yaml`` follow the
rosbag2 sqlite3 layout; the file has not been opened with a ROS 2 install, so
playback with ``ros2 bag play`` is untested here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

from calibrex.core.exceptions import DatasetError

LATCHED_QOS_PROFILE = (
    "- history: 3\n"
    "  depth: 0\n"
    "  reliability: 1\n"
    "  durability: 1\n"
    "  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  liveliness: 1\n"
    "  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  avoid_ros_namespace_conventions: false\n"
)
"""``offered_qos_profiles`` of a transient-local (latched) topic such as ``/tf_static``."""

_SCHEMA = """
CREATE TABLE topics(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  serialization_format TEXT NOT NULL,
  offered_qos_profiles TEXT NOT NULL
);
CREATE TABLE messages(
  id INTEGER PRIMARY KEY,
  topic_id INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  data BLOB NOT NULL
);
CREATE INDEX timestamp_idx ON messages (timestamp ASC);
"""


@dataclass
class _Topic:
    topic_id: int
    name: str
    message_type: str
    qos: str
    count: int = 0


class Rosbag2Writer:
    """Write a rosbag2 directory (``<directory>/<name>_0.db3`` and ``metadata.yaml``).

    Use as a context manager; ``metadata.yaml`` is written on close. Messages are
    stored in the order written; readers sort them by timestamp.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        custom_data: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            if not overwrite:
                msg = f"bag directory already exists and is not empty: {self.directory}"
                raise DatasetError(msg)
            for stale in self.directory.glob("*.db3"):
                stale.unlink()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._db_name = f"{self.directory.name}_0.db3"
        self._db_path = self.directory / self._db_name
        self._connection: sqlite3.Connection | None = sqlite3.connect(self._db_path)
        self._connection.executescript(_SCHEMA)
        self._topics: dict[str, _Topic] = {}
        self._message_id = 0
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self._custom_data = dict(custom_data or {})

    def add_topic(self, name: str, message_type: str, *, latched: bool = False) -> None:
        """Declare a topic (``latched`` records a transient-local QoS profile)."""

        if self._connection is None:
            raise DatasetError("the bag writer is closed")
        if name in self._topics:
            existing = self._topics[name]
            if existing.message_type != message_type:
                msg = f"topic {name} already declared as {existing.message_type}"
                raise DatasetError(msg)
            return
        topic_id = len(self._topics) + 1
        qos = LATCHED_QOS_PROFILE if latched else ""
        self._connection.execute(
            "INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) "
            "VALUES (?, ?, ?, 'cdr', ?)",
            (topic_id, name, message_type, qos),
        )
        self._topics[name] = _Topic(topic_id, name, message_type, qos)

    def write(self, topic: str, timestamp_ns: int, payload: bytes) -> None:
        """Append one serialized message."""

        if self._connection is None:
            raise DatasetError("the bag writer is closed")
        entry = self._topics.get(topic)
        if entry is None:
            msg = f"topic {topic} was not declared with add_topic"
            raise DatasetError(msg)
        self._message_id += 1
        self._connection.execute(
            "INSERT INTO messages (id, topic_id, timestamp, data) VALUES (?, ?, ?, ?)",
            (self._message_id, entry.topic_id, int(timestamp_ns), payload),
        )
        entry.count += 1
        self._first_ns = (
            timestamp_ns if self._first_ns is None else min(self._first_ns, timestamp_ns)
        )
        self._last_ns = timestamp_ns if self._last_ns is None else max(self._last_ns, timestamp_ns)

    def close(self) -> None:
        """Commit the database and write ``metadata.yaml`` (idempotent)."""

        if self._connection is None:
            return
        self._connection.commit()
        self._connection.close()
        self._connection = None
        start = self._first_ns or 0
        duration = (self._last_ns or 0) - start
        total = sum(topic.count for topic in self._topics.values())
        info: dict[str, Any] = {
            "version": 5,
            "storage_identifier": "sqlite3",
            "duration": {"nanoseconds": duration},
            "starting_time": {"nanoseconds_since_epoch": start},
            "message_count": total,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": topic.name,
                        "type": topic.message_type,
                        "serialization_format": "cdr",
                        "offered_qos_profiles": topic.qos,
                    },
                    "message_count": topic.count,
                }
                for topic in self._topics.values()
            ],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [self._db_name],
            "files": [
                {
                    "path": self._db_name,
                    "starting_time": {"nanoseconds_since_epoch": start},
                    "duration": {"nanoseconds": duration},
                    "message_count": total,
                }
            ],
        }
        if self._custom_data:
            info["custom_data"] = self._custom_data
        (self.directory / "metadata.yaml").write_text(
            yaml.safe_dump({"rosbag2_bagfile_information": info}, sort_keys=False),
            encoding="utf-8",
        )

    def __enter__(self) -> Rosbag2Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
