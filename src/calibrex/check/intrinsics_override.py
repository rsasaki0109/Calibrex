"""Camera intrinsics given on the command line (``--intrinsics``), by image topic.

A Kalibr camchain passed with ``--tf`` is matched to a camera by its frame name (the
entry's ``rostopic`` or its key), which on most bags is not the camera's header frame, so
it is quietly passed over. ``--intrinsics`` names the image topic instead::

    --intrinsics camchain.yaml                 # one camera in the bag, or entries with rostopic
    --intrinsics /cam/image_raw=camchain.yaml  # cam0 of the file for this topic
    --intrinsics /right/image_raw=stereo.yaml:cam1

It reads what ``calibrex intrinsics guide`` and ``calibrex intrinsics rotate`` write, and
any other Kalibr camchain. Only the intrinsics are taken; the extrinsics still come from
the bag's ``/tf_static`` or ``--tf``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_KEY = re.compile(r"^(?P<path>.+):(?P<key>cam\d+)$")


@dataclass(frozen=True)
class IntrinsicsOverride:
    """One ``--intrinsics [TOPIC=]CAMCHAIN[:camN]``."""

    path: Path
    topic: str | None = None
    key: str | None = None

    @classmethod
    def parse(cls, text: str) -> IntrinsicsOverride:
        topic: str | None = None
        rest = text
        if "=" in text and text.split("=", 1)[0].startswith("/"):
            topic, rest = text.split("=", 1)
        key: str | None = None
        match = _KEY.match(rest)
        if match is not None and not Path(rest).exists():
            rest, key = match.group("path"), match.group("key")
        if not rest:
            raise ValueError(f"--intrinsics {text!r}: no camchain file")
        return cls(Path(rest), topic, key)

    def entries(self) -> list[tuple[str, dict[str, Any]]]:
        """``(key, entry)`` of the camchain's cameras, or only ``key``'s."""

        from calibrex.core.io import read_mapping

        chain = read_mapping(self.path)
        cameras = [
            (str(name), entry)
            for name, entry in chain.items()
            if isinstance(entry, dict) and "intrinsics" in entry
        ]
        if not cameras:
            raise ValueError(f"{self.path} holds no camera with intrinsics")
        if self.key is not None:
            chosen = [(name, entry) for name, entry in cameras if name == self.key]
            if not chosen:
                raise ValueError(f"{self.path} has no camera {self.key!r}")
            return chosen
        return cameras

    def describe(self, key: str) -> str:
        return f"{self.path.name} {key} (--intrinsics)"


def match_intrinsics(
    overrides: tuple[IntrinsicsOverride, ...],
    image_topic: str,
    camera_topics: tuple[str, ...],
) -> tuple[IntrinsicsOverride, str, dict[str, Any]] | None:
    """The override and camchain entry for ``image_topic``, or ``None``.

    In order: an override naming the topic (its ``:camN``, else its first camera); an entry
    whose ``rostopic`` is the topic; and, when the bag has one camera, the only override's
    first camera (or its ``:camN``).
    """

    for override in overrides:
        if override.topic == image_topic:
            key, entry = override.entries()[0]
            return override, key, entry
    for override in overrides:
        if override.topic is not None:
            continue
        for key, entry in override.entries():
            if entry.get("rostopic") == image_topic:
                return override, key, entry
    untargeted = [override for override in overrides if override.topic is None]
    if len(untargeted) == 1 and len(set(camera_topics)) <= 1:
        override = untargeted[0]
        cameras = override.entries()
        if override.key is not None or len(cameras) == 1:
            key, entry = cameras[0]
            return override, key, entry
    return None
