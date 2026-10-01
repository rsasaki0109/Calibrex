"""Static frame tree of candidate extrinsics, built on :class:`FrameGraph`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from calibrex.core.calibration_check import CandidateSourceKind
from calibrex.core.exceptions import FrameGraphError
from calibrex.core.frames import FrameGraph, FrameNode
from calibrex.core.geometry import SE3


def normalize_frame_id(frame_id: str) -> str:
    """Strip the leading ``/`` that some drivers put on tf frame ids."""

    return frame_id.strip().lstrip("/")


@dataclass(frozen=True)
class StaticEdge:
    """A candidate static transform ``T_parent_child``."""

    parent: str
    child: str
    transform: SE3
    source: CandidateSourceKind


@dataclass(frozen=True)
class FrameHints:
    """Optional topic-to-frame and role-to-frame hints shipped with a source."""

    topic_frames: dict[str, str] = field(default_factory=dict)
    role_frames: dict[str, str] = field(default_factory=dict)


class StaticFrameTree:
    """Forest of static transforms with one parent per child frame.

    Every connected component is validated by a :class:`FrameGraph` (single
    root, no cycles); lookups compose ``T_root_frame`` through it.
    """

    def __init__(self, edges: Iterable[StaticEdge]) -> None:
        by_child: dict[str, StaticEdge] = {}
        for edge in edges:
            if edge.parent == edge.child:
                msg = f"frame '{edge.child}' cannot be its own parent"
                raise FrameGraphError(msg)
            existing = by_child.get(edge.child)
            if existing is not None and existing.parent != edge.parent:
                msg = (
                    f"frame '{edge.child}' has conflicting parents "
                    f"'{existing.parent}' and '{edge.parent}'"
                )
                raise FrameGraphError(msg)
            by_child[edge.child] = edge
        self._edges = by_child
        self._graphs: dict[str, FrameGraph] = {}
        self._root_of: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        names: set[str] = set(self._edges)
        names.update(edge.parent for edge in self._edges.values())
        components: dict[str, dict[str, FrameNode]] = {}
        for name in sorted(names):
            root = self._find_root(name)
            self._root_of[name] = root
            components.setdefault(root, {})
        for name in sorted(names):
            edge = self._edges.get(name)
            components[self._root_of[name]][name] = FrameNode(
                name=name,
                parent=None if edge is None else edge.parent,
                root=edge is None,
                transform_to_parent=SE3.identity() if edge is None else edge.transform,
                estimate=False,
            )
        for root, nodes in components.items():
            self._graphs[root] = FrameGraph(nodes)

    def _find_root(self, frame: str) -> str:
        seen: list[str] = []
        current = frame
        while current in self._edges:
            if current in seen:
                cycle = " -> ".join([*seen[seen.index(current) :], current])
                msg = f"cycle in static frame tree: {cycle}"
                raise FrameGraphError(msg)
            seen.append(current)
            current = self._edges[current].parent
        return current

    @property
    def frames(self) -> list[str]:
        """All frame names, sorted."""

        return sorted(self._root_of)

    @property
    def roots(self) -> list[str]:
        """One root per connected component, sorted."""

        return sorted(self._graphs)

    @property
    def edges(self) -> list[StaticEdge]:
        """Edges sorted by child name."""

        return [self._edges[child] for child in sorted(self._edges)]

    def __contains__(self, frame: object) -> bool:
        return frame in self._root_of

    def connected(self, first: str, second: str) -> bool:
        """Return whether both frames exist and share a root."""

        return (
            first in self._root_of
            and second in self._root_of
            and self._root_of[first] == self._root_of[second]
        )

    def lookup(self, parent: str, child: str) -> SE3 | None:
        """Return ``T_parent_child``, or ``None`` when the frames are not connected."""

        if not self.connected(parent, child):
            return None
        graph = self._graphs[self._root_of[parent]]
        return graph.transform_to_root(parent).inverse().compose(graph.transform_to_root(child))
