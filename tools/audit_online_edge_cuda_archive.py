"""Audit complete online-edge evidence without running archived Python or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_online_edge_cuda as benchmark  # noqa: E402


def _local_references(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                raise ValueError("archive schemas must use local references")
            _local_references(child)
    elif isinstance(value, list):
        for child in value:
            _local_references(child)


def audit_archive(path: Path, temporary_root: Path, *, bind_commit: str | None = None) -> dict:
    """Verify the index, all complete artifacts, and an optional later Git binding."""
    import jsonschema

    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("evidence archive is too large")
    temporary_root = temporary_root.resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="online-edge-audit-", dir=temporary_root) as name:
        folder = Path(name).resolve()
        if not folder.is_relative_to(temporary_root):
            raise ValueError("audit temporary directory escapes its intended root")
        seen: set[str] = set()
        total = 0
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member.name
                    or ":" in member.name
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe evidence member")
                target = (folder / member.name).resolve()
                if not target.is_relative_to(folder):
                    raise ValueError("evidence member escapes audit root")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.name in seen:
                    raise ValueError("duplicate evidence member")
                seen.add(member.name)
                total += member.size
                if len(seen) > 2048 or total > 256 * 1024 * 1024:
                    raise ValueError("unpacked evidence exceeds limits")
                stream = archive.extractfile(member)
                assert stream is not None
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.read())
        index = benchmark.read(folder / "index.json")
        if seen != set(index["files"]) | {"index.json"}:
            raise ValueError("evidence index file set differs")
        for relative, info in index["files"].items():
            target = folder / relative
            if target.stat().st_size != info["size"] or benchmark.sha(target) != info["sha256"]:
                raise ValueError(f"evidence digest mismatch: {relative}")
        for schema in (folder / "schemas").glob("*.json"):
            _local_references(benchmark.read(schema))
        result = benchmark.audit(folder)
        protocol = benchmark.read(folder / "protocol.json")
        if index["provenance"]["protocol_sha256"] != benchmark.sha(folder / "protocol.json"):
            raise ValueError("evidence index lineage differs")
        if bind_commit is not None:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--verify", "--end-of-options", bind_commit + "^{commit}"],
                text=True,
            ).strip()
            expected = benchmark._git_files(commit, ("src/calibrex", *benchmark.TOOLCHAIN))
            actual = benchmark._verify_sources(folder / "source")
            from_git = {
                name.removeprefix("src/"): hashlib.sha256(data).hexdigest()
                for name, data in expected.items()
                if name.startswith("src/calibrex/") and name.endswith(".py")
            }
            if actual != from_git or any(
                hashlib.sha256(expected[name]).hexdigest() != digest
                for name, digest in protocol["toolchain_sha256"].items()
            ):
                raise ValueError("later Git content differs from measured bytes")
            result["provenance"]["post_run_git_content_binding"] = commit
        result["provenance"].update(
            archive_sha256=benchmark.sha(path),
            index_sha256=benchmark.sha(folder / "index.json"),
            auditor_sha256=benchmark.sha(Path(__file__)),
            indexed_files=len(index["files"]),
        )
        jsonschema.validate(result, benchmark.read(folder / "schemas/records.schema.json"))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--temporary-root", type=Path, default=Path("outputs/online-edge-audits"))
    parser.add_argument("--bind-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit_archive(args.archive, args.temporary_root, bind_commit=args.bind_commit)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if value["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
