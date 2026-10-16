from __future__ import annotations

import datetime as dt
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from calibrex.core import io


def fingerprint(value: Any, seen: dict[int, int] | None = None) -> Any:
    """Compare types, float bytes, mapping order and collection alias topology."""
    if seen is None:
        seen = {}
    if isinstance(value, (dict, list, tuple, set)):
        if id(value) in seen:
            return ("alias", seen[id(value)])
        seen[id(value)] = len(seen)
        if isinstance(value, dict):
            return (
                "dict",
                [(fingerprint(k, seen), fingerprint(v, seen)) for k, v in value.items()],
            )
        values = [fingerprint(v, seen) for v in value]
        return (
            type(value).__name__,
            sorted(values, key=repr) if isinstance(value, set) else values,
        )
    if isinstance(value, float):
        return ("float64", struct.pack("!d", value))
    if isinstance(value, (dt.date, dt.datetime)):
        return (type(value).__name__, value.isoformat())
    return (type(value).__name__, value)


def test_all_tracked_yaml_mappings_preserve_values_and_float_bytes() -> None:
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(
        ["git", "ls-files", "*.yaml", "*.yml"],
        cwd=root,
        text=True,
    ).splitlines()
    assert names
    for name in names:
        path = root / name
        with path.open(encoding="utf-8") as stream:
            try:
                expected = yaml.safe_load(stream)
            except (yaml.YAMLError, UnicodeError) as error:
                with pytest.raises(type(error)):
                    io.read_mapping(path)
                continue
        if isinstance(expected, dict):
            assert fingerprint(io.read_mapping(path)) == fingerprint(expected), name
        else:
            with pytest.raises(ValueError, match="must contain a mapping at the top level"):
                io.read_mapping(path)


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".YAML"])
def test_compiled_mapping_preserves_safe_yaml_values_and_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    loader = getattr(yaml, "CSafeLoader", None)
    if loader is None:
        pytest.skip("LibYAML is optional")
    text = """defaults: &defaults
  enabled: yes
  null_value: ~
  floats: [-0.0, .inf, -.inf, .nan, 1.23e-12, 2:20.5]
  ints: [012, 0x10, 0b10, 1_024]
  timestamp: 2026-10-16T12:30:00+09:00
  date: 2026-10-16
  binary: !!binary SGVsbG8=
  unicode: "センサ 🚗"
copy: *defaults
merged:
  <<: *defaults
  enabled: no
repeated: old
repeated: new
block: >-
  camera
  lidar
recursive: &recursive {self: *recursive}
set: !!set {x: null, y: null}
"""
    expected = yaml.safe_load(text)
    calls = []
    original = yaml.load

    def observed(stream: Any, Loader: Any) -> Any:
        calls.append(Loader)
        return original(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "load", observed)
    path = tmp_path / ("mapping" + suffix)
    path.write_bytes(text.encode())
    actual = io.read_mapping(path)
    assert calls == [loader]
    assert fingerprint(actual) == fingerprint(expected)
    assert actual["copy"] is actual["defaults"]
    assert actual["recursive"]["self"] is actual["recursive"]


def test_python_fallback_when_libyaml_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    path = tmp_path / "mapping.yaml"
    path.write_text("a: [1, 2.0, yes]\n", encoding="utf-8")
    assert io.read_mapping(path) == {"a": [1, 2.0, True]}


@pytest.mark.parametrize("registration", ["constructor", "implicit", "path", "multi"])
def test_native_safe_loader_registrations_are_honored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registration: str,
) -> None:
    import re

    class CustomSafeLoader(yaml.SafeLoader):
        pass

    if registration == "constructor":
        CustomSafeLoader.add_constructor(
            "!sensor", lambda loader, node: loader.construct_scalar(node)
        )
        text = "value: !sensor camera\n"
    elif registration == "multi":
        CustomSafeLoader.add_multi_constructor(
            "!sensor:",
            lambda loader, tag, node: tag + loader.construct_scalar(node),
        )
        text = "value: !sensor:camera lidar\n"
    elif registration == "implicit":
        CustomSafeLoader.add_implicit_resolver(
            "tag:yaml.org,2002:str", re.compile("^sensor$"), ["s"]
        )
        text = "value: sensor\n"
    else:
        CustomSafeLoader.add_path_resolver("tag:yaml.org,2002:str", ["value"], str)
        text = "value: camera\n"
    monkeypatch.setattr(yaml, "SafeLoader", CustomSafeLoader)
    path = tmp_path / "mapping.yaml"
    path.write_text(text, encoding="utf-8")
    expected = yaml.safe_load(text)
    assert io.read_mapping(path) == expected


@pytest.mark.parametrize(
    "text",
    [
        'value: "\\uD800"\n',  # Python accepts this escape, LibYAML rejects the surrogate.
        "value: [1,\n",
        "value: !unknown x\n",
        "a: 1\n---\nb: 2\n",
        "value: !!python/object/apply:builtins.str [1]\n",
    ],
)
def test_parser_failures_retain_python_acceptance_and_diagnostics(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "mapping.yaml"
    path.write_bytes(text.encode())
    with path.open(encoding="utf-8") as stream:
        try:
            expected = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            with pytest.raises(type(error)) as actual:
                io.read_mapping(path)
            assert str(actual.value) == str(error)
            return
    assert fingerprint(io.read_mapping(path)) == fingerprint(expected)


@pytest.mark.parametrize("text", ["[]\n", "null\n", "1\n", "true\n"])
def test_yaml_top_level_mapping_requirement_is_retained(tmp_path: Path, text: str) -> None:
    path = tmp_path / "mapping.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping at the top level"):
        io.read_mapping(path)


def test_json_keeps_its_original_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(stream: Any) -> Any:
        raise AssertionError("JSON must not enter a YAML parser")

    monkeypatch.setattr(io, "_read_yaml", unexpected)
    path = tmp_path / "mapping.JSON"
    value = {"negative_zero": -0.0, "string": "no", "points": [[1.0, 2.0, 3.0]]}
    path.write_text(json.dumps(value), encoding="utf-8")
    assert fingerprint(io.read_mapping(path)) == fingerprint(value)
