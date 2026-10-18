"""Safe YAML emission semantics, custom hooks and atomic failure behavior."""

import io as streams
import json
import math
import struct
from datetime import date

import pytest
import yaml

from calibrex.core import io

WRITERS = (io.write_mapping, io.write_mapping_atomic)


@pytest.mark.parametrize("writer", WRITERS)
def test_complete_yaml_values_float_bits_order_and_aliases(writer, tmp_path):
    shared = [1.0, -0.0, 1e-300, 1e300, None, True, False]
    data = {
        "strings": ["true", "no", "on", "~", ".inf", "1e3", "0x10", "2026-10-17",
                    "calibration: result", "first\nsecond\n", "日本語 😺", "\0\x1f"],
        "numbers": shared, "alias": shared,
        "nested": {"b": [3, {"a": -0.0}], "a": {}},
        "nonfinite": [float("nan"), float("inf"), -float("inf")],
    }
    path = tmp_path / "result.yaml"
    writer(path, data)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(loaded) == list(data)
    assert list(loaded["nested"]) == ["b", "a"]
    assert loaded["strings"] == data["strings"]
    assert loaded["alias"] is loaded["numbers"]
    assert struct.pack("<4d", *loaded["numbers"][:4]) == struct.pack("<4d", *shared[:4])
    assert math.isnan(loaded["nonfinite"][0])
    assert loaded["nonfinite"][1:] == data["nonfinite"][1:]
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("writer", WRITERS)
def test_json_emission_bytes_are_unchanged(writer, tmp_path):
    data = {"z": -0.0, "a": {"c": "日本語", "b": [1, 2.0, True, None]}}
    path = tmp_path / "result.json"
    writer(path, data)
    assert path.read_bytes() == (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


@pytest.mark.skipif(not getattr(yaml, "CSafeDumper", None), reason="optional LibYAML")
@pytest.mark.parametrize("writer", WRITERS)
def test_native_emission_is_used_for_plain_data(writer, tmp_path, monkeypatch):
    calls = []
    original = io._C_DUMP

    def native(*args, **kwargs):
        calls.append(kwargs["Dumper"])
        return original(*args, **kwargs)

    monkeypatch.setattr(io, "_C_DUMP", native)
    writer(tmp_path / "result.yaml", {"values": [0.0, 1.0, 2.0]})
    assert calls == [yaml.CSafeDumper]


@pytest.mark.parametrize("writer", WRITERS)
def test_missing_libyaml_keeps_python_emission(writer, tmp_path, monkeypatch):
    monkeypatch.setattr(yaml, "CSafeDumper", None, raising=False)
    data = {"value": "line\nnext\n"}
    path = tmp_path / "result.yaml"
    writer(path, data)
    assert path.read_text(encoding="utf-8") == yaml.safe_dump(
        data, sort_keys=False, allow_unicode=False
    )


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("value", ["\ud800", date(2026, 10, 17)])
def test_python_only_values_never_reach_native_emission(writer, value, tmp_path, monkeypatch):
    def native(*args, **kwargs):
        raise AssertionError("unsupported value reached C emitter")

    monkeypatch.setattr(io, "_C_DUMP", native)
    path = tmp_path / "result.yaml"
    writer(path, {"value": value})
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"value": value}


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("in_place", [False, True])
def test_python_representer_hook_runs_once_with_python_dumper(
    writer, in_place, tmp_path, monkeypatch
):
    calls = []
    original = yaml.SafeDumper.yaml_representers[str]

    def represent(dumper, value):
        assert type(dumper) is yaml.SafeDumper
        calls.append(value)
        return original(dumper, value)

    if in_place:
        monkeypatch.setitem(yaml.SafeDumper.yaml_representers, str, represent)
    else:
        monkeypatch.setattr(yaml.SafeDumper, "yaml_representers", {
            **yaml.SafeDumper.yaml_representers, str: represent,
        })
    writer(tmp_path / "result.yaml", {"key": "value"})
    assert calls == ["key", "value"]


@pytest.mark.parametrize("writer", WRITERS)
def test_custom_alias_policy_keeps_python_dumper(writer, tmp_path, monkeypatch):
    monkeypatch.setattr(yaml.SafeDumper, "ignore_aliases", lambda self, value: True)
    def native(*args, **kwargs):
        raise AssertionError("custom alias policy reached C emitter")
    monkeypatch.setattr(io, "_C_DUMP", native)
    shared = [1.0]
    path = tmp_path / "result.yaml"
    writer(path, {"a": shared, "b": shared})
    assert "&id" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("writer", WRITERS)
def test_cycles_retain_alias_identity(writer, tmp_path):
    cycle = []
    cycle.append(cycle)
    path = tmp_path / "result.yaml"
    writer(path, {"cycle": cycle})
    actual = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert actual["cycle"][0] is actual["cycle"]


@pytest.mark.skipif(not getattr(yaml, "CSafeDumper", None), reason="optional LibYAML")
def test_emitter_failure_falls_back_before_writing_a_prefix(monkeypatch):
    def native(*args, **kwargs):
        raise yaml.YAMLError("fixture native emitter failure")
    monkeypatch.setattr(io, "_C_DUMP", native)
    stream = streams.StringIO()
    io._write_yaml({"result": 0.5}, stream)
    assert stream.getvalue() == yaml.safe_dump({"result": 0.5}, sort_keys=False)


def test_atomic_failure_preserves_existing_result_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "result.yaml"
    path.write_bytes(b"previous: result\n")

    def failure(data, stream, **kwargs):
        stream.write("partial: ")
        raise RuntimeError("fixture write failure")

    monkeypatch.setattr(yaml, "safe_dump", failure)
    with pytest.raises(RuntimeError, match="fixture write failure"):
        io.write_mapping_atomic(path, {"new": "result"})
    assert path.read_bytes() == b"previous: result\n"
    assert list(tmp_path.iterdir()) == [path]
