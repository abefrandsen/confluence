"""Unit tests for path parsing, traversal, and value resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from confluence.errors import PathSyntaxError, ResolutionError
from confluence.path import (
    AttrSegment,
    CompiledRef,
    IndexSegment,
    InvalidRef,
    compile_value,
    parse_path,
    resolve_compiled_value,
    resolve_path,
    resolve_value,
)


@dataclass
class DummyObj:
    value: int
    items: list[dict[str, object]]

    def double(self) -> int:
        return self.value * 2


def test_parse_simple_and_chained_paths() -> None:
    p = parse_path("x")
    assert p.root == "x"
    assert p.segments == ()

    p2 = parse_path('prep.batches[0].values[-1]["my.key"][\'other key\']')
    assert p2.root == "prep"
    assert p2.segments == (
        AttrSegment("batches"),
        IndexSegment(0),
        AttrSegment("values"),
        IndexSegment(-1),
        IndexSegment("my.key"),
        IndexSegment("other key"),
    )


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "123abc",
        "a.",
        "a..b",
        "a[]",
        "a[0:5]",
        "a[b]",
        "a[0",
        "a.b-c",
    ],
)
def test_parse_invalid_paths(bad_path: str) -> None:
    with pytest.raises(PathSyntaxError):
        parse_path(bad_path)


def test_mapping_key_first_then_attribute() -> None:
    data = {"prep": {"values": [10, 20, 30], "x_train": [1, 2]}}
    # Mapping key "values" wins over dict.values method
    assert resolve_path("prep.values", [data]) == [10, 20, 30]
    assert resolve_path("prep.values[-1]", [data]) == 30
    # dict.get is not a key in prep, so attribute lookup still works
    assert resolve_path("prep.get", [data]) == data["prep"].get


def test_object_attribute_and_indexing() -> None:
    obj = DummyObj(value=7, items=[{"my.key": 99}])
    scopes = [{"obj": obj}]
    assert resolve_path("obj.value", scopes) == 7
    assert resolve_path('obj.items[0]["my.key"]', scopes) == 99
    assert resolve_path("obj.double", scopes)() == 14


def test_scope_precedence() -> None:
    data = {"x": "from_data"}
    functions = {"x": "from_fn", "y": "from_fn"}
    builtins_scope = {"y": "from_builtin", "z": "from_builtin"}
    scopes = [data, functions, builtins_scope]
    assert resolve_path("x", scopes) == "from_data"
    assert resolve_path("y", scopes) == "from_fn"
    assert resolve_path("z", scopes) == "from_builtin"


def test_resolution_errors() -> None:
    scopes = [{"d": {"a": [1, 2]}}]
    with pytest.raises(ResolutionError, match="not defined"):
        resolve_path("missing", scopes)
    with pytest.raises(ResolutionError, match="Cannot resolve segment"):
        resolve_path("d.nope", scopes)
    with pytest.raises(ResolutionError, match="Cannot index"):
        resolve_path("d.a[5]", scopes)


def test_resolve_value_literals_refs_escaping_and_containers() -> None:
    scopes = [{"x": 10, "y": {"z": 20}}]
    assert resolve_value(42, scopes) == 42
    assert resolve_value("plain", scopes) == "plain"
    assert resolve_value("$x", scopes) == 10
    assert resolve_value("$$not_a_ref", scopes) == "$not_a_ref"
    assert resolve_value(["$x", 1, {"k": "$y.z", "esc": "$$x"}], scopes) == [
        10,
        1,
        {"k": 20, "esc": "$x"},
    ]


def test_compile_and_resolve_compiled_value() -> None:
    refs: list[CompiledRef | InvalidRef] = []
    compiled = compile_value(["$x", 1, {"k": "$y.z", "esc": "$$x", "bad": "$a..b"}], refs)

    assert isinstance(compiled[0], CompiledRef)
    assert compiled[2]["esc"] == "$x"
    assert isinstance(compiled[2]["bad"], InvalidRef)
    assert len(refs) == 3
    assert isinstance(refs[0], CompiledRef)
    assert isinstance(refs[1], CompiledRef)
    assert isinstance(refs[2], InvalidRef)

    valid_compiled = compile_value(["$x", 1, {"k": "$y.z", "esc": "$$x"}])
    scopes = [{"x": 10, "y": {"z": 20}}]
    assert resolve_compiled_value(valid_compiled, scopes) == [10, 1, {"k": 20, "esc": "$x"}]

    with pytest.raises(PathSyntaxError):
        resolve_compiled_value(compiled, scopes)
