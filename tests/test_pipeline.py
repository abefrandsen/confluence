"""End-to-end execution tests for Confluence pipelines."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from confluence import (
    OutputBindingError,
    PathSyntaxError,
    Pipeline,
    ValidationError,
)


class LinearScaler:
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def transform(self, values: list[float]) -> list[float]:
        return [v * self.factor for v in values]

    def __call__(self, x: float) -> float:
        return x * self.factor


def split_two(values: list[float], split_idx: int) -> tuple[list[float], list[float], int]:
    return values[:split_idx], values[split_idx:], len(values)


def test_yaml_end_to_end_with_overrides_and_interpolation(tmp_path: Path) -> None:
    yaml_content = """
imports:
  - math

params:
  factor: 2.0
  split_idx: 2

inputs: [raw]

steps:
  - name: create_scaler
    fn: LinearScaler
    kwargs:
      factor: "${params.factor}"
    outputs: [scaler]

  - name: scale
    fn: scaler.transform
    args: [$raw]
    outputs: [scaled]

  - name: call_instance_directly
    fn: scaler
    args: ["$scaled[0]"]
    outputs: [first_doubled_again]

  - name: split
    fn: split_two
    kwargs:
      values: $scaled
      split_idx: "${params.split_idx}"
    outputs: [head, tail, "_"]

  - name: sqrt_first
    fn: math.sqrt
    args: ["$head[0]"]
    outputs: [root]
"""
    cfg_path = tmp_path / "pipe.yaml"
    cfg_path.write_text(yaml_content)

    pipe = Pipeline.from_yaml(
        cfg_path,
        functions={"LinearScaler": LinearScaler, "split_two": split_two},
        overrides=["params.factor=4.0"],
    )

    initial_data = {"raw": [1.0, 2.0, 3.0]}
    result = pipe.run(initial_data)

    # Caller's dict was not mutated
    assert list(initial_data.keys()) == ["raw"]

    # Results are accurate and "_" was discarded
    assert result["scaled"] == [4.0, 8.0, 12.0]
    assert result["first_doubled_again"] == 16.0
    assert result["head"] == [4.0, 8.0]
    assert result["tail"] == [12.0]
    assert "_" not in result
    assert result["root"] == 2.0

    # Imports / functions / builtins did not leak into returned data dict
    assert "math" not in result
    assert "LinearScaler" not in result
    assert "len" not in result


def test_single_output_does_not_unpack_tuple() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["vals"],
            "steps": [
                {"fn": "split_two", "args": ["$vals", 1], "outputs": ["whole_tuple"]},
            ],
        },
        functions={"split_two": split_two},
    )
    res = pipe.run({"vals": [10.0, 20.0]})
    assert res["whole_tuple"] == ([10.0], [20.0], 2)


def test_overwrite_and_side_effect_step() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["items"],
            "steps": [
                {"fn": "items.append", "args": [4]},  # outputs omitted -> side effect
                {"fn": "sum", "args": ["$items"], "outputs": ["items"]},  # overwrite
            ],
        }
    )
    res = pipe.run({"items": [1, 2, 3]})
    assert res["items"] == 10


def test_output_unpacking_mismatch_raises_output_binding_error_with_note() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["vals"],
            "steps": [
                {"name": "bad_unpack", "fn": "split_two", "args": ["$vals", 1], "outputs": ["a", "b"]},
            ],
        },
        functions={"split_two": split_two},
    )
    with pytest.raises(OutputBindingError) as exc_info:
        pipe.run({"vals": [1.0, 2.0]})

    assert any("bad_unpack" in note for note in exc_info.value.__notes__)


def test_sub_pipeline_composition_via_callable() -> None:
    sub = Pipeline.from_dict(
        {
            "inputs": ["x"],
            "steps": [
                {"fn": "sum", "args": ["$x"], "outputs": ["total"]},
                {"fn": "len", "args": ["$x"], "outputs": ["count"]},
            ],
        }
    )

    outer = Pipeline.from_dict(
        {
            "inputs": ["nums"],
            "steps": [
                {"fn": "sub_pipe", "args": [{"x": "$nums"}], "outputs": ["stats"]},
                {"fn": "div", "args": ["$stats.total", "$stats.count"], "outputs": ["mean"]},
            ],
        },
        functions={"sub_pipe": sub, "div": lambda a, b: a / b},
    )

    res = outer({"nums": [2, 4, 6]})
    assert res["mean"] == 4.0


def test_custom_omegaconf_resolver_and_higher_order_builtin() -> None:
    OmegaConf.register_resolver("add_one", lambda x: int(x) + 1, replace=True)
    pipe = Pipeline.from_dict(
        {
            "params": {"n": "${add_one:3}"},
            "steps": [
                {"fn": "range", "args": ["${params.n}"], "outputs": ["r"]},
                {"fn": "map", "args": ["$str", "$r"], "outputs": ["mapped"]},
                {"fn": "list", "args": ["$mapped"], "outputs": ["strs"]},
            ],
        }
    )
    res = pipe.run()
    assert res["strs"] == ["0", "1", "2", "3"]


def test_single_discard_output() -> None:
    pipe = Pipeline.from_dict(
        {
            "steps": [
                {"fn": "abs", "args": [-5], "outputs": ["_"]},
            ],
        }
    )
    res = pipe.run()
    assert res == {}


def test_submodule_import_binds_top_level_package() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["urllib.parse"],
            "steps": [
                {
                    "fn": "urllib.parse.quote",
                    "args": ["hello world"],
                    "outputs": ["quoted"],
                }
            ],
        }
    )
    res = pipe.run()
    assert res["quoted"] == "hello%20world"


def test_run_validate_false_skips_validation_and_loads_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "inputs": ["x"],
            "steps": [
                {"fn": "math.sqrt", "args": ["$x"], "outputs": ["root"]},
            ],
        }
    )

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("validate() should not be called when validate=False")

    monkeypatch.setattr(pipe, "validate", _fail_if_called)

    # Works even when validate() was never called beforehand (loads imports lazily once)
    assert pipe.run({"x": 9.0}, validate=False)["root"] == 3.0
    assert pipe({"x": 16.0}, validate=False)["root"] == 4.0


def test_run_validate_false_raises_on_bad_import() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["definitely_not_a_real_module_xyz"],
            "steps": [{"fn": "abs", "args": [-1], "outputs": ["y"]}],
        }
    )
    with pytest.raises(ValidationError, match="definitely_not_a_real_module_xyz"):
        pipe.run(validate=False)


def test_run_validate_false_raises_path_syntax_error_with_note() -> None:
    pipe_bad_fn = Pipeline.from_dict({"steps": [{"fn": "a..b"}]})
    with pytest.raises(PathSyntaxError) as exc_info:
        pipe_bad_fn.run(validate=False)
    assert any("Step 0" in note and "a..b" in note for note in exc_info.value.__notes__)

    pipe_bad_ref = Pipeline.from_dict({"steps": [{"fn": "abs", "args": ["$a..b"]}]})
    with pytest.raises(PathSyntaxError) as exc_info2:
        pipe_bad_ref.run(validate=False)
    assert any("Step 0" in note and "abs" in note for note in exc_info2.value.__notes__)


def test_paths_are_preparsed_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import confluence.path as path_mod
    import confluence.pipeline as pipe_mod

    pipe = Pipeline.from_dict(
        {
            "inputs": ["x"],
            "steps": [
                {"fn": "abs", "args": ["$x"], "outputs": ["y"]},
            ],
        }
    )

    def _forbid_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("parse_path() should not be called after Pipeline.__init__")

    monkeypatch.setattr(path_mod, "parse_path", _forbid_parse)
    monkeypatch.setattr(pipe_mod, "parse_path", _forbid_parse)

    pipe.validate()
    assert pipe.run({"x": -5})["y"] == 5
    assert pipe.run({"x": -7}, validate=False)["y"] == 7


def test_compiled_literal_containers_are_fresh_per_run() -> None:
    def append_one(items: list[int]) -> list[int]:
        items.append(1)
        return items

    pipe = Pipeline.from_dict(
        {
            "steps": [
                {"fn": "append_one", "args": [[0]], "outputs": ["out"]},
            ],
        },
        functions={"append_one": append_one},
    )
    assert pipe.run(validate=False)["out"] == [0, 1]
    assert pipe.run(validate=False)["out"] == [0, 1]


def test_pipeline_with_imports_supports_deepcopy_and_pickle() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "inputs": ["x"],
            "steps": [
                {"fn": "math.sqrt", "args": ["$x"], "outputs": ["root"]},
            ],
        }
    )
    # Trigger import loading before copying/pickling
    assert pipe.run({"x": 9.0})["root"] == 3.0

    copied = copy.deepcopy(pipe)
    assert copied.run({"x": 16.0}, validate=False)["root"] == 4.0
    assert copied.run({"x": 25.0})["root"] == 5.0

    unpickled = pickle.loads(pickle.dumps(pipe))
    assert unpickled.run({"x": 36.0}, validate=False)["root"] == 6.0
    assert unpickled.run({"x": 49.0})["root"] == 7.0


def test_multi_output_unpacking_rejects_mapping_and_sub_pipeline() -> None:
    sub = Pipeline.from_dict(
        {
            "inputs": ["a"],
            "steps": [{"fn": "abs", "args": ["$a"], "outputs": ["n"]}],
        }
    )
    outer = Pipeline.from_dict(
        {
            "inputs": ["x"],
            "steps": [
                {"fn": "sub", "args": [{"a": "$x"}], "outputs": ["k1", "k2"]},
            ],
        },
        functions={"sub": sub},
    )
    with pytest.raises(OutputBindingError) as exc_info:
        outer.run({"x": -3})
    assert "mapping" in str(exc_info.value).lower()
    assert any("Step 0" in note for note in exc_info.value.__notes__)


def test_step_exception_preserves_type_and_adds_note() -> None:
    class FakeOOMError(RuntimeError):
        pass

    def boom() -> None:
        raise FakeOOMError("out of memory")

    pipe = Pipeline.from_dict(
        {"steps": [{"name": "forward_block", "fn": "boom"}]},
        functions={"boom": boom},
    )
    with pytest.raises(FakeOOMError, match="out of memory") as exc_info:
        pipe.run(validate=False)

    assert any(
        "Step 0 (forward_block)" in note and "boom" in note
        for note in exc_info.value.__notes__
    )
