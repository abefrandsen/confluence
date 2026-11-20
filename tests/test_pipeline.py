"""End-to-end execution tests for Confluence pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from confluence import OutputBindingError, Pipeline, StepExecutionError


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


def test_output_unpacking_mismatch_raises_step_execution_error() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["vals"],
            "steps": [
                {"name": "bad_unpack", "fn": "split_two", "args": ["$vals", 1], "outputs": ["a", "b"]},
            ],
        },
        functions={"split_two": split_two},
    )
    with pytest.raises(StepExecutionError) as exc_info:
        pipe.run({"vals": [1.0, 2.0]})

    assert isinstance(exc_info.value.cause, OutputBindingError)
    assert "bad_unpack" in str(exc_info.value)


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
