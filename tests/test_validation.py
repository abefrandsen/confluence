"""Tests for pre-flight static validation and OmegaConf schema checking."""

from __future__ import annotations

import pytest
from omegaconf.errors import ConfigKeyError

from confluence import Pipeline, ValidationError


def test_schema_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ConfigKeyError):
        Pipeline.from_dict({"unknown_key": 123, "steps": []})


def test_validate_reports_missing_declared_inputs() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["raw_x", "raw_y"],
            "steps": [{"fn": "len", "args": ["$raw_x"], "outputs": ["n"]}],
        }
    )
    # Offline validation succeeds because `inputs` declares raw_x & raw_y
    pipe.validate()

    # Providing incomplete runtime keys fails with clear error
    with pytest.raises(ValidationError, match="Missing required pipeline inputs"):
        pipe.validate(input_keys=["raw_x"])


def test_validate_catches_undefined_refs_and_bad_imports() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["definitely_not_a_real_module_xyz"],
            "steps": [
                {"name": "bad_step", "fn": "missing_fn", "args": ["$missing_var"]},
            ],
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        pipe.validate()

    msg = str(exc_info.value)
    assert "definitely_not_a_real_module_xyz" in msg
    assert "missing_fn" in msg
    assert "missing_var" in msg
    assert "bad_step" in msg


def test_validate_catches_non_callable_static_fn() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "steps": [{"fn": "math.pi", "outputs": ["x"]}],
        }
    )
    with pytest.raises(ValidationError, match="is not callable"):
        pipe.validate()


def test_validate_tracks_dataflow_across_steps() -> None:
    pipe = Pipeline.from_dict(
        {
            "inputs": ["nums"],
            "steps": [
                {"fn": "sum", "args": ["$nums"], "outputs": ["total"]},
                {"fn": "abs", "args": ["$total"], "outputs": ["abs_total"]},
            ],
        }
    )
    pipe.validate()


def test_validate_rejects_excluded_builtins() -> None:
    pipe = Pipeline.from_dict(
        {
            "steps": [
                {"fn": "eval", "args": ["1 + 1"], "outputs": ["res"]},
            ],
        }
    )
    with pytest.raises(ValidationError, match="undefined name 'eval'"):
        pipe.validate()


def test_validate_warns_on_shadowing_static_scopes_and_deduplicates() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "steps": [
                {"fn": "abs", "args": [-1], "outputs": ["len"]},
                {"fn": "abs", "args": [-2], "outputs": ["len"]},  # duplicate should not warn twice
                {"fn": "abs", "args": [-3], "outputs": ["math"]},
                {"fn": "abs", "args": [-4], "outputs": ["my_fn"]},
            ],
        },
        functions={"my_fn": lambda x: x},
    )
    with pytest.warns(UserWarning) as record:
        pipe.validate(input_keys=["sum"])  # undeclared runtime input shadowing builtin

    messages = [str(w.message) for w in record]
    assert len([m for m in messages if "'len' shadows" in m]) == 1
    assert any("'sum' shadows a name in builtins" in m for m in messages)
    assert any("'math' shadows a name in imports" in m for m in messages)
    assert any("'my_fn' shadows a name in functions" in m for m in messages)


def test_validate_hints_only_for_real_missing_submodules() -> None:
    # Real submodule that isn't imported when only `xml` is imported
    pipe_real = Pipeline.from_dict(
        {
            "imports": ["xml"],
            "steps": [{"fn": "xml.dom.getDOMImplementation", "args": []}],
        }
    )
    with pytest.raises(
        ValidationError,
        match="did you mean to add 'xml.dom' to imports",
    ):
        pipe_real.validate()

    # Typo on a module should NOT trigger the submodule hint
    pipe_typo = Pipeline.from_dict(
        {
            "imports": ["math"],
            "steps": [{"fn": "math.sqroot", "args": [4]}],
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        pipe_typo.validate()
    assert "did you mean to add" not in str(exc_info.value)


def test_validate_keeps_successful_imports_when_another_import_fails() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math", "definitely_not_a_real_module_xyz"],
            "steps": [{"fn": "math.sqrt", "args": [4], "outputs": ["r"]}],
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        pipe.validate()

    assert len(exc_info.value.errors) == 1
    assert "definitely_not_a_real_module_xyz" in exc_info.value.errors[0]


def test_validate_respects_data_precedence_over_static_scope() -> None:
    # Step 0 binds `math` in data (shadowing the imported `math` module with a custom callable);
    # Step 1 calls `math.custom_method`, which must NOT fail static validation against the module.
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "inputs": ["obj"],
            "steps": [
                {"fn": "identity", "args": ["$obj"], "outputs": ["math"]},
                {"fn": "math.custom_method", "outputs": ["res"]},
            ],
        },
        functions={"identity": lambda x: x},
    )
    with pytest.warns(UserWarning, match="'math' shadows a name in imports"):
        pipe.validate()


@pytest.mark.parametrize("bad_name", ["my-key", "123abc", "a.b", "has space", ""])
def test_validate_rejects_non_identifier_inputs_and_outputs(bad_name: str) -> None:
    pipe_in = Pipeline.from_dict(
        {
            "inputs": [bad_name],
            "steps": [{"fn": "abs", "args": [-1], "outputs": ["ok"]}],
        }
    )
    with pytest.raises(ValidationError, match="invalid input name"):
        pipe_in.validate()

    pipe_out = Pipeline.from_dict(
        {
            "steps": [{"fn": "abs", "args": [-1], "outputs": [bad_name]}],
        }
    )
    with pytest.raises(ValidationError, match="invalid output binding name"):
        pipe_out.validate()


def test_validate_rejects_duplicate_inputs_and_step_outputs_but_allows_multiple_discards() -> None:
    pipe_dup_in = Pipeline.from_dict(
        {
            "inputs": ["x", "x"],
            "steps": [{"fn": "abs", "args": ["$x"], "outputs": ["y"]}],
        }
    )
    with pytest.raises(ValidationError, match="duplicate input name 'x'"):
        pipe_dup_in.validate()

    pipe_dup_out = Pipeline.from_dict(
        {
            "steps": [{"fn": "range", "args": [2], "outputs": ["a", "a"]}],
        }
    )
    with pytest.raises(ValidationError, match="duplicate output binding name 'a'"):
        pipe_dup_out.validate()

    # Multiple "_" discards in the same step remain valid
    pipe_multi_discard = Pipeline.from_dict(
        {
            "steps": [{"fn": "range", "args": [3], "outputs": ["_", "_", "last"]}],
        }
    )
    pipe_multi_discard.validate()


def test_validate_resolves_full_static_argument_refs() -> None:
    pipe = Pipeline.from_dict(
        {
            "imports": ["math"],
            "steps": [
                {"fn": "list", "args": [["$math.nope"]], "outputs": ["out"]},
            ],
        }
    )
    with pytest.raises(ValidationError, match="Cannot resolve segment 'nope'"):
        pipe.validate()

    # Valid static ref passes
    pipe_ok = Pipeline.from_dict(
        {
            "imports": ["math"],
            "steps": [
                {"fn": "abs", "args": ["$math.pi"], "outputs": ["out"]},
            ],
        }
    )
    pipe_ok.validate()
