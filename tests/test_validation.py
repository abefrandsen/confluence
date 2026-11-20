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
