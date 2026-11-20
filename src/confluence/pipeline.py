"""Core Pipeline loader, static validator, and sequential executor."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from confluence.builtins_scope import get_safe_builtins
from confluence.config import (
    PipelineConfig,
    StepConfig,
    load_pipeline_config,
    resolve_pipeline_config,
)
from confluence.errors import (
    OutputBindingError,
    PathSyntaxError,
    ResolutionError,
    StepExecutionError,
    ValidationError,
)
from confluence.path import (
    collect_references,
    parse_path,
    resolve_parsed_path,
    resolve_value,
)


def _step_label(index: int, step: StepConfig) -> str:
    return f"Step {index} ({step.name})" if step.name else f"Step {index}"


class Pipeline:
    """A resolved, validated, callable data pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        functions: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.functions: dict[str, Any] = dict(functions) if functions else {}
        self._builtins: dict[str, Any] = get_safe_builtins()
        self._imports: dict[str, Any] = {}
        self._imports_loaded = False

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        functions: Mapping[str, Any] | None = None,
        overrides: Sequence[str] | None = None,
    ) -> Pipeline:
        """Create a Pipeline from a YAML file with optional OmegaConf dotlist overrides."""
        config = load_pipeline_config(path, overrides=overrides)
        return cls(config=config, functions=functions)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | DictConfig,
        functions: Mapping[str, Any] | None = None,
        overrides: Sequence[str] | None = None,
    ) -> Pipeline:
        """Create a Pipeline from a Python dict or OmegaConf DictConfig."""
        raw: Any = data if isinstance(data, DictConfig) else dict(data)
        config = resolve_pipeline_config(raw, overrides=overrides)
        return cls(config=config, functions=functions)

    def _load_imports(self) -> list[str]:
        """Import declared modules and bind top-level packages. Returns import error messages."""
        if self._imports_loaded:
            return []

        errors: list[str] = []
        loaded: dict[str, Any] = {}
        for mod_path in self.config.imports:
            if not isinstance(mod_path, str) or not mod_path.strip():
                errors.append(f"Invalid import entry: {mod_path!r}")
                continue
            try:
                importlib.import_module(mod_path)
                top_level = mod_path.split(".", 1)[0]
                loaded[top_level] = sys.modules[top_level]
            except Exception as exc:
                errors.append(f"Failed to import {mod_path!r}: {exc}")

        if not errors:
            self._imports = loaded
            self._imports_loaded = True
        return errors

    def validate(self, input_keys: Iterable[str] | None = None) -> None:
        """Run pre-flight static validation on imports, paths, references, and callables.

        Raises ValidationError containing all discovered errors.
        """
        errors: list[str] = []
        errors.extend(self._load_imports())

        static_scopes = (self.functions, self._imports, self._builtins)
        static_names = set(self.functions) | set(self._imports) | set(self._builtins)

        known_keys: set[str] = set()
        if self.config.inputs is not None:
            known_keys.update(self.config.inputs)
        if input_keys is not None:
            provided = set(input_keys)
            if self.config.inputs is not None:
                missing_inputs = [k for k in self.config.inputs if k not in provided]
                if missing_inputs:
                    errors.append(f"Missing required pipeline inputs: {missing_inputs}")
            known_keys.update(provided)

        for idx, step in enumerate(self.config.steps):
            label = _step_label(idx, step)

            # 1. Validate `fn` path syntax & resolution
            if not step.fn or not step.fn.strip():
                errors.append(f"{label}: 'fn' must be a non-empty string")
            else:
                try:
                    parsed_fn = parse_path(step.fn)
                except PathSyntaxError as exc:
                    errors.append(f"{label}: invalid 'fn' path — {exc}")
                else:
                    if parsed_fn.root not in known_keys and parsed_fn.root not in static_names:
                        errors.append(
                            f"{label}: 'fn' references undefined name {parsed_fn.root!r} "
                            f"in {step.fn!r}"
                        )
                    elif parsed_fn.root not in known_keys and parsed_fn.root in static_names:
                        try:
                            target = resolve_parsed_path(parsed_fn, static_scopes)
                        except ResolutionError as exc:
                            errors.append(f"{label}: cannot resolve 'fn' — {exc}")
                        else:
                            if not callable(target):
                                errors.append(
                                    f"{label}: resolved 'fn' {step.fn!r} "
                                    f"(type {type(target).__name__}) is not callable"
                                )

            # 2. Validate all `$ref` paths in args and kwargs
            refs = collect_references(step.args) + collect_references(step.kwargs)
            for ref in refs:
                try:
                    parsed_ref = parse_path(ref)
                except PathSyntaxError as exc:
                    errors.append(f"{label}: invalid reference '${ref}' — {exc}")
                else:
                    if parsed_ref.root not in known_keys and parsed_ref.root not in static_names:
                        errors.append(
                            f"{label}: argument references undefined name {parsed_ref.root!r} "
                            f"in '${ref}'"
                        )

            # 3. Validate `outputs` and update known_keys
            if step.outputs is not None:
                for out_name in step.outputs:
                    if not isinstance(out_name, str) or not out_name.strip():
                        errors.append(f"{label}: invalid output binding name {out_name!r}")
                    elif out_name != "_":
                        known_keys.add(out_name)

        if errors:
            raise ValidationError(errors)

    def run(self, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate and execute the pipeline, returning the resulting data dict."""
        state: dict[str, Any] = dict(data) if data is not None else {}
        self.validate(input_keys=state.keys())

        scopes = (state, self.functions, self._imports, self._builtins)

        for idx, step in enumerate(self.config.steps):
            try:
                parsed_fn = parse_path(step.fn)
                target = resolve_parsed_path(parsed_fn, scopes)
                if not callable(target):
                    raise ResolutionError(
                        f"Resolved 'fn' {step.fn!r} (type {type(target).__name__}) is not callable"
                    )

                resolved_args = resolve_value(step.args, scopes)
                resolved_kwargs = resolve_value(step.kwargs, scopes)

                result = target(*resolved_args, **resolved_kwargs)
                _bind_outputs(state, step.outputs, result)
            except Exception as exc:
                raise StepExecutionError(
                    step_index=idx,
                    fn=step.fn,
                    cause=exc,
                    step_name=step.name,
                ) from exc

        return state

    def __call__(self, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Alias for `run(data)` so a Pipeline can be composed as a callable step."""
        return self.run(data)


def _bind_outputs(
    state: dict[str, Any],
    outputs: list[str] | None,
    result: Any,
) -> None:
    """Bind a step's return value into `state` according to v1 output rules."""
    if not outputs:
        return

    if len(outputs) == 1:
        key = outputs[0]
        if key != "_":
            state[key] = result
        return

    # len(outputs) > 1 -> unpack iterable
    if isinstance(result, (str, bytes)) or not isinstance(result, Iterable):
        raise OutputBindingError(
            f"Expected an iterable of length {len(outputs)} to unpack into {outputs}, "
            f"got non-unpackable {type(result).__name__}"
        )

    items = list(result)
    if len(items) != len(outputs):
        raise OutputBindingError(
            f"Cannot unpack {len(items)} value(s) into {len(outputs)} outputs {outputs}"
        )

    for key, val in zip(outputs, items, strict=True):
        if key != "_":
            state[key] = val
