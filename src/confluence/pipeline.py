"""Core Pipeline loader, static validator, and sequential executor."""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from confluence.builtins_scope import get_default_builtins
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
    ValidationError,
)
from confluence.path import (
    CompiledRef,
    InvalidRef,
    ParsedPath,
    compile_value,
    is_valid_identifier,
    parse_path,
    resolve_compiled_value,
    resolve_parsed_path,
)


def load_imports(import_paths: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Import declared modules and bind top-level packages, returning `(loaded, errors)`."""
    errors: list[str] = []
    loaded: dict[str, Any] = {}
    for mod_path in import_paths:
        if not isinstance(mod_path, str) or not mod_path.strip():
            errors.append(f"Invalid import entry: {mod_path!r}")
            continue
        try:
            importlib.import_module(mod_path)
            top_level = mod_path.split(".", 1)[0]
            loaded[top_level] = sys.modules[top_level]
        except Exception as exc:
            errors.append(f"Failed to import {mod_path!r}: {exc}")

    return loaded, errors


def _step_label(index: int, step: StepConfig) -> str:
    return f"Step {index} ({step.name})" if step.name else f"Step {index}"


@dataclass(frozen=True)
class _CompiledStep:
    """Pre-parsed representation of a StepConfig for fast validation and execution."""

    step: StepConfig
    fn: ParsedPath | None
    fn_error: PathSyntaxError | None
    fn_is_empty: bool
    args: list[Any]
    kwargs: dict[str, Any]
    refs: tuple[CompiledRef | InvalidRef, ...]


def _compile_step(step: StepConfig) -> _CompiledStep:
    fn: ParsedPath | None = None
    fn_error: PathSyntaxError | None = None
    fn_is_empty = not step.fn or not step.fn.strip()
    if fn_is_empty:
        fn_error = PathSyntaxError(f"Invalid empty path: {step.fn!r}")
    else:
        try:
            fn = parse_path(step.fn)
        except PathSyntaxError as exc:
            fn_error = exc

    refs: list[CompiledRef | InvalidRef] = []
    compiled_args = compile_value(step.args, refs)
    compiled_kwargs = compile_value(step.kwargs, refs)
    return _CompiledStep(
        step=step,
        fn=fn,
        fn_error=fn_error,
        fn_is_empty=fn_is_empty,
        args=compiled_args,
        kwargs=compiled_kwargs,
        refs=tuple(refs),
    )


class Pipeline:
    """A resolved, validated, callable data pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        functions: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.functions: dict[str, Any] = dict(functions) if functions else {}
        self._builtins: dict[str, Any] = get_default_builtins()
        self._imports: dict[str, Any] = {}
        self._imports_loaded = False
        self._compiled_steps: tuple[_CompiledStep, ...] = tuple(
            _compile_step(step) for step in config.steps
        )

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

        loaded, errors = load_imports(self.config.imports)
        self._imports = loaded
        if not errors:
            self._imports_loaded = True
        return errors

    def validate(self, input_keys: Iterable[str] | None = None) -> None:
        """Run pre-flight static validation on imports, paths, references, and callables.

        Raises ValidationError containing all discovered errors.
        """
        errors: list[str] = []
        errors.extend(self._load_imports())

        static_scopes = (self.functions, self._imports, self._builtins)
        named_static_scopes = (
            ("functions", self.functions),
            ("imports", self._imports),
            ("builtins", self._builtins),
        )
        static_names = set(self.functions) | set(self._imports) | set(self._builtins)

        warned_keys: set[str] = set()

        def _warn_if_shadowed(key: str, context: str) -> None:
            if key in warned_keys:
                return
            for scope_name, scope in named_static_scopes:
                if key in scope:
                    warned_keys.add(key)
                    warnings.warn(
                        f"{context}: data key {key!r} shadows a name in {scope_name}",
                        UserWarning,
                        stacklevel=3,
                    )
                    break

        known_keys: set[str] = set()
        if self.config.inputs is not None:
            seen_inputs: set[str] = set()
            for inp in self.config.inputs:
                if not is_valid_identifier(inp):
                    errors.append(
                        f"Pipeline inputs: invalid input name {inp!r} "
                        "(must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]*)"
                    )
                elif inp in seen_inputs:
                    errors.append(f"Pipeline inputs: duplicate input name {inp!r}")
                else:
                    seen_inputs.add(inp)
                    _warn_if_shadowed(inp, "Pipeline inputs")
                    known_keys.add(inp)
        if input_keys is not None:
            provided = set(input_keys)
            for inp in provided:
                _warn_if_shadowed(inp, "Runtime inputs")
            if self.config.inputs is not None:
                missing_inputs = [
                    k for k in self.config.inputs if is_valid_identifier(k) and k not in provided
                ]
                if missing_inputs:
                    errors.append(f"Missing required pipeline inputs: {missing_inputs}")
            known_keys.update(provided)

        for idx, compiled in enumerate(self._compiled_steps):
            step = compiled.step
            label = _step_label(idx, step)

            # 1. Validate `fn` path syntax & resolution
            if compiled.fn_is_empty:
                errors.append(f"{label}: 'fn' must be a non-empty string")
            elif compiled.fn is None:
                errors.append(f"{label}: invalid 'fn' path — {compiled.fn_error}")
            else:
                parsed_fn = compiled.fn
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
            for ref in compiled.refs:
                if isinstance(ref, InvalidRef):
                    errors.append(f"{label}: invalid reference '${ref.raw}' — {ref.error}")
                else:
                    parsed_ref = ref.parsed
                    if parsed_ref.root not in known_keys and parsed_ref.root not in static_names:
                        errors.append(
                            f"{label}: argument references undefined name {parsed_ref.root!r} "
                            f"in '${parsed_ref.raw}'"
                        )
                    elif parsed_ref.root not in known_keys and parsed_ref.root in static_names:
                        try:
                            resolve_parsed_path(parsed_ref, static_scopes)
                        except ResolutionError as exc:
                            errors.append(
                                f"{label}: cannot resolve argument reference "
                                f"'${parsed_ref.raw}' — {exc}"
                            )

            # 3. Validate `outputs` and update known_keys
            if step.outputs is not None:
                seen_outputs: set[str] = set()
                for out_name in step.outputs:
                    if not is_valid_identifier(out_name):
                        errors.append(
                            f"{label}: invalid output binding name {out_name!r} "
                            "(must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]*)"
                        )
                    elif out_name != "_":
                        if out_name in seen_outputs:
                            errors.append(f"{label}: duplicate output binding name {out_name!r}")
                        else:
                            seen_outputs.add(out_name)
                            _warn_if_shadowed(out_name, label)
                            known_keys.add(out_name)

        if errors:
            raise ValidationError(errors)

    def run(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Execute the pipeline, returning the resulting data dict.

        Args:
            data: Optional initial data mapping (shallow-copied before execution).
            validate: If True (default), run pre-flight `validate()` before step 0.
                Set to False for hot inner loops after validating once up front.
        """
        state: dict[str, Any] = dict(data) if data is not None else {}
        if validate:
            self.validate(input_keys=state.keys())
        elif not self._imports_loaded:
            import_errors = self._load_imports()
            if import_errors:
                raise ValidationError(import_errors)

        scopes = (state, self.functions, self._imports, self._builtins)

        for idx, compiled in enumerate(self._compiled_steps):
            step = compiled.step
            try:
                if compiled.fn is None:
                    assert compiled.fn_error is not None
                    raise compiled.fn_error

                target = resolve_parsed_path(compiled.fn, scopes)
                if not callable(target):
                    raise ResolutionError(
                        f"Resolved 'fn' {step.fn!r} (type {type(target).__name__}) is not callable"
                    )

                resolved_args = resolve_compiled_value(compiled.args, scopes)
                resolved_kwargs = resolve_compiled_value(compiled.kwargs, scopes)

                result = target(*resolved_args, **resolved_kwargs)
                _bind_outputs(state, step.outputs, result)
            except Exception as exc:
                exc.add_note(f"In {_step_label(idx, step)} [fn={step.fn!r}]")
                raise

        return state

    def __call__(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Alias for `run(data, validate=validate)` so a Pipeline can be composed as a callable step."""
        return self.run(data, validate=validate)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude unpicklable module objects and transient caches when pickling/deepcopying."""
        return {
            "config": self.config,
            "functions": self.functions,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.config = state["config"]
        self.functions = state["functions"]
        self._builtins = get_default_builtins()
        self._imports = {}
        self._imports_loaded = False
        self._compiled_steps = tuple(_compile_step(step) for step in self.config.steps)


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

    # len(outputs) > 1 -> unpack iterable (reject str, bytes, and mappings)
    if isinstance(result, (str, bytes, Mapping)) or not isinstance(result, Iterable):
        hint = (
            " (bind a single output name to capture a mapping, then access keys via `$out.key`)"
            if isinstance(result, Mapping)
            else ""
        )
        raise OutputBindingError(
            f"Expected an iterable of length {len(outputs)} to unpack into {outputs}, "
            f"got non-unpackable {type(result).__name__}{hint}"
        )

    items = list(result)
    if len(items) != len(outputs):
        raise OutputBindingError(
            f"Cannot unpack {len(items)} value(s) into {len(outputs)} outputs {outputs}"
        )

    for key, val in zip(outputs, items, strict=True):
        if key != "_":
            state[key] = val
