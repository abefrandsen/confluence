"""Config-driven `DictModule` backed by `nn.ModuleDict` and a Confluence forward pipeline."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from confluence.builtins_scope import get_default_builtins
from confluence.config import PipelineConfig
from confluence.errors import PathSyntaxError, ResolutionError, ValidationError
from confluence.path import (
    CompiledRef,
    InvalidRef,
    compile_value,
    is_valid_identifier,
    resolve_compiled_value,
    resolve_parsed_path,
    resolve_path,
)
from confluence.pipeline import Pipeline, load_imports
from confluence.torch.config import (
    DictModuleConfig,
    load_dict_module_config,
    resolve_dict_module_config,
)

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "confluence.torch requires PyTorch. "
        "Install it with `pip install 'confluence[torch]'` or `uv sync --extra torch`."
    ) from exc

_RESERVED_MODULE_DICT_ATTRS: frozenset[str] = frozenset(dir(nn.ModuleDict()))


def _import_dotted_target(dotted_path: str) -> Any:
    """Resolve a dotted `package.module.Class` path via importlib.

    Re-raises `ModuleNotFoundError` immediately if it came from a missing transitive
    dependency inside an existing module rather than the target path itself.
    """
    parts = dotted_path.split(".")
    for split_idx in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split_idx])
        try:
            obj: Any = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            missing = exc.name
            if missing is not None and (missing == mod_name or mod_name.startswith(f"{missing}.")):
                continue
            raise
        for attr in parts[split_idx:]:
            if not hasattr(obj, attr):
                raise ResolutionError(
                    f"Module {mod_name!r} has no attribute {attr!r} in type {dotted_path!r}"
                )
            obj = getattr(obj, attr)
        return obj

    raise ResolutionError(f"Could not import component type {dotted_path!r}")


def _resolve_component_type(
    type_path: str,
    registry: Mapping[str, Any],
    imports: Mapping[str, Any],
) -> Any:
    """Resolve a component's `type` from registry, imports, or a dotted import path."""
    if not isinstance(type_path, str) or not type_path.strip():
        raise ResolutionError("Component 'type' must be a non-empty string")

    try:
        target = resolve_path(type_path, (registry, imports))
    except ResolutionError:
        if "." not in type_path:
            raise
        target = _import_dotted_target(type_path)

    if not callable(target):
        raise ResolutionError(
            f"Resolved component type {type_path!r} (type {type(target).__name__}) is not callable"
        )
    return target


class DictModule(nn.Module):
    """A PyTorch `nn.Module` whose components and forward graph are defined in config.

    Components are instantiated in declaration order and registered (unwrapped) in
    `self.components` (`nn.ModuleDict`), so later components may reference earlier ones
    (or their attributes) in `args` / `params` using Confluence `$ref` syntax. Any
    per-component `compile` wrappers are applied only to the callables wired into
    `self.forward_pipe.functions`, keeping `state_dict()` keys free of `_orig_mod.` prefixes.
    """

    def __init__(
        self,
        config: DictModuleConfig,
        *,
        registry: Mapping[str, Any] | None = None,
        functions: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        reg = dict(registry) if registry else {}
        extra_functions = dict(functions) if functions else {}
        loaded_imports, errors = load_imports(config.imports)
        builtins_scope = get_default_builtins()
        static_scopes = (reg, loaded_imports, builtins_scope)
        static_names = set(reg) | set(loaded_imports) | set(builtins_scope)

        # Reject component names that collide with explicitly passed `functions`
        overlap = sorted(set(extra_functions) & set(config.components))
        if overlap:
            errors.append(
                f"Component name(s) {overlap} conflict with keys in 'functions'"
            )

        # Pre-flight validation: collect all component config errors before instantiating anything
        known_components: set[str] = set()
        resolved_types: dict[str, Any] = {}
        compiled_args: dict[str, Any] = {}
        compiled_params: dict[str, Any] = {}

        for comp_name, spec in config.components.items():
            if not is_valid_identifier(comp_name):
                errors.append(
                    f"Component {comp_name!r}: invalid component name "
                    "(must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]*)"
                )
            elif comp_name in _RESERVED_MODULE_DICT_ATTRS:
                errors.append(
                    f"Component {comp_name!r}: name conflicts with an existing "
                    "nn.ModuleDict attribute"
                )

            try:
                resolved_types[comp_name] = _resolve_component_type(spec.type, reg, loaded_imports)
            except (PathSyntaxError, ResolutionError) as exc:
                errors.append(f"Component {comp_name!r}: {exc}")

            refs: list[CompiledRef | InvalidRef] = []
            compiled_args[comp_name] = compile_value(spec.args, refs)
            compiled_params[comp_name] = compile_value(spec.params, refs)
            for ref in refs:
                if isinstance(ref, InvalidRef):
                    errors.append(
                        f"Component {comp_name!r}: invalid reference '${ref.raw}' — {ref.error}"
                    )
                else:
                    parsed_ref = ref.parsed
                    if (
                        parsed_ref.root not in known_components
                        and parsed_ref.root not in static_names
                    ):
                        errors.append(
                            f"Component {comp_name!r}: argument references undefined name "
                            f"{parsed_ref.root!r} in '${parsed_ref.raw}'"
                        )
                    elif (
                        parsed_ref.root not in known_components
                        and parsed_ref.root in static_names
                    ):
                        try:
                            resolve_parsed_path(parsed_ref, static_scopes)
                        except ResolutionError as exc:
                            errors.append(
                                f"Component {comp_name!r}: cannot resolve argument reference "
                                f"'${parsed_ref.raw}' — {exc}"
                            )

            if (
                spec.compile is not False
                and spec.compile is not None
                and spec.compile is not True
                and not isinstance(spec.compile, Mapping)
            ):
                errors.append(
                    f"Component {comp_name!r}: 'compile' must be a bool or mapping of kwargs, "
                    f"got {type(spec.compile).__name__}"
                )

            if is_valid_identifier(comp_name):
                known_components.add(comp_name)

        # Pre-flight validate `forward` (syntax, dataflow, non-component static callables)
        # and `outputs` before instantiating any nn.Module components.
        forward_pipeline_cfg = PipelineConfig(
            imports=list(config.imports),
            inputs=list(config.forward.inputs),
            steps=list(config.forward.steps),
        )
        pre_pipe = Pipeline(config=forward_pipeline_cfg, functions=extra_functions)
        pre_input_keys = set(config.forward.inputs) | known_components
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pre_pipe.validate(input_keys=pre_input_keys)
        except ValidationError as exc:
            for err in exc.errors:
                if err not in errors:
                    errors.append(err)

        if config.outputs is not None:
            if not config.outputs:
                errors.append(
                    "DictModule 'outputs' must be non-empty when specified "
                    "(omit 'outputs' or set to null to return the full state dict)"
                )
            else:
                available_keys: set[str] = set(config.forward.inputs)
                for step in config.forward.steps:
                    if step.outputs:
                        available_keys.update(k for k in step.outputs if k != "_")
                seen_outputs: set[str] = set()
                missing_outputs: list[str] = []
                for k in config.outputs:
                    if k in seen_outputs:
                        errors.append(f"DictModule 'outputs' contains duplicate key {k!r}")
                    else:
                        seen_outputs.add(k)
                        if k not in available_keys:
                            missing_outputs.append(k)
                if missing_outputs:
                    errors.append(
                        f"DictModule outputs {missing_outputs} are not produced by "
                        "forward inputs or steps"
                    )

        if errors:
            raise ValidationError(errors)

        # Pass 1: ordered component instantiation and optional freezing
        built: dict[str, nn.Module] = {}
        for comp_name, spec in config.components.items():
            cls = resolved_types[comp_name]
            try:
                scopes = (built, reg, loaded_imports, builtins_scope)
                resolved_pos = resolve_compiled_value(compiled_args[comp_name], scopes)
                resolved_kw = resolve_compiled_value(compiled_params[comp_name], scopes)
                mod = cls(*resolved_pos, **resolved_kw)
            except Exception as exc:
                exc.add_note(f"While building component {comp_name!r} [type={spec.type!r}]")
                raise
            if not isinstance(mod, nn.Module):
                raise TypeError(
                    f"Component {comp_name!r} (type {spec.type!r}) must instantiate an nn.Module, "
                    f"got {type(mod).__name__}"
                )
            if spec.freeze:
                for param in mod.parameters():
                    param.requires_grad_(False)
            built[comp_name] = mod

        # Register raw unwrapped modules so `state_dict()` keys stay clean across compile settings
        self.components = nn.ModuleDict(built)

        # Wire and validate the forward pipeline against the real instantiated components
        pipe_functions: dict[str, Any] = {**extra_functions, **self.components}
        self.forward_pipe = Pipeline(config=forward_pipeline_cfg, functions=pipe_functions)
        self.forward_pipe.validate()
        self._input_names: tuple[str, ...] = tuple(config.forward.inputs)
        self._input_set: frozenset[str] = frozenset(self._input_names)

        # Pass 2: apply per-component `torch.compile` only to forward_pipe.functions
        self._compiled_names: tuple[str, ...] = tuple(
            name
            for name, spec in config.components.items()
            if spec.compile is not False and spec.compile is not None
        )
        self._apply_compiles()

        self.output_keys: list[str] | None = (
            list(config.outputs) if config.outputs is not None else None
        )

    def _apply_compiles(self) -> None:
        """Wrap configured components with `torch.compile` inside `self.forward_pipe.functions`."""
        for comp_name in self._compiled_names:
            spec = self.config.components[comp_name]
            raw_mod = self.components[comp_name]
            if spec.compile is True:
                self.forward_pipe.functions[comp_name] = torch.compile(raw_mod)
            elif isinstance(spec.compile, Mapping):
                self.forward_pipe.functions[comp_name] = torch.compile(raw_mod, **spec.compile)

    def __getstate__(self) -> dict[str, Any]:
        """Avoid pickling/deepcopying `torch.compile` wrappers; re-apply them in `__setstate__`."""
        parent_getstate = getattr(super(), "__getstate__", None)
        state = dict(parent_getstate()) if parent_getstate is not None else self.__dict__.copy()
        if self._compiled_names:
            raw_functions = dict(self.forward_pipe.functions)
            for comp_name in self._compiled_names:
                raw_functions[comp_name] = self.components[comp_name]
            state["forward_pipe"] = Pipeline(
                config=self.forward_pipe.config,
                functions=raw_functions,
            )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        parent_setstate = getattr(super(), "__setstate__", None)
        if parent_setstate is not None:
            parent_setstate(state)
        else:
            self.__dict__.update(state)
        self._apply_compiles()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        registry: Mapping[str, Any] | None = None,
        functions: Mapping[str, Any] | None = None,
        overrides: Sequence[str] | None = None,
    ) -> DictModule:
        """Create a DictModule from a YAML file with optional OmegaConf dotlist overrides."""
        config = load_dict_module_config(path, overrides=overrides)
        return cls(config=config, registry=registry, functions=functions)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | DictConfig,
        *,
        registry: Mapping[str, Any] | None = None,
        functions: Mapping[str, Any] | None = None,
        overrides: Sequence[str] | None = None,
    ) -> DictModule:
        """Create a DictModule from a Python dict or OmegaConf DictConfig."""
        raw: Any = data if isinstance(data, DictConfig) else dict(data)
        config = resolve_dict_module_config(raw, overrides=overrides)
        return cls(config=config, registry=registry, functions=functions)

    def forward_dict(self, batch: Mapping[str, Any]) -> Any:
        """Execute the module via `__call__` using a batch mapping (extra unused keys are ignored)."""
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"DictModule.forward_dict() expects a mapping, got {type(batch).__name__}"
            )
        if not self._input_set.issubset(batch):
            missing = [k for k in self._input_names if k not in batch]
            raise TypeError(f"DictModule.forward() missing required input(s): {missing}")
        return self(**{k: batch[k] for k in self._input_names})

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the configured forward pipeline with standard Python positional/keyword binding.

        Positional arguments always bind to `forward.inputs` in order (even when an argument
        is a dict). To unpack a batch mapping with extra keys, use `forward_dict(batch)`.
        """
        if kwargs and not self._input_set.issuperset(kwargs):
            unexpected = [k for k in kwargs if k not in self._input_set]
            raise TypeError(f"DictModule.forward() got unexpected keyword argument(s): {unexpected}")

        if not args:
            inputs: Mapping[str, Any] = kwargs
        else:
            if len(args) > len(self._input_names):
                raise TypeError(
                    f"Expected at most {len(self._input_names)} positional argument(s) "
                    f"({list(self._input_names)}), got {len(args)}"
                )
            merged: dict[str, Any] = dict(zip(self._input_names, args))
            overlap = set(merged) & set(kwargs)
            if overlap:
                raise TypeError(f"Got multiple values for input(s): {sorted(overlap)}")
            merged.update(kwargs)
            inputs = merged

        if not self._input_set.issubset(inputs):
            missing = [k for k in self._input_names if k not in inputs]
            raise TypeError(f"DictModule.forward() missing required input(s): {missing}")

        state = self.forward_pipe.run(inputs, validate=False)
        if self.output_keys is None:
            return state
        if len(self.output_keys) == 1:
            return state[self.output_keys[0]]
        return {k: state[k] for k in self.output_keys}
