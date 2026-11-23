"""OmegaConf structured configuration schemas for `confluence.torch`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf

from confluence.config import StepConfig, build_step_configs


@dataclass
class ComponentConfig:
    """Configuration for a single `nn.Module` component inside a `DictModule`."""

    type: str = ""
    args: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    compile: Any = False
    freeze: bool = False


@dataclass
class ForwardConfig:
    """Forward pipeline specification for a `DictModule` (shares top-level `imports` and `params`)."""

    inputs: list[str] = field(default_factory=list)
    steps: list[StepConfig] = field(default_factory=list)


@dataclass
class DictModuleConfig:
    """Top-level configuration for a config-driven `DictModule`."""

    imports: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    components: dict[str, ComponentConfig] = field(default_factory=dict)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    outputs: Optional[list[str]] = None


def resolve_dict_module_config(
    raw_cfg: Union[DictConfig, dict[str, Any], DictModuleConfig],
    overrides: Optional[Sequence[str]] = None,
) -> DictModuleConfig:
    """Validate against DictModuleConfig schema, apply overrides, and resolve interpolations."""
    schema = OmegaConf.structured(DictModuleConfig)
    configs: list[Any] = [schema, raw_cfg]
    if overrides:
        configs.append(OmegaConf.from_dotlist(list(overrides)))

    merged = OmegaConf.merge(*configs)
    container = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    assert isinstance(container, dict)

    raw_components = container.get("components") or {}
    components: dict[str, ComponentConfig] = {}
    for comp_name, comp_data in raw_components.items():
        if not isinstance(comp_data, dict):
            raise ValueError(
                f"Component {comp_name!r} must be a mapping, got {type(comp_data).__name__}"
            )
        components[comp_name] = ComponentConfig(
            type=comp_data.get("type") or "",
            args=list(comp_data.get("args") or []),
            params=dict(comp_data.get("params") or {}),
            compile=comp_data.get("compile", False),
            freeze=bool(comp_data.get("freeze", False)),
        )

    raw_forward = container.get("forward") or {}
    forward_cfg = ForwardConfig(
        inputs=list(raw_forward.get("inputs") or []),
        steps=build_step_configs(raw_forward.get("steps")),
    )

    raw_outputs = container.get("outputs")
    outputs = list(raw_outputs) if raw_outputs is not None else None

    return DictModuleConfig(
        imports=list(container.get("imports") or []),
        params=dict(container.get("params") or {}),
        components=components,
        forward=forward_cfg,
        outputs=outputs,
    )


def load_dict_module_config(
    path: str | Path,
    overrides: Optional[Sequence[str]] = None,
) -> DictModuleConfig:
    """Load, schema-validate, and resolve a DictModuleConfig from a YAML file."""
    raw = OmegaConf.load(Path(path))
    if not isinstance(raw, DictConfig):
        raise ValueError(f"Expected a YAML mapping at root of {path}, got {type(raw).__name__}")
    return resolve_dict_module_config(raw, overrides=overrides)
