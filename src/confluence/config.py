"""OmegaConf structured configuration schemas and loading helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""

    fn: str = ""
    name: Optional[str] = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    outputs: Optional[list[str]] = None


@dataclass
class PipelineConfig:
    """Top-level configuration for a Confluence pipeline."""

    imports: list[str] = field(default_factory=list)
    inputs: Optional[list[str]] = None
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[StepConfig] = field(default_factory=list)


def build_step_configs(raw_steps: Optional[Sequence[dict[str, Any]]]) -> list[StepConfig]:
    """Convert raw resolved step dicts into `StepConfig` instances."""
    return [
        StepConfig(
            fn=step["fn"],
            name=step.get("name"),
            args=list(step.get("args") or []),
            kwargs=dict(step.get("kwargs") or {}),
            outputs=list(step["outputs"]) if step.get("outputs") is not None else None,
        )
        for step in (raw_steps or [])
    ]


def resolve_pipeline_config(
    raw_cfg: Union[DictConfig, dict[str, Any], PipelineConfig],
    overrides: Optional[Sequence[str]] = None,
) -> PipelineConfig:
    """Validate against PipelineConfig schema, apply overrides, and resolve interpolations.

    Returns a plain Python PipelineConfig instance with all OmegaConf interpolations resolved
    and all containers converted to native Python lists/dicts.
    """
    schema = OmegaConf.structured(PipelineConfig)
    configs: list[Any] = [schema, raw_cfg]
    if overrides:
        configs.append(OmegaConf.from_dotlist(list(overrides)))

    merged = OmegaConf.merge(*configs)
    container = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    assert isinstance(container, dict)

    return PipelineConfig(
        imports=list(container.get("imports") or []),
        inputs=list(container["inputs"]) if container.get("inputs") is not None else None,
        params=dict(container.get("params") or {}),
        steps=build_step_configs(container.get("steps")),
    )


def load_pipeline_config(
    path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
) -> PipelineConfig:
    """Load, schema-validate, and resolve a PipelineConfig from a YAML file."""
    raw = OmegaConf.load(Path(path))
    if not isinstance(raw, DictConfig):
        raise ValueError(f"Expected a YAML mapping at root of {path}, got {type(raw).__name__}")
    return resolve_pipeline_config(raw, overrides=overrides)
