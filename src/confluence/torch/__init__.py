"""Optional PyTorch utilities for config-driven module construction (`confluence[torch]`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from confluence.torch.config import (
    ComponentConfig,
    DictModuleConfig,
    ForwardConfig,
    load_dict_module_config,
    resolve_dict_module_config,
)
from confluence.torch.registry import (
    clear_component_registry,
    get_component_registry,
    register_component,
    unregister_component,
)

if TYPE_CHECKING:
    from confluence.torch.module import DictModule

__all__ = [
    "ComponentConfig",
    "DictModule",
    "DictModuleConfig",
    "ForwardConfig",
    "clear_component_registry",
    "get_component_registry",
    "load_dict_module_config",
    "register_component",
    "resolve_dict_module_config",
    "unregister_component",
]


def __getattr__(name: str) -> Any:
    if name == "DictModule":
        from confluence.torch.module import DictModule

        return DictModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
