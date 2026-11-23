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

if TYPE_CHECKING:
    from confluence.torch.module import DictModule

__all__ = [
    "ComponentConfig",
    "DictModule",
    "DictModuleConfig",
    "ForwardConfig",
    "load_dict_module_config",
    "resolve_dict_module_config",
]


def __getattr__(name: str) -> Any:
    if name == "DictModule":
        from confluence.torch.module import DictModule

        return DictModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
