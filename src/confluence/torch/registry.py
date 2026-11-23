"""Global component registry for `confluence.torch` (`@register_component`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

from confluence.path import is_valid_identifier

_T = TypeVar("_T", bound=Callable[..., Any])

_COMPONENT_REGISTRY: dict[str, Any] = {}


@overload
def register_component(
    cls: str | None = None, *, name: str | None = None, replace: bool = False
) -> Callable[[_T], _T]: ...


@overload
def register_component(cls: _T, *, name: str | None = None, replace: bool = False) -> _T: ...


def register_component(
    cls: Any = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> Any:
    """Register a callable/module class for lookup by name in `DictModule` configs.

    Supports both bare and parameterized decorator forms:
        @register_component
        class MyBlock(nn.Module): ...

        @register_component("custom_name")
        class MyBlock(nn.Module): ...

        @register_component(name="custom_name", replace=True)
        class MyBlock(nn.Module): ...
    """
    if isinstance(cls, str):
        if name is not None:
            raise TypeError("Cannot pass both a positional name and keyword 'name'")
        reg_name: str | None = cls
        target_cls = None
    else:
        reg_name = name
        target_cls = cls

    def _decorator(obj: _T) -> _T:
        if not callable(obj):
            raise TypeError(
                f"Registered component must be callable, got {type(obj).__name__}"
            )
        key = reg_name if reg_name is not None else getattr(obj, "__name__", None)
        if not is_valid_identifier(key):
            raise ValueError(
                f"Invalid component registry name {key!r} "
                "(must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]*)"
            )
        assert isinstance(key, str)
        existing = _COMPONENT_REGISTRY.get(key)
        if existing is not None and existing is not obj and not replace:
            raise ValueError(
                f"Component {key!r} is already registered to {existing!r} "
                "(pass replace=True to overwrite)"
            )
        _COMPONENT_REGISTRY[key] = obj
        return obj

    if target_cls is not None:
        return _decorator(target_cls)
    return _decorator


def get_component_registry() -> dict[str, Any]:
    """Return a shallow copy of the global component registry."""
    return dict(_COMPONENT_REGISTRY)


def unregister_component(name: str) -> None:
    """Remove a component from the global registry by name."""
    del _COMPONENT_REGISTRY[name]


def clear_component_registry() -> None:
    """Remove all entries from the global component registry."""
    _COMPONENT_REGISTRY.clear()
