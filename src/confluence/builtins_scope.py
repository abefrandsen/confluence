"""Safe Python builtins scope for pipeline resolution."""

from __future__ import annotations

import builtins
from typing import Any

EXCLUDED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "exit",
        "quit",
        "help",
        "copyright",
        "credits",
        "license",
        "memoryview",
    }
)


def get_safe_builtins() -> dict[str, Any]:
    """Return a dict of safe Python builtins, excluding dangerous and dunder entries."""
    return {
        name: obj
        for name, obj in vars(builtins).items()
        if name not in EXCLUDED_BUILTINS and not (name.startswith("__") and name.endswith("__"))
    }
