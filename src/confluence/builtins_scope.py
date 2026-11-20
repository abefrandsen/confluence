"""Default Python builtins scope for pipeline resolution.

Note: Excluding interactive/footgun builtins is a guardrail against accidental misuse,
not a security boundary. Pipeline YAML must be treated as trusted code.
"""

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


def get_default_builtins() -> dict[str, Any]:
    """Return a dict of default Python builtins, excluding interactive/footgun and dunder entries."""
    return {
        name: obj
        for name, obj in vars(builtins).items()
        if name not in EXCLUDED_BUILTINS and not (name.startswith("__") and name.endswith("__"))
    }
