"""Custom exception hierarchy for Confluence pipelines."""

from __future__ import annotations


class ConfluenceError(Exception):
    """Base exception for all Confluence errors."""


class PathSyntaxError(ConfluenceError):
    """Raised when a dotted/indexed path string cannot be parsed."""


class ResolutionError(ConfluenceError):
    """Raised when a reference or callable path cannot be resolved at runtime."""


class OutputBindingError(ConfluenceError):
    """Raised when a step's return value cannot be bound to its declared outputs."""


class ValidationError(ConfluenceError):
    """Raised when static pre-flight validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  - {err}" for err in errors)
        super().__init__(f"Pipeline validation failed with {len(errors)} error(s):\n{bullet_list}")
