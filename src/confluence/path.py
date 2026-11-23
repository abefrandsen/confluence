"""Path parsing and traversal for dotted attributes and bracket indexing."""

from __future__ import annotations

import ast
import importlib.util
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from confluence.errors import PathSyntaxError, ResolutionError

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOT_SEGMENT_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)")
_INT_BRACKET_RE = re.compile(r"\[(-?[0-9]+)\]")
_STR_BRACKET_RE = re.compile(r"""\[("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\]""")


def is_valid_identifier(name: Any) -> bool:
    """Return True if `name` is a valid Confluence root identifier (`[A-Za-z_][A-Za-z0-9_]*`)."""
    return isinstance(name, str) and _IDENT_RE.fullmatch(name) is not None


@dataclass(frozen=True)
class AttrSegment:
    """A dotted segment such as `.foo`."""

    name: str


@dataclass(frozen=True)
class IndexSegment:
    """A bracket subscript segment such as `[0]`, `[-1]`, or `["my.key"]`."""

    key: int | str


Segment = AttrSegment | IndexSegment


@dataclass(frozen=True)
class ParsedPath:
    """A parsed reference path with a root identifier and traversal segments."""

    raw: str
    root: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class CompiledRef:
    """A pre-parsed `$ref` inside a compiled argument template."""

    parsed: ParsedPath


@dataclass(frozen=True)
class InvalidRef:
    """A syntactically invalid `$ref` captured during compilation for deferred reporting."""

    raw: str
    error: PathSyntaxError


def parse_path(path: str) -> ParsedPath:
    """Parse a path like `prep.batches[0].values[-1]` or `cfg["my.key"]`.

    Raises PathSyntaxError on invalid syntax (e.g. empty path, slices, nested expressions).
    """
    if not isinstance(path, str) or not path:
        raise PathSyntaxError(f"Invalid empty path: {path!r}")

    match = _IDENT_RE.match(path, 0)
    if match is None:
        raise PathSyntaxError(f"Path {path!r} must start with a valid identifier")

    root = match.group(0)
    pos = match.end()
    segments: list[Segment] = []

    while pos < len(path):
        ch = path[pos]
        if ch == ".":
            m = _DOT_SEGMENT_RE.match(path, pos)
            if m is None:
                raise PathSyntaxError(f"Invalid dot segment at position {pos} in path {path!r}")
            segments.append(AttrSegment(m.group(1)))
            pos = m.end()
        elif ch == "[":
            m_int = _INT_BRACKET_RE.match(path, pos)
            if m_int is not None:
                segments.append(IndexSegment(int(m_int.group(1))))
                pos = m_int.end()
                continue

            m_str = _STR_BRACKET_RE.match(path, pos)
            if m_str is not None:
                try:
                    key = ast.literal_eval(m_str.group(1))
                except (SyntaxError, ValueError) as exc:
                    raise PathSyntaxError(
                        f"Invalid string key at position {pos} in path {path!r}"
                    ) from exc
                segments.append(IndexSegment(key))
                pos = m_str.end()
                continue

            raise PathSyntaxError(
                f"Invalid bracket index at position {pos} in path {path!r} "
                "(only integer indices and quoted string keys are supported)"
            )
        else:
            raise PathSyntaxError(
                f"Unexpected character {ch!r} at position {pos} in path {path!r}"
            )

    return ParsedPath(raw=path, root=root, segments=tuple(segments))


def resolve_parsed_path(
    parsed: ParsedPath,
    scopes: Sequence[Mapping[str, Any]],
) -> Any:
    """Resolve a ParsedPath against an ordered sequence of scopes."""
    found = False
    obj: Any = None
    for scope in scopes:
        if parsed.root in scope:
            obj = scope[parsed.root]
            found = True
            break

    if not found:
        raise ResolutionError(f"Name {parsed.root!r} (in path {parsed.raw!r}) is not defined")

    traversed = parsed.root
    for seg in parsed.segments:
        if isinstance(seg, AttrSegment):
            if isinstance(obj, Mapping) and seg.name in obj:
                obj = obj[seg.name]
            elif hasattr(obj, seg.name):
                obj = getattr(obj, seg.name)
            else:
                hint = ""
                if isinstance(obj, types.ModuleType) and getattr(obj, "__name__", None):
                    candidate = f"{obj.__name__}.{seg.name}"
                    try:
                        if importlib.util.find_spec(candidate) is not None:
                            hint = f" (did you mean to add {candidate!r} to imports?)"
                    except (ImportError, ValueError):
                        pass
                raise ResolutionError(
                    f"Cannot resolve segment {seg.name!r} on {traversed!r} "
                    f"(type {type(obj).__name__}) in path {parsed.raw!r}{hint}"
                )
            traversed = f"{traversed}.{seg.name}"
        else:
            try:
                obj = obj[seg.key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ResolutionError(
                    f"Cannot index {traversed!r} (type {type(obj).__name__}) "
                    f"with [{seg.key!r}] in path {parsed.raw!r}: {exc}"
                ) from exc
            traversed = f"{traversed}[{seg.key!r}]"

    return obj


def resolve_path(path: str, scopes: Sequence[Mapping[str, Any]]) -> Any:
    """Parse and resolve a path string against an ordered sequence of scopes."""
    return resolve_parsed_path(parse_path(path), scopes)


def resolve_value(value: Any, scopes: Sequence[Mapping[str, Any]]) -> Any:
    """Recursively resolve `$`-prefixed references in arguments and containers."""
    return resolve_compiled_value(compile_value(value), scopes)


def compile_value(
    value: Any,
    refs_out: list[CompiledRef | InvalidRef] | None = None,
) -> Any:
    """Pre-parse `$`-prefixed references and unescape `$$` literals in argument templates."""
    if isinstance(value, str):
        if value.startswith("$$"):
            return value[1:]
        if value.startswith("$"):
            raw = value[1:]
            ref: CompiledRef | InvalidRef
            try:
                ref = CompiledRef(parse_path(raw))
            except PathSyntaxError as exc:
                ref = InvalidRef(raw=raw, error=exc)
            if refs_out is not None:
                refs_out.append(ref)
            return ref
        return value
    if isinstance(value, list):
        return [compile_value(item, refs_out) for item in value]
    if isinstance(value, tuple):
        return tuple(compile_value(item, refs_out) for item in value)
    if isinstance(value, dict):
        return {k: compile_value(v, refs_out) for k, v in value.items()}
    return value


def resolve_compiled_value(value: Any, scopes: Sequence[Mapping[str, Any]]) -> Any:
    """Resolve a value tree previously compiled by `compile_value`."""
    if isinstance(value, CompiledRef):
        return resolve_parsed_path(value.parsed, scopes)
    if isinstance(value, InvalidRef):
        raise value.error
    if isinstance(value, list):
        return [resolve_compiled_value(item, scopes) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_compiled_value(item, scopes) for item in value)
    if isinstance(value, dict):
        return {k: resolve_compiled_value(v, scopes) for k, v in value.items()}
    return value
