# Confluence — Data Pipeline Specification (v1)

## 1. Overview & Goals

Confluence is a domain-agnostic, config-driven system for composing arbitrary Python callables into linear data pipelines defined in YAML.

**Core Principles**
- **Unified value model:** Data, objects, classes, methods, functions, and modules are all treated uniformly as Python values resolvable by name, dotted path, and indexing.
- **Clean config/runtime boundary:** [OmegaConf](https://omegaconf.readthedocs.io/) handles YAML loading, merging, `${...}` interpolation, custom resolvers, and schema validation. Pipeline execution operates on plain Python objects in a standard `dict[str, Any]`.
- **Linear & minimal:** Control flow (`if`, `for`, etc.) lives in Python functions, not YAML.
- **Fail fast:** Schema and static dataflow validation run before any step executes.

---

## 2. YAML Schema & Structured Config

Pipelines are validated against OmegaConf structured configs (Python dataclasses) before execution.

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class StepConfig:
    fn: str                                             # Path to callable (e.g. "numpy.mean", "scaler.fit_transform")
    name: Optional[str] = None                          # Optional human-readable label for logs/errors
    args: list[Any] = field(default_factory=list)       # Positional arguments
    kwargs: dict[str, Any] = field(default_factory=dict)  # Keyword arguments
    outputs: Optional[list[str]] = None                 # None/[] to discard, [k] to bind, [k1, k2, ...] to unpack

@dataclass
class PipelineConfig:
    imports: list[str] = field(default_factory=list)       # Dotted module names (e.g. "numpy", "torch.nn")
    inputs: Optional[list[str]] = None                     # Optional declared initial data keys (for static validation)
    params: dict[str, Any] = field(default_factory=dict)   # Free-form config values for OmegaConf ${params.*} interpolation
    steps: list[StepConfig] = field(default_factory=list)  # Ordered list of steps
```

### Example Pipeline

```yaml
imports:
  - numpy
  - sklearn.preprocessing

params:
  scale_factor: 2.0
  test_size: 0.2

inputs: [raw_x, raw_y]

steps:
  - name: scale_features
    fn: numpy.multiply
    args: [$raw_x, "${params.scale_factor}"]
    outputs: [scaled_x]

  - name: init_scaler
    fn: sklearn.preprocessing.StandardScaler
    outputs: [scaler]

  - name: fit_transform
    fn: scaler.fit_transform
    args: [$scaled_x]
    outputs: [scaled_x]  # Overwriting an existing key is allowed

  - name: split
    fn: split_data       # Provided via the functions dict
    kwargs:
      x: $scaled_x
      y: $raw_y
      test_size: "${params.test_size}"
    outputs: [x_train, x_val, x_test, "_"]  # "_" discards an unpacked value

  - name: first_val_sample
    fn: numpy.mean
    args: ["$x_val[0]"]  # Indexing supported (quote bracketed paths inside YAML [...] flow lists)
    outputs: [sample_mean]
```

---

## 3. Reference Syntax & Value Resolution

### 3.1 Literals vs. References
In `args` and `kwargs`, YAML primitives (`int`, `float`, `bool`, `None`, `str`) and containers (`list`, `dict`) are passed inline as literals, **except** strings beginning with `$`:

| YAML Value | Meaning |
| --- | --- |
| `42`, `3.14`, `true`, `null` | Inline literal |
| `"hello"` / `hello` | Literal string `"hello"` |
| `$x` | Lookup `"x"` in the runtime scopes (valid unquoted in YAML) |
| `$a.b.c` | Lookup `"a"`, then traverse `.b`, then `.c` |
| `$items[0]`, `$items[-1]` | Integer indexing into a sequence/mapping |
| `$cfg["my.key"]` | Quoted string key lookup (useful for keys with dots/spaces) |
| `$prep.batches[0].values[-1]` | Arbitrary chaining of dots and brackets |
| `$$literal` | Escaped literal string `"$literal"` (leading `$$` unescapes to `$`) |
| `[$x, 1, {k: $y.z}]` | Containers are resolved **recursively** |

OmegaConf evaluates `${...}` interpolations first during config resolution (Phase 1). Plain `$name` references without braces are ignored by OmegaConf and resolved at runtime (Phase 3).

In a step's `fn` field, the value is always a path to a callable and does **not** use the `$` prefix (e.g., `fn: numpy.mean`, `fn: scaler.fit_transform`, `fn: models[0].forward`).

### 3.2 Scope Lookup Order
When resolving the root segment of a path (e.g., `foo` in `foo.bar[0]`), scopes are searched in this order:

1. **`data`** — the mutable runtime data dict (initial inputs + step outputs)
2. **`functions`** — user-supplied callable/object dict passed at pipeline creation/execution
3. **`imports`** — top-level module bindings created from `imports`
4. **`builtins`** — safe Python builtins

Keeping `data` separate from `functions`, `imports`, and `builtins` ensures that returning `data` at the end of execution returns only pipeline data, without polluting it with modules or builtins.

### 3.3 Path Traversal (Dot & Bracket Indexing)
After resolving the root object `obj`, each subsequent segment is resolved in order:

- **Dot segment (`.part`):**
  1. Mapping key lookup (`obj[part]`) if `isinstance(obj, Mapping)` and `part in obj`
  2. Attribute lookup (`getattr(obj, part)`) if `hasattr(obj, part)`
  3. Otherwise raise `ResolutionError` with step context and the failing segment.
- **Bracket segment (`[idx]`):**
  - Supports integer indices (`[0]`, `[-1]`) and quoted string keys (`["key"]`, `['key']`).
  - Evaluates `obj[idx]`; wraps `KeyError`/`IndexError`/`TypeError` in `ResolutionError` with step context.
  - Slices (`[0:5]`) and nested expressions (`a[b[0]]`) are out of scope in v1.

---

## 4. Imports & Safe Builtins

### 4.1 Imports
- Entries in `imports` are plain dotted module paths (e.g., `"numpy"`, `"torch.nn"`).
- Each entry is loaded via `importlib.import_module(mod_path)` and the **top-level package** is bound in the `imports` scope (matching standard Python `import torch.nn` semantics, so `torch.nn.Linear` resolves naturally).
- No aliasing or `from X import Y` in v1, keeping module references unambiguous.

### 4.2 Builtins
Python builtins are available by default in the lowest-precedence scope, **excluding** dangerous/interactive builtins:

- **Excluded:** `eval`, `exec`, `compile`, `__import__`, `open`, `input`, `breakpoint`, `globals`, `locals`, `vars`, `exit`, `quit`, `help`, `copyright`, `credits`, `license`, `memoryview`
- **Included (examples):** `len`, `sum`, `min`, `max`, `abs`, `round`, `sorted`, `reversed`, `zip`, `enumerate`, `map`, `filter`, `range`, `list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`, `getattr`, `setattr`, `hasattr`, `isinstance`, `type`, `all`, `any`, `print`, etc.

---

## 5. Output Binding & Return Value

Each step's `outputs` field (`list[str] | None`) controls how the return value `result` is stored in `data`:

| `outputs` Value | Behavior |
| --- | --- |
| `None` or `[]` | Return value is discarded (for side-effecting steps). |
| `["key"]` (`len == 1`) | `data["key"] = result` directly — **never** unpacked, even if `result` is a tuple or list. |
| `["k1", "k2", ...]` (`len > 1`) | Unpacks `result` as a sequence/iterable of exact length `len(outputs)` and binds each element (`data[k] = val`). Raises `OutputBindingError` on length mismatch or non-iterable result. |

- **Discard placeholder:** Binding to `"_"` discards that value without writing to `data`.
- **Overwriting:** Writing to an existing key in `data` is allowed and overwrites the previous value.
- **Pipeline return value:** Executing a pipeline returns the full `data` dict (`dict[str, Any]`).

---

## 6. Lifecycle: Load, Validate, Execute

### Phase 1 — Config Resolution (OmegaConf)
1. Load YAML (and optional CLI/dict overrides) with OmegaConf.
2. Merge with `PipelineConfig` structured config schema.
3. Resolve all `${...}` interpolations and custom resolvers (`OmegaConf.to_container(cfg, resolve=True)`).

### Phase 2 — Static Validation (Pre-flight)
Before executing step 0, validate:
1. **Schema & path syntax:** Every step has a non-empty, syntactically valid `fn` path; every `$ref` path parses cleanly; `outputs` names are non-empty strings.
2. **Import resolution:** All declared `imports` load cleanly.
3. **Simulated dataflow:**
   - Initialize `known_keys` from `data.keys()` (if validating with runtime data) and/or `cfg.inputs` (if declared). If both are provided, verify all declared `cfg.inputs` are present in `data`.
   - For each step:
     - Verify the root of `fn` and every `$ref` in `args`/`kwargs` exists in `known_keys | functions | imports | builtins`.
     - If `fn` resolves to a static scope (`functions`, `imports`, `builtins`), verify the full path exists and is `callable(...)`.
     - Add all non-`"_"` output keys to `known_keys`.
4. Collect and report **all** validation errors together with step index and step `name` (if set).

### Phase 3 — Execution
1. Shallow-copy the caller's initial `data` dict so the caller's input dict isn't unexpectedly mutated.
2. For each step in `steps`:
   - Resolve `fn` → callable.
   - Recursively resolve `args` and `kwargs`.
   - Invoke `result = fn(*resolved_args, **resolved_kwargs)`.
   - Bind `result` into `data` according to `outputs`.
   - On failure, wrap/annotate the error with step index, step `name`, and `fn` string.
3. Return `data`.

---

## 7. Python API Sketch

```python
pipeline = Pipeline.from_yaml(
    "train.yaml",
    functions={"split_data": split_data},
    overrides=["params.scale_factor=3.0"],
)

# Optional standalone validation before running
pipeline.validate(input_keys=["raw_x", "raw_y"])

# Execute (Pipeline is also callable)
result: dict[str, Any] = pipeline.run({"raw_x": x, "raw_y": y})
# or: result = pipeline({"raw_x": x, "raw_y": y})
```

---

## 8. Scope Boundaries (v1 vs. Future)

| Feature | v1 Status | Notes |
| --- | --- | --- |
| Linear step execution | **In scope** | Core execution model |
| Classes, methods, functions, safe builtins | **In scope** | Unified callable resolution |
| Dotted paths + integer/string indexing | **In scope** | `$a.b[0]["k"]` (no slices or nested expressions) |
| Sub-pipelines via Python | **Free in v1** | Because `Pipeline` is callable, pass a `Pipeline` instance in `functions` |
| Sub-pipelines declared in YAML | **Deferred (v2)** | Easy to add later without changing v1 syntax |
| Control flow (`if`, `for`, `map`) | **Out of scope** | Implement in Python functions |
| Import aliases (`numpy as np`) | **Deferred (v2)** | Plain imports only in v1 |
