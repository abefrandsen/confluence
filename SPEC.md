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
    fn: str = ""                                        # Path to callable (e.g. "numpy.mean", "scaler.fit_transform")
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
    outputs: [x_train, x_test, y_train, y_test]

  - name: first_test_label
    fn: int
    args: ["$y_test[0]"]  # Indexing supported (quote bracketed paths inside YAML [...] flow lists)
    outputs: [sample_label]
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
| `[$x, 1, {k: $y.z}]` | Containers are resolved **recursively** (in dicts, only **values** are resolved; keys are left as literals) |

OmegaConf evaluates `${...}` interpolations first during config resolution (Phase 1). Plain `$name` references without braces are ignored by OmegaConf and resolved at runtime (Phase 3).

In a step's `fn` field, the value is always a path to a callable and does **not** use the `$` prefix (e.g., `fn: numpy.mean`, `fn: scaler.fit_transform`, `fn: models[0].forward`).

### 3.2 Scope Lookup Order
When resolving the root segment of a path (e.g., `foo` in `foo.bar[0]`), scopes are searched in this order:

1. **`data`** — the mutable runtime data dict (initial inputs + step outputs)
2. **`functions`** — user-supplied callable/object dict passed at pipeline creation
3. **`imports`** — top-level module bindings created from `imports`
4. **`builtins`** — default Python builtins (see Section 4.2)

Keeping `data` separate from `functions`, `imports`, and `builtins` ensures that returning `data` at the end of execution returns only pipeline data, without polluting it with modules or builtins.

### 3.3 Path Traversal (Dot & Bracket Indexing)
After resolving the root object `obj`, each subsequent segment is resolved in order:

- **Dot segment (`.part`):**
  1. Mapping key lookup (`obj[part]`) if `isinstance(obj, Mapping)` and `part in obj`
  2. Attribute lookup (`getattr(obj, part)`) if `hasattr(obj, part)`
  3. Otherwise raise `ResolutionError` with step context and the failing segment. If `obj` is a Python module (`types.ModuleType`) and `<obj.__name__>.<part>` is a resolvable submodule spec (`importlib.util.find_spec`), the error includes a hint suggesting adding `<obj.__name__>.<part>` to `imports`.
- **Bracket segment (`[idx]`):**
  - Supports integer indices (`[0]`, `[-1]`) and quoted string keys (`["key"]`, `['key']`).
  - Evaluates `obj[idx]`; wraps `KeyError`/`IndexError`/`TypeError` in `ResolutionError` with step context.
  - Slices (`[0:5]`) and nested expressions (`a[b[0]]`) are out of scope in v1.

---

## 4. Imports & Builtins

### 4.1 Imports
- Entries in `imports` are plain dotted module paths (e.g., `"numpy"`, `"torch.nn"`).
- Each entry is loaded via `importlib.import_module(mod_path)` and the **top-level package** is bound in the `imports` scope (matching standard Python `import torch.nn` semantics, so `torch.nn.Linear` resolves naturally).
- No aliasing or `from X import Y` in v1, keeping module references unambiguous.

### 4.2 Builtins & Trust Model
Python builtins are available by default in the lowest-precedence scope, excluding interactive and footgun builtins to keep the namespace clean and guard against accidental misuse:

- **Excluded by default:** `eval`, `exec`, `compile`, `__import__`, `open`, `input`, `breakpoint`, `globals`, `locals`, `vars`, `exit`, `quit`, `help`, `copyright`, `credits`, `license`, `memoryview`
- **Included (examples):** `len`, `sum`, `min`, `max`, `abs`, `round`, `sorted`, `reversed`, `zip`, `enumerate`, `map`, `filter`, `range`, `list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`, `getattr`, `setattr`, `hasattr`, `isinstance`, `type`, `all`, `any`, `print`, etc.

> **Trust model:** Pipeline YAML is executable code (it can import arbitrary modules and traverse Python object attributes) and **must be trusted**. Confluence does not sandbox execution; excluding builtins above is a convenience guardrail against mistakes, not a security boundary.

---

## 5. Output Binding & Return Value

Each step's `outputs` field (`list[str] | None`) controls how the return value `result` is stored in `data`:

| `outputs` Value | Behavior |
| --- | --- |
| `None`, `[]`, or `["_"]` | Return value is discarded without writing to `data` (for side-effecting steps). |
| `["key"]` (`len == 1`) | `data["key"] = result` directly — **never** unpacked, even if `result` is a tuple or list. |
| `["k1", "k2", ...]` (`len > 1`) | Unpacks `result` as a sequence/iterable of exact length `len(outputs)` and binds each element (`data[k] = val`). Rejects `str`, `bytes`, and `Mapping` values (bind a single output key for a mapping and access keys via `$out.key`), and raises `OutputBindingError` on length mismatch or non-unpackable result. |

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
1. **Schema & path syntax:** Every step has a non-empty, syntactically valid `fn` path; every `$ref` path parses cleanly; declared `inputs` and step `outputs` names are valid identifiers (`[A-Za-z_][A-Za-z0-9_]*`).
2. **Import resolution:** All declared `imports` load cleanly (loaded modules are cached on the `Pipeline` instance after the first successful load).
3. **Simulated dataflow** (following the same `data` → `functions` → `imports` → `builtins` lookup order as execution):
   - Initialize `known_keys` from `data.keys()` (if validating with runtime data) and/or `cfg.inputs` (if declared). If both are provided, verify all declared `cfg.inputs` are present in `data`.
   - Emit a `UserWarning` (once per key) whenever a declared input, runtime input, or step output key shadows a name in `functions`, `imports`, or `builtins`.
   - For each step:
     - Verify the root of `fn` and every `$ref` in `args`/`kwargs` exists in `known_keys | functions | imports | builtins`.
     - If `fn`'s root is **not** in `known_keys` (i.e. it resolves to a static scope: `functions`, `imports`, or `builtins`), verify the full path exists and is `callable(...)`.
     - Similarly, for any `$ref` whose root is **not** in `known_keys` (resolving to a static scope), verify the full path resolves cleanly.
     - Add all non-`"_"` output keys to `known_keys`.
4. Collect and report **all** validation errors together with step index and step `name` (if set).

Validation is **not** cached across `run()` calls—it reruns on every `run(data)` by default because caller-supplied `data.keys()` may differ between calls (while module imports and parsed `fn`/`$ref` paths are cached on the `Pipeline` instance). For repeated hot-loop invocations with a known input schema, callers may validate once up front and pass `validate=False` to `run(data, validate=False)` (imports are still loaded once if not already loaded).

### Phase 3 — Execution
1. Shallow-copy the caller's initial `data` dict so the caller's input dict isn't unexpectedly mutated.
2. When `validate=True` (the default), automatically run `validate(input_keys=data.keys())` before executing step 0 (failing fast on missing inputs, bad paths, or invalid static callables). When `validate=False`, ensure declared `imports` are loaded (once) and skip static validation.
3. For each pre-compiled step in `steps`:
   - Resolve pre-parsed `fn` → callable.
   - Recursively resolve pre-parsed `args` and `kwargs`.
   - Invoke `result = fn(*resolved_args, **resolved_kwargs)`.
   - Bind `result` into `data` according to `outputs`.
   - On failure, annotate the exception with step index, step `name`, and `fn` string via `exc.add_note(...)` and re-raise the original exception so callers can catch specific exception types directly (e.g. `torch.cuda.OutOfMemoryError`).
4. Return `data`.

---

## 7. Error Hierarchy

All custom exceptions inherit from `ConfluenceError`:

- `ConfluenceError` — base exception
  - `PathSyntaxError` — malformed dotted/indexed path string
  - `ResolutionError` — undefined root name, missing attribute/key, invalid index, or non-callable target
  - `OutputBindingError` — non-iterable return value or length mismatch during multi-output unpacking
  - `ValidationError` — pre-flight validation failure (`.errors: list[str]` holds all collected messages)

Runtime exceptions raised during step execution are re-raised directly with step context attached via `BaseException.add_note()`.

---

## 8. Python API Sketch

```python
# Construct from YAML, a dict/DictConfig, or a PipelineConfig instance:
pipeline = Pipeline.from_yaml(
    "train.yaml",
    functions={"split_data": split_data},
    overrides=["params.scale_factor=3.0"],
)
# Also available:
# pipeline = Pipeline.from_dict(cfg_dict, functions=..., overrides=...)
# pipeline = Pipeline(config=pipeline_cfg, functions=...)

# Optional standalone validation before running
pipeline.validate(input_keys=["raw_x", "raw_y"])

# Execute (validates automatically; Pipeline is also callable)
result: dict[str, Any] = pipeline.run({"raw_x": x, "raw_y": y})
# or: result = pipeline({"raw_x": x, "raw_y": y})
# or for hot inner loops after validate():
# result = pipeline.run({"raw_x": x, "raw_y": y}, validate=False)
```

---

## 9. Scope Boundaries (v1 vs. Future)

| Feature | v1 Status | Notes |
| --- | --- | --- |
| Linear step execution | **In scope** | Core execution model |
| Classes, methods, functions, default builtins | **In scope** | Unified callable resolution |
| Dotted paths + integer/string indexing | **In scope** | `$a.b[0]["k"]` (no slices or nested expressions) |
| Sub-pipelines via Python | **Free in v1** | Because `Pipeline` is callable, pass a `Pipeline` instance in `functions` |
| Sub-pipelines declared in YAML | **Deferred (v2)** | Easy to add later without changing v1 syntax |
| Control flow (`if`, `for`, `map`) | **Out of scope** | Implement in Python functions |
| Import aliases (`numpy as np`) | **Deferred (v2)** | Plain imports only in v1 |

---

## 10. Optional PyTorch Extension (`confluence.torch`)

`confluence.torch` is an optional subpackage (`pip install "confluence[torch]"`) that keeps core `confluence` free of any `torch` dependency while providing a config-driven `nn.Module` container (`DictModule`).

### 10.1 Structured Config Schema
```python
@dataclass
class ComponentConfig:
    type: str = ""                                     # Registry name, imported symbol, or dotted import path
    args: list[Any] = field(default_factory=list)      # Positional args (supports $ref to earlier components)
    params: dict[str, Any] = field(default_factory=dict)  # Keyword args (supports $ref to earlier components)
    compile: Any = False                               # False/None, True, or dict of torch.compile(**kwargs)
    freeze: bool = False                               # If True, sets requires_grad_(False) on parameters

@dataclass
class ForwardConfig:
    inputs: list[str] = field(default_factory=list)    # Declared forward input names
    steps: list[StepConfig] = field(default_factory=list)

@dataclass
class DictModuleConfig:
    imports: list[str] = field(default_factory=list)   # Shared by both `components` and `forward`
    params: dict[str, Any] = field(default_factory=dict)  # Shared `${params.*}` interpolations
    components: dict[str, ComponentConfig] = field(default_factory=dict)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    outputs: Optional[list[str]] = None                # None -> full state dict, [k] -> state[k], [k1, ...] -> subset dict
```

### 10.2 Pre-flight Validation & Two-Pass Construction
1. **Pre-flight validation** (before instantiating any `nn.Module`): collects all errors into a single `ValidationError`, checking:
   - `imports` load cleanly.
   - All `components` keys are valid identifiers (`[A-Za-z_][A-Za-z0-9_]*`) and do not collide with passed `functions`.
   - Each `type` resolves to a callable via explicit `registry` → global `@register_component` registry (populated after `imports` load) → `imports` → dotted `importlib` lookup (preserving `ModuleNotFoundError` from missing transitive dependencies inside user modules rather than masking them).
   - All `$ref`s in `args` and `params` parse cleanly, reference earlier components or static scopes (`registry | imports | builtins`), and resolve fully when targeting static scopes.
   - `compile` is `bool`, `None`, or a `Mapping`.
2. **Pass 1 (ordered instantiation):** Instantiates components in declaration order (`cls(*args, **params)`), attaching `While building component '<name>' [type='<type>']` via `exc.add_note(...)` on failure. Verifies each result is an `nn.Module`, applies `freeze` (`requires_grad_(False)` only; `model.train()` still toggles train/eval mode), and registers the **raw unwrapped** modules in `self.components` (`nn.ModuleDict`). Note: because every entry in `components` is registered in `self.components`, passing an earlier `nn.Module` instance into a later component (e.g. `torch.nn.Sequential`) will register shared weights under both keys in `state_dict()` (aliasing tensors, which breaks serializers like `safetensors`); referencing scalar attributes (e.g. `$embedder.out_dim`) has no aliasing issue.
3. **Forward wiring & validation:** Constructs `self.forward_pipe` with top-level `imports`, `forward.inputs`, `forward.steps`, and `{**functions, **self.components}`, validates it once, and verifies any declared `outputs` keys are produced by `forward.inputs` or `forward.steps`.
4. **Pass 2 (compilation):** Wraps configured components with `torch.compile(...)` **only inside `self.forward_pipe.functions`**, leaving `self.components` unwrapped so `state_dict()` keys never acquire `_orig_mod.` prefixes. `__getstate__` / `__setstate__` strip compiled wrappers before pickling/deepcopying and re-apply them on restore.

### 10.3 Forward Calling Rules & Outputs
- **`forward(*args, **kwargs)`:** Standard Python positional/keyword binding against `forward.inputs` (positional arguments always bind in order, even when an argument is a `dict`). Raises `TypeError` on unexpected keyword arguments, duplicate bindings, too many positional arguments, or missing required inputs.
- **`forward_dict(batch)`:** Explicit helper for passing a batch `Mapping` (ignores extra unused keys such as labels, while still raising `TypeError` on missing required inputs).
- **Execution & return value:** Runs `self.forward_pipe.run(inputs, validate=False)`. Returns the full `state` dict when `outputs` is `None`, `state[k]` when `len(outputs) == 1`, or `{k: state[k] for k in outputs}` when `len(outputs) > 1`.
