# Confluence

A minimal, domain-agnostic Python library for defining and running modular data pipelines in YAML.

Confluence treats data, objects, classes, methods, functions, and imported modules uniformly as resolvable Python values, powered by [OmegaConf](https://omegaconf.readthedocs.io/) for config schema validation, `${...}` interpolation, custom resolvers, and overrides.

## Installation

Requires Python 3.12+.

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Quick Start

Define a pipeline in YAML (`pipeline.yaml`):

```yaml
imports:
  - math

params:
  factor: 2.0

inputs: [raw]

steps:
  - name: create_scaler
    fn: LinearScaler
    kwargs:
      factor: "${params.factor}"
    outputs: [scaler]

  - name: scale
    fn: scaler.transform
    args: [$raw]
    outputs: [scaled]

  - name: sqrt_first
    fn: math.sqrt
    args: ["$scaled[0]"]
    outputs: [first_root]
```

Load and run it from Python:

```python
from confluence import Pipeline

class LinearScaler:
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def transform(self, values: list[float]) -> list[float]:
        return [v * self.factor for v in values]

pipeline = Pipeline.from_yaml(
    "pipeline.yaml",
    functions={"LinearScaler": LinearScaler},
    overrides=["params.factor=4.0"],
)

result = pipeline.run({"raw": [1.0, 2.0, 3.0]})
# result["scaled"] == [4.0, 8.0, 12.0]
# result["first_root"] == 2.0
```

## User Guide

### References & Literals
In `args` and `kwargs`, values are passed inline as literals unless prefixed with `$`:

- `$x` — look up `x` in runtime scopes (valid unquoted in YAML)
- `$a.b.c` — traverse mapping keys (checked first) and object attributes
- `$items[0]`, `$items[-1]`, `$cfg["my.key"]` — integer and quoted string indexing (quote bracketed paths inside YAML `[...]` flow lists, e.g. `args: ["$items[0]"]`)
- `$$literal` — escape a literal string starting with `$` → `"$literal"`
- Nested lists and dicts are resolved recursively.

A step's `fn` field is always a callable path and does **not** use `$` (e.g. `fn: math.sqrt`, `fn: scaler.transform`, `fn: model`).

### Scope Lookup Order
When resolving a root name, Confluence searches in order:
1. **`data`** — initial inputs and outputs bound by prior steps
2. **`functions`** — callables/objects passed to `Pipeline(..., functions=...)`
3. **`imports`** — top-level packages loaded from `imports` (e.g. `torch.nn` binds `torch`)
4. **`builtins`** — safe Python builtins (`len`, `sum`, `map`, `list`, etc.; dangerous builtins like `eval`, `exec`, `open`, `__import__` are excluded)

`pipeline.run(data)` shallow-copies the input dict and returns the resulting `data` dict without leaking `functions`, `imports`, or `builtins`.

### Output Binding
- Omitted / `[]` — discard return value (for side-effecting steps)
- `[key]` — bind return value directly without unpacking (even if it is a tuple/list)
- `[k1, k2, ...]` — unpack iterable of matching length; use `"_"` to discard an element
- Overwriting existing keys in `data` is allowed.

### OmegaConf Integration
- Put shared config values under `params:` and reference them with `${params.*}`.
- Pass dotlist overrides via `Pipeline.from_yaml(..., overrides=["params.lr=0.01"])`.
- Any custom resolvers registered via `OmegaConf.register_new_resolver(...)` work automatically.

### Pre-flight Validation & Sub-pipelines
- Call `pipeline.validate()` (optionally with `input_keys=[...]`) to verify imports, path syntax, dataflow references, and static callables before running. `run()` also validates automatically before step 0.
- `Pipeline` instances are callable (`pipeline(data)`), so you can compose sub-pipelines by passing a `Pipeline` in `functions`.

## Development

Run the test suite:

```bash
uv sync
uv run pytest
```

For full design details and edge-case semantics, see [SPEC.md](SPEC.md).
