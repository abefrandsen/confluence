# `confluence.torch`

Optional PyTorch utilities for defining config-driven `nn.Module` architectures (`DictModule`) using an ordered `components` map and a Confluence `forward` pipeline.

## Installation

Install Confluence with the optional `torch` extra:

```bash
uv sync --extra torch
```

Or with `pip`:

```bash
pip install -e ".[torch]"
```

## Quick Start

Define a model in YAML (`model.yaml`):

```yaml
imports:
  - torch.nn.functional                 # shared by both `components` and `forward`

params:
  in_dim: 8
  hidden_dim: 32
  num_classes: 3

components:
  embedder:
    type: BoardEmbedder                 # resolved via `registry`, `imports`, or dotted import path
    params:
      in_dim: "${params.in_dim}"
      hidden_dim: "${params.hidden_dim}"
    compile: true                       # optional: bool or dict of torch.compile(**kwargs)
    freeze: false                       # optional: sets requires_grad_(False) on parameters

  head:
    type: torch.nn.Linear
    params:
      in_features: $embedder.out_dim    # $ref to earlier component or its attributes
      out_features: "${params.num_classes}"

forward:
  inputs: [x]
  steps:
    - name: embed
      fn: embedder
      args: [$x]
      outputs: [h]

    - name: activate
      fn: torch.nn.functional.gelu
      args: [$h]
      outputs: [h]

    - name: classify
      fn: head
      args: [$h]
      outputs: [logits]

outputs: [logits]                       # omit for full state dict, [k] for single tensor, [k1, k2] for dict
```

Load and run it from Python:

```python
import torch
import torch.nn as nn
from confluence.torch import DictModule, register_component

@register_component
class BoardEmbedder(nn.Module):
    ...

model = DictModule.from_yaml(
    "model.yaml",
    overrides=[
        "params.hidden_dim=64",
        "components.embedder.compile=false",
    ],
)

x = torch.randn(4, 8)
logits = model(x)                       # positional (binds to `forward.inputs` in order)
logits = model(x=x)                     # keyword arguments
logits = model.forward_dict({"x": x, "label": 0})  # unpack a batch mapping (ignores extra keys)
```

You can also pass a custom name (`@register_component("embedder")`), overwrite an existing registration (`@register_component(replace=True)`), or pass an explicit `registry={"BoardEmbedder": BoardEmbedder}` to `DictModule.from_yaml(...)` (explicit `registry` entries take precedence over the global registry). If a decorated class lives in another module, listing that module in top-level `imports:` ensures the decorator runs before components are resolved.

## Config Structure

### Top-Level (`DictModuleConfig`)
| Field | Type | Description |
| --- | --- | --- |
| `imports` | `list[str]` | Modules imported once and shared by both `components` and `forward`. |
| `params` | `dict[str, Any]` | Shared values available for OmegaConf `${params.*}` interpolation across `components` and `forward`. |
| `components` | `dict[str, ComponentConfig]` | Ordered mapping of component names to module specs. Instantiated in declaration order and registered (unwrapped) in `self.components` (`nn.ModuleDict`). |
| `forward` | `ForwardConfig` | `inputs` + `steps` executed via a Confluence `Pipeline` with `validate=False` (validated once in `__init__`). |
| `outputs` | `list[str] \| None` | `None` returns the full pipeline state dict; `[k]` returns `state[k]`; `[k1, k2, ...]` returns `{k1: state[k1], ...}`. |

### Component Spec (`ComponentConfig`)
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | `str` | `""` | Registry name, imported symbol, or dotted import path (e.g. `torch.nn.Linear`). |
| `args` | `list[Any]` | `[]` | Positional arguments passed to `type(*args, **params)` (e.g. `[8, 32]` for `torch.nn.Linear`). Supports `$ref` to earlier components or their attributes. |
| `params` | `dict[str, Any]` | `{}` | Keyword arguments passed to `type(*args, **params)`. Supports `$ref` to earlier components (e.g. `$embedder.out_dim`). |
| `compile` | `bool \| dict` | `false` | If `true`, wraps the component with `torch.compile(mod)`; if a mapping, calls `torch.compile(mod, **compile)`. Compiled wrappers live only in `forward_pipe.functions`, so `self.components` and `state_dict()` keys stay clean (no `_orig_mod.` prefix) regardless of whether `compile` is enabled. |
| `freeze` | `bool` | `false` | If `true`, sets `requires_grad_(False)` on all parameters of the component. Note: this only freezes gradients—calling `model.train()` will still toggle train/eval mode (e.g. Dropout / BatchNorm running stats) unless you call `.eval()` on the component yourself. |

> **Note on referencing earlier components:** Every entry in `components` is registered in `self.components` (`nn.ModuleDict`). Referencing scalar attributes (e.g. `$embedder.out_dim`) is completely safe, but passing an earlier `nn.Module` *instance* (e.g. `args: [$layer1, $layer2]` into `torch.nn.Sequential`) registers the same module in two places—which duplicates keys in `state_dict()` and can break serializers that reject aliased tensors (such as `safetensors`). Prefer chaining layers in `forward.steps` or building composite sub-modules in Python.
