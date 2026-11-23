"""Tests for `confluence.torch` config schema and `DictModule`."""

from __future__ import annotations

import copy
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from omegaconf.errors import ConfigKeyError

import confluence.torch as confluence_torch_pkg
from confluence import ValidationError
from confluence.torch import (
    ComponentConfig,
    DictModuleConfig,
    ForwardConfig,
    load_dict_module_config,
    resolve_dict_module_config,
)


def test_dict_module_config_yaml_overrides_and_order(tmp_path: Path) -> None:
    yaml_content = """
imports:
  - math

params:
  in_dim: 8
  hidden_dim: 32
  num_classes: 3

components:
  embedder:
    type: BoardEmbedder
    params:
      in_dim: "${params.in_dim}"
      hidden_dim: "${params.hidden_dim}"
    compile: true
  head:
    type: torch.nn.Linear
    args: [$embedder.out_dim, "${params.num_classes}"]
    compile:
      mode: reduce-overhead
      fullgraph: true
    freeze: true

forward:
  inputs: [x]
  steps:
    - fn: embedder
      args: [$x]
      outputs: [h]
    - fn: head
      args: [$h]
      outputs: [logits]

outputs: [logits]
"""
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(yaml_content)

    cfg = load_dict_module_config(
        cfg_path,
        overrides=[
            "params.hidden_dim=64",
            "components.embedder.compile=false",
        ],
    )
    assert isinstance(cfg, DictModuleConfig)
    assert isinstance(cfg.forward, ForwardConfig)
    assert cfg.imports == ["math"]
    assert list(cfg.components.keys()) == ["embedder", "head"]

    embedder = cfg.components["embedder"]
    assert isinstance(embedder, ComponentConfig)
    assert embedder.type == "BoardEmbedder"
    assert embedder.args == []
    assert embedder.params == {"in_dim": 8, "hidden_dim": 64}
    assert embedder.compile is False
    assert embedder.freeze is False

    head = cfg.components["head"]
    assert head.type == "torch.nn.Linear"
    assert head.args == ["$embedder.out_dim", 3]
    assert head.params == {}
    assert head.compile == {"mode": "reduce-overhead", "fullgraph": True}
    assert head.freeze is True

    assert cfg.forward.inputs == ["x"]
    assert len(cfg.forward.steps) == 2
    assert cfg.outputs == ["logits"]


def test_dict_module_config_rejects_unknown_and_duplicate_forward_keys() -> None:
    with pytest.raises(ConfigKeyError):
        resolve_dict_module_config({"unknown_key": 1})

    with pytest.raises(ConfigKeyError):
        resolve_dict_module_config(
            {
                "components": {
                    "enc": {"type": "Foo", "not_a_field": True},
                }
            }
        )

    # `forward.imports` and `forward.params` are rejected in favor of top-level `imports` / `params`
    with pytest.raises(ConfigKeyError):
        resolve_dict_module_config({"forward": {"imports": ["math"], "steps": []}})

    with pytest.raises(ConfigKeyError):
        resolve_dict_module_config({"forward": {"params": {"a": 1}, "steps": []}})


def test_import_dict_module_without_torch_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "confluence.torch.module", raising=False)

    with pytest.raises(ImportError, match="confluence\\[torch\\]"):
        importlib.import_module("confluence.torch.module")


@pytest.fixture
def fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Install a minimal fake `torch` / `torch.nn` module and clean up `confluence.torch.module`."""
    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    class FakeParam:
        def __init__(self) -> None:
            self.requires_grad = True

        def requires_grad_(self, mode: bool) -> FakeParam:
            self.requires_grad = mode
            return self

    class FakeModule:
        def __init__(self) -> None:
            self._params: list[FakeParam] = [FakeParam()]

        def parameters(self) -> list[FakeParam]:
            return self._params

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    class FakeModuleDict(dict, FakeModule):
        def __init__(self, modules: dict[str, FakeModule] | None = None) -> None:
            FakeModule.__init__(self)
            dict.__init__(self, modules or {})

    class FakeLinear(FakeModule):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x: float) -> float:
            return x * self.in_features + self.out_features

    class FakeSequential(FakeModule):
        def __init__(self, *layers: FakeModule) -> None:
            super().__init__()
            self.layers = layers

        def forward(self, x: float) -> float:
            for layer in self.layers:
                x = layer(x)
            return x

    class FakeCompiledWrapper:
        """Distinct wrapper object (like torch._dynamo OptimizedModule) not stored in ModuleDict."""

        def __init__(self, orig_mod: FakeModule, kwargs: dict[str, Any]) -> None:
            self._orig_mod = orig_mod
            self.kwargs = kwargs

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._orig_mod(*args, **kwargs)

    def fake_compile(mod: Any, **kwargs: Any) -> Any:
        compile_calls.append((mod, kwargs))
        return FakeCompiledWrapper(mod, kwargs)

    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = FakeModule  # type: ignore[attr-defined]
    fake_nn.ModuleDict = FakeModuleDict  # type: ignore[attr-defined]
    fake_nn.Linear = FakeLinear  # type: ignore[attr-defined]
    fake_nn.Sequential = FakeSequential  # type: ignore[attr-defined]

    fake_torch = types.ModuleType("torch")
    fake_torch.nn = fake_nn  # type: ignore[attr-defined]
    fake_torch.compile = fake_compile  # type: ignore[attr-defined]
    fake_torch._FakeCompiledWrapper = FakeCompiledWrapper  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    sys.modules.pop("confluence.torch.module", None)
    mod = importlib.import_module("confluence.torch.module")
    try:
        yield mod, compile_calls
    finally:
        sys.modules.pop("confluence.torch.module", None)
        if hasattr(confluence_torch_pkg, "module"):
            delattr(confluence_torch_pkg, "module")


def test_dict_module_ordered_refs_args_freeze_compile_and_calling_modes(
    fake_torch: tuple[Any, list[tuple[Any, dict[str, Any]]]],
) -> None:
    torch_mod, compile_calls = fake_torch
    DictModule = torch_mod.DictModule
    FakeModule = sys.modules["torch.nn"].Module
    FakeCompiledWrapper = sys.modules["torch"]._FakeCompiledWrapper

    class FakeEncoder(FakeModule):
        def __init__(self, scale: int) -> None:
            super().__init__()
            self.scale = scale
            self.out_dim = scale * 2

        def forward(self, x: float) -> float:
            return x * self.scale

    model = DictModule.from_dict(
        {
            "components": {
                "enc": {
                    "type": "FakeEncoder",
                    "params": {"scale": 3},
                    "freeze": True,
                    "compile": True,
                },
                "head": {
                    "type": "torch.nn.Linear",
                    "params": {"in_features": "$enc.out_dim", "out_features": 1},
                    "compile": {"mode": "reduce-overhead"},
                },
                "seq": {
                    "type": "torch.nn.Sequential",
                    "args": ["$enc", "$head"],
                },
            },
            "forward": {
                "inputs": ["x"],
                "steps": [
                    {"fn": "seq", "args": ["$x"], "outputs": ["logits"]},
                ],
            },
            "outputs": ["logits"],
        },
        registry={"FakeEncoder": FakeEncoder},
    )

    # Ordered $ref resolution worked: head.in_features == enc.out_dim == 6
    assert model.components["head"].in_features == 6
    # Freeze applied only to enc
    assert all(not p.requires_grad for p in model.components["enc"].parameters())
    assert all(p.requires_grad for p in model.components["head"].parameters())
    # self.components holds RAW unwrapped modules (no _orig_mod. prefix in state_dict)
    assert isinstance(model.components["enc"], FakeEncoder)
    assert not isinstance(model.components["enc"], FakeCompiledWrapper)
    assert not isinstance(model.components["head"], FakeCompiledWrapper)
    # Compiled wrappers live in forward_pipe.functions only
    assert isinstance(model.forward_pipe.functions["enc"], FakeCompiledWrapper)
    assert isinstance(model.forward_pipe.functions["head"], FakeCompiledWrapper)
    assert len(compile_calls) == 2
    assert compile_calls[0][1] == {}
    assert compile_calls[1][1] == {"mode": "reduce-overhead"}

    # Calling modes: positional, kwargs, and explicit forward_dict all return (2 * 3) * 6 + 1 == 37
    assert model(2) == 37
    assert model(x=2) == 37
    assert model.forward_dict({"x": 2, "label": 99}) == 37


def test_dict_module_validation_and_error_handling(
    fake_torch: tuple[Any, list[tuple[Any, dict[str, Any]]]],
) -> None:
    torch_mod, _ = fake_torch
    DictModule = torch_mod.DictModule

    base_cfg: dict[str, Any] = {
        "components": {
            "layer": {
                "type": "torch.nn.Linear",
                "params": {"in_features": 2, "out_features": 5},
            }
        },
        "forward": {
            "inputs": ["x"],
            "steps": [
                {"fn": "layer", "args": ["$x"], "outputs": ["y"]},
            ],
        },
    }

    # outputs=None -> returns full state dict (forward_dict does not leak unused batch keys)
    m_all = DictModule.from_dict(base_cfg)
    assert m_all(x=3) == {"x": 3, "y": 11}
    assert m_all.forward_dict({"x": 3, "label": 99}) == {"x": 3, "y": 11}

    # multi-output -> returns subset dict
    m_multi = DictModule.from_dict({**base_cfg, "outputs": ["x", "y"]})
    assert m_multi(3) == {"x": 3, "y": 11}

    # empty or duplicate outputs fail fast in pre-flight validation
    with pytest.raises(ValidationError, match="must be non-empty"):
        DictModule.from_dict({**base_cfg, "outputs": []})
    with pytest.raises(ValidationError, match="duplicate key 'y'"):
        DictModule.from_dict({**base_cfg, "outputs": ["y", "y"]})

    # undeclared output key or broken forward step fails fast BEFORE instantiating components
    instantiated = False

    class TrackingModule(sys.modules["torch.nn"].Module):
        def __init__(self) -> None:
            nonlocal instantiated
            super().__init__()
            instantiated = True

    with pytest.raises(ValidationError) as pre_exc:
        DictModule.from_dict(
            {
                "components": {"heavy": {"type": "TrackingModule"}},
                "forward": {
                    "inputs": ["x"],
                    "steps": [
                        {"fn": "heavy_typo", "args": ["$x"], "outputs": ["y"]},
                    ],
                },
                "outputs": ["missing_key"],
            },
            registry={"TrackingModule": TrackingModule},
        )
    assert not instantiated
    assert "heavy_typo" in str(pre_exc.value)
    assert "missing_key" in str(pre_exc.value)

    # component name colliding with passed `functions` raises ValidationError
    with pytest.raises(ValidationError, match="conflict with keys in 'functions'"):
        DictModule.from_dict(base_cfg, functions={"layer": lambda x: x})

    # component name colliding with nn.ModuleDict attributes fails in pre-flight validation
    with pytest.raises(ValidationError, match="conflicts with an existing nn.ModuleDict attribute"):
        DictModule.from_dict(
            {
                "components": {
                    "keys": {
                        "type": "torch.nn.Linear",
                        "params": {"in_features": 2, "out_features": 2},
                    }
                },
                "forward": {"steps": []},
            }
        )

    # pre-flight validation collects all component errors together without instantiating
    with pytest.raises(ValidationError) as exc_info:
        DictModule.from_dict(
            {
                "imports": ["math"],
                "components": {
                    "my-layer": {
                        "type": "DoesNotExist",
                        "args": ["$missing_pos"],
                        "params": {"a": "$missing_var", "b": "$math.nope"},
                        "compile": 123,
                    }
                },
                "forward": {"steps": []},
            }
        )
    msg = str(exc_info.value)
    assert "Component 'my-layer': invalid component name" in msg
    assert "DoesNotExist" in msg
    assert "missing_pos" in msg
    assert "missing_var" in msg
    assert "Cannot resolve segment 'nope'" in msg
    assert "'compile' must be a bool or mapping" in msg

    # instantiation-time errors preserve original exception type with component context note
    with pytest.raises(TypeError) as type_exc:
        DictModule.from_dict(
            {
                "components": {
                    "bad_kwargs": {
                        "type": "torch.nn.Linear",
                        "params": {"not_a_real_arg": 1},
                    }
                },
                "forward": {"steps": []},
            }
        )
    assert any(
        "bad_kwargs" in note and "torch.nn.Linear" in note for note in type_exc.value.__notes__
    )

    # positional/kwarg/missing-input errors
    with pytest.raises(TypeError, match="Expected at most 1"):
        m_multi(1, 2)
    with pytest.raises(TypeError, match="multiple values"):
        m_multi(1, x=2)
    with pytest.raises(TypeError, match="missing required input"):
        m_multi()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        m_multi(x=1, bogus=2)
    with pytest.raises(TypeError, match="missing required input"):
        m_multi.forward_dict({"other": 1})

    # zero-input model (omitted forward.inputs -> []) rejects unexpected kwargs and positional args
    m_zero = DictModule.from_dict({"forward": {"steps": [{"fn": "abs", "args": [-5], "outputs": ["v"]}]}, "outputs": ["v"]})
    assert m_zero() == 5
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        m_zero(unexpected=123)
    with pytest.raises(TypeError, match="Expected at most 0"):
        m_zero(123)


def test_dict_module_from_yaml_and_unambiguous_dict_positional_arg(
    tmp_path: Path,
    fake_torch: tuple[Any, list[tuple[Any, dict[str, Any]]]],
) -> None:
    torch_mod, _ = fake_torch
    DictModule = torch_mod.DictModule

    cfg_path = tmp_path / "dict_in.yaml"
    cfg_path.write_text(
        """
params:
  out_dim: 4

components:
  layer:
    type: torch.nn.Linear
    params:
      in_features: 2
      out_features: "${params.out_dim}"

forward:
  inputs: [batch]
  steps:
    - fn: layer
      args: [$batch.val]
      outputs: [y]

outputs: [y]
"""
    )

    model = DictModule.from_yaml(cfg_path, overrides=["params.out_dim=10"])
    # Positional dict ALWAYS binds to `batch` — even if it happens to contain a key named "batch"!
    assert model({"val": 3, "batch": "unrelated"}) == 3 * 2 + 10
    # Use forward_dict() to unpack a mapping of input names
    assert model.forward_dict({"batch": {"val": 3}, "extra": 1}) == 3 * 2 + 10


def test_dict_module_with_imports_and_compile_supports_deepcopy(
    fake_torch: tuple[Any, list[tuple[Any, dict[str, Any]]]],
) -> None:
    torch_mod, compile_calls = fake_torch
    DictModule = torch_mod.DictModule
    FakeCompiledWrapper = sys.modules["torch"]._FakeCompiledWrapper

    model = DictModule.from_dict(
        {
            "imports": ["math"],
            "components": {
                "layer": {
                    "type": "torch.nn.Linear",
                    "params": {"in_features": 2, "out_features": 1},
                    "compile": True,
                }
            },
            "forward": {
                "inputs": ["x"],
                "steps": [
                    {"fn": "layer", "args": ["$x"], "outputs": ["h"]},
                    {"fn": "math.sqrt", "args": ["$h"], "outputs": ["y"]},
                ],
            },
            "outputs": ["y"],
        }
    )
    assert model(4) == 3.0  # sqrt(4 * 2 + 1)
    assert len(compile_calls) == 1

    copied = copy.deepcopy(model)
    assert copied.components["layer"] is not model.components["layer"]
    # __setstate__ re-compiled the copied raw module (not the original)
    assert len(compile_calls) == 2
    assert isinstance(copied.forward_pipe.functions["layer"], FakeCompiledWrapper)
    assert copied.forward_pipe.functions["layer"]._orig_mod is copied.components["layer"]

    # Mutating copied component affects copied forward pass, not the original
    copied.components["layer"].out_features = 8
    assert copied(4) == 4.0  # sqrt(4 * 2 + 8)
    assert model(4) == 3.0


def test_import_dotted_target_reraises_transitive_missing_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: tuple[Any, list[tuple[Any, dict[str, Any]]]],
) -> None:
    torch_mod, _ = fake_torch
    DictModule = torch_mod.DictModule

    # Create a real temporary module whose body imports a missing dependency
    broken_mod = tmp_path / "broken_user_mod.py"
    broken_mod.write_text("import definitely_missing_dep_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError, match="definitely_missing_dep_xyz"):
        DictModule.from_dict(
            {
                "components": {"m": {"type": "broken_user_mod.SomeClass"}},
                "forward": {"steps": []},
            }
        )


def test_dict_module_with_real_pytorch() -> None:
    torch = pytest.importorskip("torch")
    from confluence.torch import DictModule

    model = DictModule.from_dict(
        {
            "imports": ["torch.nn.functional"],
            "params": {"num_dim": 4, "hidden_dim": 8, "out_dim": 2},
            "components": {
                "proj": {
                    "type": "torch.nn.Linear",
                    "params": {
                        "in_features": "${params.num_dim}",
                        "out_features": "${params.hidden_dim}",
                    },
                    "compile": {"backend": "eager"},
                },
                "frozen": {
                    "type": "torch.nn.Linear",
                    "params": {
                        "in_features": "$proj.out_features",
                        "out_features": "$proj.out_features",
                    },
                    "freeze": True,
                },
                "head": {
                    "type": "torch.nn.Linear",
                    "params": {
                        "in_features": "$frozen.out_features",
                        "out_features": "${params.out_dim}",
                    },
                },
            },
            "forward": {
                "inputs": ["x"],
                "steps": [
                    {"fn": "proj", "args": ["$x"], "outputs": ["h"]},
                    {"fn": "torch.nn.functional.gelu", "args": ["$h"], "outputs": ["h"]},
                    {"fn": "frozen", "args": ["$h"], "outputs": ["h"]},
                    {"fn": "head", "args": ["$h"], "outputs": ["logits"]},
                ],
            },
            "outputs": ["logits"],
        }
    )

    x = torch.randn(3, 4)
    logits = model(x)
    assert logits.shape == (3, 2)
    logits.sum().backward()
    assert model.components["proj"].weight.grad is not None
    assert model.components["frozen"].weight.grad is None
    assert model.components["head"].weight.grad is not None

    # state_dict() keys must NOT have `_orig_mod.` prefix even with compile enabled
    sd_keys = set(model.state_dict().keys())
    assert "components.proj.weight" in sd_keys
    assert not any("_orig_mod" in k for k in sd_keys)

    # deepcopy and torch.save/torch.load work cleanly with imports + compiled component
    copied = copy.deepcopy(model)
    assert torch.allclose(copied(x), model(x))

    # forward_dict dispatches through nn.Module.__call__ so forward hooks fire
    hook_fired = False

    def _hook(_mod: Any, _inp: Any, _out: Any) -> None:
        nonlocal hook_fired
        hook_fired = True

    handle = model.register_forward_hook(_hook)
    try:
        out_dict = model.forward_dict({"x": x, "unused_label": torch.zeros(3)})
        assert hook_fired
        assert torch.allclose(out_dict, model(x))
    finally:
        handle.remove()

    import io

    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    reloaded = torch.load(buf, weights_only=False)
    assert torch.allclose(reloaded(x), model(x))

    # Reserved nn.Module / nn.ModuleDict attributes (e.g. "train", "type") fail in pre-flight validation
    with pytest.raises(ValidationError, match="conflicts with an existing nn.ModuleDict attribute"):
        DictModule.from_dict(
            {
                "components": {
                    "train": {
                        "type": "torch.nn.Linear",
                        "params": {"in_features": 2, "out_features": 2},
                    }
                },
                "forward": {"steps": []},
            }
        )
