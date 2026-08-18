from __future__ import annotations

"""Load official baseline model files with minimal dependency shims.

Official repos often import timm/basicsr helpers; we inject tiny stand-ins so
the architecture files run with the existing torch/numpy/Pillow stack.
"""

import importlib.util
import sys
import types
from pathlib import Path

_OFFICIAL_DIR = Path(__file__).parent / "official"


def _install_timm_shim() -> None:
    if "timm" in sys.modules:
        return
    import torch
    from torch import nn

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float | None = None) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob) if drop_prob is not None else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.to_2tuple = lambda value: value if isinstance(value, (tuple, list)) else (value, value)
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    layers.DropPath = DropPath
    models.layers = layers
    timm.models = models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers


def load_official(module_name: str, file_name: str, factory: str, **kwargs):
    _install_timm_shim()
    path = _OFFICIAL_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"official model file missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, factory)(**kwargs)
