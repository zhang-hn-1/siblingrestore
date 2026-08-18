from __future__ import annotations

"""Loader for the official GRL-Image-Restoration model with dependency shims.

GRL imports fairscale.checkpoint_wrapper (only used when enabled), omegaconf,
and package-relative models.common.* modules. We inject minimal stand-ins and
register the official files as a synthetic `models.common` package.
"""

import importlib.util
import sys
import types
from pathlib import Path

_GRL_DIR = Path(__file__).parent / "official" / "grl"


def _install_grl_shims() -> None:
    import importlib.machinery
    if "fairscale" not in sys.modules:
        fairscale = types.ModuleType("fairscale")
        nn_mod = types.ModuleType("fairscale.nn")
        nn_mod.checkpoint_wrapper = lambda module, **kwargs: module
        fairscale.nn = nn_mod
        for name, module in (("fairscale", fairscale), ("fairscale.nn", nn_mod)):
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            sys.modules[name] = module
    if "omegaconf" not in sys.modules:
        omegaconf = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def create(value):
                if isinstance(value, dict):
                    return types.SimpleNamespace(**value)
                return types.SimpleNamespace()
            @staticmethod
            def load(*args, **kwargs):
                return {}
            @staticmethod
            def merge(*args, **kwargs):
                return {}

        omegaconf.OmegaConf = _OmegaConf
        omegaconf.__spec__ = importlib.machinery.ModuleSpec("omegaconf", None)
        sys.modules["omegaconf"] = omegaconf
    if "timm" not in sys.modules:
        from siblingrestore.official_loader import _install_timm_shim
        _install_timm_shim()


def _load_as_package_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_grl(**kwargs):
    _install_grl_shims()
    # Build synthetic models.common package.
    common = types.ModuleType("models.common")
    sys.modules["models"] = types.ModuleType("models")
    sys.modules["models.common"] = common
    upsample = _load_as_package_module("models.common.upsample", _GRL_DIR / "upsample.py")
    for name in ("Upsample", "UpsampleOneStep"):
        setattr(common, name, getattr(upsample, name))
    _load_as_package_module("models.common.ops", _GRL_DIR / "ops.py")
    _load_as_package_module("models.common.resblock", _GRL_DIR / "resblock.py")
    _load_as_package_module("models.common.swin_v1_block", _GRL_DIR / "swin_v1_block.py")
    _load_as_package_module("models.common.swin_v2_block", _GRL_DIR / "swin_v2_block.py")
    _load_as_package_module("models.common.mixed_attn_block", _GRL_DIR / "mixed_attn_block.py")
    _load_as_package_module("models.common.mixed_attn_block_efficient", _GRL_DIR / "mixed_attn_block_efficient.py")
    _load_as_package_module("models.networks.grl", _GRL_DIR / "grl.py")
    from models.networks.grl import GRL
    return GRL(**kwargs)
