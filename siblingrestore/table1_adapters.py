"""Adapters for Table 1 official baselines.

The upstream AirNet implementation has a two-input training API.  The local
runner intentionally exposes the same one-input restoration API for all
baselines, so this adapter keeps AirNet's official architecture while using
the degraded image as both query and key.  This is suitable for a comparable
restoration-only baseline; AirNet's original contrastive pretraining is not
claimed here.
"""

from types import SimpleNamespace
from pathlib import Path
import types

import torch
from torch import nn


class AirNetAdapter(nn.Module):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        from siblingrestore.official.third_party.airnet_net.model import AirNet

        opt = SimpleNamespace(batch_size=batch_size)
        self.model = AirNet(opt)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.training:
            restored, _, _ = self.model(image, image.detach())
        else:
            restored = self.model(image, image)
        return torch.clamp(restored, 0.0, 1.0)


class TransWeatherAdapter(nn.Module):
    """Adapter for TransWeather (jeya-maria-jose/TransWeather) all-in-one model.

    TransWeather outputs tanh-activated reconstructions in [-1, 1]; the local
    training protocol expects [0, 1] images, so we re-normalize here.
    """

    def __init__(self) -> None:
        super().__init__()
        import sys

        sys.path.insert(0, str(Path(__file__).parent / "official" / "third_party"))
        from siblingrestore.official_loader import _install_timm_shim
        _install_timm_shim()
        from transweather_model import Transweather

        self.model = Transweather()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # TransWeather downsamples by 32 (patch_embed1 stride 4, then 2x4 more);
        # validation tiles whose size is not a multiple of 32 crash in the
        # decoder's fixed-offset size correction. Pad reflectively to the next
        # multiple of 32 and crop the output back.
        _, _, h, w = image.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.model(image)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)


def _load_dfpir() -> nn.Module:
    """Load DFPIR (TxpHome/DFPIR) Restormer-style model.

    The upstream file imports `net.arch_util`/`net.local_arch` as a package and
    `clip` (imported but unused); we register synthetic modules so the official
    file loads as-is. Output is not normalized by the upstream network, so we
    clamp to [0, 1] here.
    """
    import importlib.machinery
    import sys

    dfpir_dir = Path(__file__).parent / "official" / "third_party" / "dfpir_net"
    sys.path.insert(0, str(dfpir_dir.parent))

    net = types.ModuleType("net")
    net.__spec__ = importlib.machinery.ModuleSpec("net", None)
    sys.modules["net"] = net
    for sub in ("arch_util", "local_arch"):
        spec = importlib.util.spec_from_file_location(f"net.{sub}", dfpir_dir / f"{sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"net.{sub}"] = module
        spec.loader.exec_module(module)

    if "clip" not in sys.modules:
        clip = types.ModuleType("clip")
        clip.__spec__ = importlib.machinery.ModuleSpec("clip", None)
        sys.modules["clip"] = clip

    sys.path.insert(0, str(dfpir_dir))
    from model import Restormer as _DFPIRRestormer

    net_module = _DFPIRRestormer()

    class _DFPIRAdapter(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.clamp(net_module(image), 0.0, 1.0)

    adapter = _DFPIRAdapter()
    adapter.net_module = net_module  # keep a reference so params are tracked
    return adapter


def _load_clearair() -> nn.Module:
    """Load ClearAIR (House-yuyu/ClearAIR) with dummy auxiliaries enabled so no
    external SAM2/DACLIP/DeQA pretrained weights are required. The model is a
    single-input all-in-one restorer; upstream returns raw reconstructions, so
    clamp to [0, 1] here."""
    import importlib.machinery
    import sys

    src_dir = Path(__file__).parent / "official" / "third_party" / "clearair"
    sys.path.insert(0, str(src_dir.parent))

    if "clearair" not in sys.modules:
        pkg = types.ModuleType("clearair")
        pkg.__spec__ = importlib.machinery.ModuleSpec("clearair", None)
        pkg.__path__ = [str(src_dir)]
        sys.modules["clearair"] = pkg
    for sub in ("blocks", "modules", "auxiliary", "real_auxiliary", "model"):
        spec = importlib.util.spec_from_file_location(f"clearair.{sub}", src_dir / f"{sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"clearair.{sub}"] = module
        spec.loader.exec_module(module)

    from clearair.model import ClearAIR, ClearAIRConfig

    cfg = ClearAIRConfig(dummy_auxiliaries=True)
    net = ClearAIR(cfg)

    class _ClearAIRAdapter(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.clamp(net(image), 0.0, 1.0)

    adapter = _ClearAIRAdapter()
    adapter.net_module = net
    return adapter


def _load_r2r() -> nn.Module:
    """Load R2R (cscxwang/R2R) in inference mode.

    R2R's degradation-memory forward expects (degraded, clean, ids); in
    eval mode the memory is empty and the fallback returns a zero prompt, so a
    single-input forward is safe. The upstream R2RTest class requires ckpt
    files, so we build R2R(is_train=False) directly.
    """
    import importlib.machinery
    import sys

    r2r_dir = Path(__file__).parent / "official" / "third_party" / "r2r_net"
    sys.path.insert(0, str(r2r_dir.parent))
    net = types.ModuleType("net")
    net.__spec__ = importlib.machinery.ModuleSpec("net", None)
    sys.modules["net"] = net
    for sub in ("feature_bank_5D", "model_5D"):
        spec = importlib.util.spec_from_file_location(f"net.{sub}", r2r_dir / f"{sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"net.{sub}"] = module
        spec.loader.exec_module(module)
    sys.path.insert(0, str(r2r_dir))
    from model_5D import R2R

    net_model = R2R(is_train=False, num_classes=5)

    class _R2RAdapter(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            # R2R pads input to a multiple of its padder_size (16) internally;
            # crop the output back to the input size so PSNR comparisons against
            # the clean target match.
            _, _, h, w = image.shape
            out = net_model(image, interact_label=None)
            if out.shape[-2] != h or out.shape[-1] != w:
                out = out[..., :h, :w]
            return torch.clamp(out, 0.0, 1.0)

    adapter = _R2RAdapter()
    adapter.net_module = net_model
    return adapter
