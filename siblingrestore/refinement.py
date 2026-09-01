from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RefineLayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ──────────────────────────────────────────────
# Unified registry
# ──────────────────────────────────────────────

REFINEMENT_REGISTRY: dict[str, type[nn.Module]] = {}


def register(name: str):
    def decorator(cls):
        REFINEMENT_REGISTRY[name] = cls
        return cls
    return decorator


def build_refinement(refinement_type: str, dim: int, heads: tuple[int, int, int], latent_dim: int) -> nn.Module | None:
    """Build a refinement module by type, or return None for 'none'.

    Args:
        refinement_type: one of 'none', 'transformer', 'naf', 'alcrb',
                         'rerh', 'lmrb', 'haar'
        dim: base channel dimension
        heads: (enc1_head, enc2_head, latent_head)
        latent_dim: latent dimension (dim * 4)
    """
    if refinement_type == 'none' or refinement_type is None:
        return None
    cls = REFINEMENT_REGISTRY.get(refinement_type)
    if cls is None:
        raise ValueError(f"Unknown refinement_type={refinement_type!r}; "
                         f"available: {list(REFINEMENT_REGISTRY)}")
    return cls(dim=dim, heads=heads, latent_dim=latent_dim)


# ──────────────────────────────────────────────
# M1: TransformerRefine
# ──────────────────────────────────────────────

@register("transformer")
class TransformerRefine(nn.Module):
    """Single TransformerBlock as a refine module after dec1."""

    def __init__(self, dim: int = 48, heads: tuple[int, int, int] = (1, 2, 4), **kwargs):
        super().__init__()
        # Import lazily to avoid model <-> refinement circular import at module load.
        from .model import TransformerBlock
        self.block = TransformerBlock(channels=dim, heads=heads[0])
        # Zero the residual-producing projections so the insertion is initially identity.
        nn.init.zeros_(self.block.attention.project.weight)
        nn.init.zeros_(self.block.ffn.out_project.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────────────────────────────
# M2: NAFRefine
# ──────────────────────────────────────────────

@register("naf")
class NAFRefine(nn.Module):
    """Single NAFBlock as a refine module after dec1."""

    def __init__(self, dim: int = 48, **kwargs):
        super().__init__()
        self.norm = RefineLayerNorm2d(dim)
        # 1x1 Conv: C -> 2C
        self.conv1 = nn.Conv2d(dim, dim * 2, 1, bias=False)
        # 3x3 Depthwise Conv on 2C
        self.dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False)
        # Simplified channel attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(dim, dim, 1, bias=False)
        # 1x1 Conv: C -> C
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        # Second FFN branch
        self.conv3 = nn.Conv2d(dim, dim, 1, bias=False)
        # Learnable residual scaling
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        # First branch: gated convolution
        x = self.conv1(x)
        x = self.dw(x)
        a, b = x.chunk(2, dim=1)
        x = a * b
        attention = self.pool(x)
        attention = torch.sigmoid(self.ca_conv(attention))
        x = self.conv2(x * attention)
        x = shortcut + self.beta * x

        # Second FFN branch without an activation, as specified for NAF.
        shortcut2 = x
        x = self.norm(x)
        x = self.conv3(x)
        x = shortcut2 + self.gamma * x
        return x


# ──────────────────────────────────────────────
# M3: ALCRB — Adaptive Local Context Refinement Block
# ──────────────────────────────────────────────

@register("alcrb")
class ALCRB(nn.Module):
    """Adaptive Local Context Refinement Block."""

    def __init__(self, dim: int = 48, reduction: int = 8, **kwargs):
        super().__init__()
        self.norm = RefineLayerNorm2d(dim)
        self.conv_expand = nn.Conv2d(dim, dim * 2, 1, bias=False)
        # Local branch: 3x3 Depthwise
        self.local_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # Context branch: 3x3 Depthwise dilation=2
        self.context_dw = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # MLP: 2C -> 2C/r -> 2C  (output two per-channel weight sets)
        mlp_hidden = max(dim * 2 // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim * 2),
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        # Expand
        x = self.conv_expand(x)
        local_feat, context_feat = x.chunk(2, dim=1)

        local_feat = self.local_dw(local_feat)
        context_feat = self.context_dw(context_feat)

        # Pool + weights
        b, c, h, w = local_feat.shape
        pooled_local = self.pool(local_feat).view(b, c)
        pooled_context = self.pool(context_feat).view(b, c)
        pooled_cat = torch.cat([pooled_local, pooled_context], dim=1)
        weights = self.mlp(pooled_cat).view(b, 2, c, 1, 1)  # [B, 2, C, 1, 1]
        weights = F.softmax(weights, dim=1)

        # Weighted sum
        x = weights[:, 0] * local_feat + weights[:, 1] * context_feat
        x = self.proj(x)
        return shortcut + self.beta * x


# ──────────────────────────────────────────────
# M4: RERH — Residual Error Refinement Head
# ──────────────────────────────────────────────

class _SimplifiedNAFBlock(nn.Module):
    """Simplified 1-block NAF used inside RERH."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = RefineLayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dw(x)
        a, b = x.chunk(2, dim=1)
        x = a * b
        x = self.conv2(x)
        return shortcut + self.beta * x


@register("rerh")
class RERH(nn.Module):
    """Residual Error Refinement Head after the base residual output."""

    def __init__(self, dim: int = 48, **kwargs):
        super().__init__()
        # Input: cat(decoded1, original_input, base_residual) = dim + 3 + 3 = dim + 6
        self.proj_in = nn.Conv2d(dim + 6, dim, 1, bias=False)
        self.naf = _SimplifiedNAFBlock(dim)
        self.conv_out = nn.Conv2d(dim, 3, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, decoded1: torch.Tensor, original_input: torch.Tensor,
                base_residual: torch.Tensor) -> torch.Tensor:
        x = torch.cat([decoded1, original_input, base_residual], dim=1)
        x = self.proj_in(x)
        x = self.naf(x)
        correction = self.conv_out(x)
        return self.alpha * correction


# ──────────────────────────────────────────────
# M5: LMRB — Latent Multi Receptive Field Block
# ──────────────────────────────────────────────

@register("lmrb")
class LMRB(nn.Module):
    """Latent Multi Receptive Field Block, operates on latent before decoder."""

    def __init__(self, dim: int = 48, latent_dim: int = 192, **kwargs):
        super().__init__()
        self.norm = RefineLayerNorm2d(latent_dim)
        self.conv_compress = nn.Conv2d(latent_dim, latent_dim // 2, 1, bias=False)
        mid = latent_dim // 2  # 96
        self.local_dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.context_dw = nn.Conv2d(mid, mid, 3, padding=3, dilation=3, groups=mid, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        mlp_hidden = max(mid * 2 // 8, 4)
        self.mlp = nn.Sequential(
            nn.Linear(mid * 2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mid * 2),
        )
        self.proj = nn.Conv2d(mid, latent_dim, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.conv_compress(x)
        local_feat = self.local_dw(x)
        context_feat = self.context_dw(x)

        b, c, h, w = local_feat.shape
        pooled_local = self.pool(local_feat).view(b, c)
        pooled_context = self.pool(context_feat).view(b, c)
        pooled_cat = torch.cat([pooled_local, pooled_context], dim=1)
        weights = self.mlp(pooled_cat).view(b, 2, c, 1, 1)
        weights = F.softmax(weights, dim=1)

        x = weights[:, 0] * local_feat + weights[:, 1] * context_feat
        x = self.proj(x)
        return shortcut + self.beta * x


# ──────────────────────────────────────────────
# M6: HFRB — Haar Frequency Refinement Block
# ──────────────────────────────────────────────

class _HaarTransform(nn.Module):
    """Fixed orthogonal Haar wavelet transform (2D, single level).

    Input/Output: [B, C, H, W]
    The transform is fixed (not learned).
    """

    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale
        # Build fixed Haar filters
        h = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)  # LL
        v = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32)  # LH
        h2 = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32)  # HL
        d = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)  # HH

        filters = torch.stack([h, v, h2, d]).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer('filters', filters * scale)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [LL, LH, HL, HH] each [B, C, H/2, W/2]."""
        batch, channels, height, width = x.shape
        # Ensure even dimensions
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        filters = self.filters.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        out = F.conv2d(
            x.reshape(batch, channels, x.shape[2], x.shape[3]),
            filters,
            stride=2,
            groups=channels,
        )
        subbands = out.reshape(batch, channels, 4, out.shape[2], out.shape[3])
        ll, lh, hl, hh = subbands.unbind(dim=2)
        return [ll, lh, hl, hh]

    def inverse(self, subbands: list[torch.Tensor], original_size: tuple[int, int] | None = None) -> torch.Tensor:
        """Reconstruct from [LL, LH, HL, HH]."""
        ll, lh, hl, hh = subbands
        batch, channels, h, w = ll.shape
        # Group four subbands belonging to each channel for grouped synthesis.
        block = torch.stack([ll, lh, hl, hh], dim=2).reshape(batch, channels * 4, h, w)
        filters = self.filters.to(device=block.device, dtype=block.dtype).repeat(channels, 1, 1, 1)
        recon = F.conv_transpose2d(
            block,
            filters,
            stride=2,
            groups=channels,
            padding=0,
        )
        if original_size is not None:
            recon = recon[:, :, :original_size[0], :original_size[1]]
        return recon


@register("haar")
class HFRB(nn.Module):
    """Haar Frequency Refinement Block: Haar transform -> refine -> inverse."""

    def __init__(self, dim: int = 48, **kwargs):
        super().__init__()
        self.haar = _HaarTransform(scale=0.5)
        # Shared 3x3 Depthwise Conv for all subbands
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # Shared 1x1 Conv
        self.conv_shared = nn.Conv2d(dim, dim, 1, bias=False)
        # Global pooling + MLP for per-subband per-channel weights
        self.pool = nn.AdaptiveAvgPool2d(1)
        mlp_hidden = max(dim // 4, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim * 4),  # 4 subbands x C weights
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        original_h, original_w = x.shape[-2], x.shape[-1]

        # Haar transform
        subbands = self.haar(x)  # [LL, LH, HL, HH] each [B, C, H/2, W/2]

        # Shared depthwise + 1x1 on each subband
        refined = []
        for sb in subbands:
            r = self.dw(sb)
            r = self.conv_shared(r)
            refined.append(r)

        # Generate per-subband per-channel weights via global pooling
        # Use pooled original subband features for weight generation
        weight_inputs = []
        for sb in subbands:
            weight_inputs.append(self.pool(sb).view(sb.shape[0], sb.shape[1]))
        # Average across subbands for the shared weight computation
        weight_in = torch.stack(weight_inputs, dim=2).mean(dim=2)  # [B, C]
        weights = self.mlp(weight_in)  # [B, C*4]
        weights = weights.view(-1, 4, subbands[0].shape[1], 1, 1)  # [B, 4, C, 1, 1]
        weights = torch.sigmoid(weights)

        # Apply weights
        weighted = []
        for i in range(4):
            weighted.append(weights[:, i] * refined[i])

        # Inverse Haar
        recon = self.haar.inverse(weighted, original_size=(original_h, original_w))
        return shortcut + self.beta * recon


def haar_reconstruction_test(device: str = 'cpu', eps: float = 1e-6) -> bool:
    """Test that Haar transform + inverse is numerically consistent."""
    haar = _HaarTransform(scale=0.5)
    x = torch.randn(2, 8, 32, 32, device=torch.device(device))
    subbands = haar(x)
    recon = haar.inverse(subbands, original_size=(32, 32))
    err = (x - recon).abs().max().item()
    return err < eps, err
