from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .refinement import build_refinement


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.reshape(batch, self.heads, head_dim, height * width)
        k = k.reshape(batch, self.heads, head_dim, height * width)
        v = v.reshape(batch, self.heads, head_dim, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        output = (attention.softmax(dim=-1) @ v).reshape(batch, channels, height, width)
        return self.project(output)


class GatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.in_project = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False
        )
        self.out_project = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.depthwise(self.in_project(x)).chunk(2, dim=1)
        return self.out_project(F.gelu(first) * second)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = ChannelAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def blocks(channels: int, count: int, heads: int) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(channels, heads) for _ in range(count)))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False), nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SiblingRestormer(nn.Module):
    """Restormer-style backbone with training-only content/degradation heads."""

    def __init__(
        self,
        dim: int = 32,
        blocks_per_level: tuple[int, int, int, int, int] = (2, 2, 3, 2, 2),
        heads: tuple[int, int, int] = (1, 2, 4),
        projection_dim: int = 128,
        degradation_count: int = 7,
        degradation_conditioned: bool = False,
        identity_mode: str | None = None,
        identity_source_count: int = 71,
        refinement_type: str | None = "none",
    ) -> None:
        super().__init__()
        b1, b2, latent_blocks, d2, d1 = blocks_per_level
        self.refinement_type = str(refinement_type) if refinement_type is not None else "none"
        self.degradation_conditioned = bool(degradation_conditioned)
        if identity_mode not in (None, "classify", "branch", "contrast"):
            raise ValueError(f"unknown identity_mode: {identity_mode}")
        self.identity_mode = identity_mode
        self.identity_source_count = int(identity_source_count)
        self.patch = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.enc1 = blocks(dim, b1, heads[0])
        self.down1 = Downsample(dim)
        self.enc2 = blocks(dim * 2, b2, heads[1])
        self.down2 = Downsample(dim * 2)
        self.latent = blocks(dim * 4, latent_blocks, heads[2])
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = blocks(dim * 2, d2, heads[1])
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = blocks(dim, d1, heads[0])
        self.output = nn.Conv2d(dim, 3, 3, padding=1)
        self.content_projector = nn.Sequential(
            nn.Linear(dim * 4, dim * 4), nn.GELU(), nn.Linear(dim * 4, projection_dim)
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, degradation_count)
        )
        if self.identity_mode == "classify":
            # Source classification on the shared latent (gentle CE, same recipe
            # as the degradation aux that proved stable at 180x gradient ratio).
            self.identity_head = nn.Sequential(
                nn.Linear(dim * 4, dim * 2), nn.GELU(),
                nn.Linear(dim * 2, self.identity_source_count),
            )
        elif self.identity_mode == "branch":
            # Independent identity branch reading the restored image with a
            # detached input, so identity gradients never touch the shared net.
            self.identity_branch = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, self.identity_source_count),
            )
        if self.degradation_conditioned:
            # FiLM modulation of the three decoder levels from the predicted
            # degradation softmax; last layer is zero-init so conditioning is
            # identity at initialization and ramps up only when it helps.
            modulation_channels = dim * 4 + dim * 2 + dim
            self.condition_proj = nn.Sequential(
                nn.Linear(degradation_count, 64), nn.GELU(),
                nn.Linear(64, modulation_channels * 2),
            )
            nn.init.zeros_(self.condition_proj[-1].weight)
            with torch.no_grad():
                bias = torch.cat(
                    [torch.ones(modulation_channels), torch.zeros(modulation_channels)]
                )
                self.condition_proj[-1].bias.copy_(bias)

        # Construct refinement after all original backbone/heads are initialized.
        self.refine_module = build_refinement(self.refinement_type, dim, heads, dim * 4)

    @staticmethod
    def _pad_input(image: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = image.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return image, (height, width)
        mode = "reflect" if height > 1 and width > 1 and pad_h < height and pad_w < width else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), (height, width)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, _ = self._pad_input(image)
        level1 = self.enc1(self.patch(image))
        level2 = self.enc2(self.down1(level1))
        latent = self.latent(self.down2(level2))
        return level1, level2, latent

    def project_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = latent.mean(dim=(-2, -1))
        content = F.normalize(self.content_projector(pooled), dim=-1)
        degradation_logits = self.degradation_head(pooled)
        return content, degradation_logits

    def encode_content(self, image: torch.Tensor) -> torch.Tensor:
        _, _, latent = self.encode(image)
        return self.project_latent(latent)[0]

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        original_height, original_width = image.shape[-2:]
        image, _ = self._pad_input(image)
        level1, level2, latent = self.encode(image)
        content, degradation_logits = self.project_latent(latent)
        if self.degradation_conditioned:
            latent, decoded_level2, decoded_level1 = self._film_modulate(
                latent, degradation_logits
            )
        else:
            decoded_level2 = decoded_level1 = None

        # LMRB: refine latent before decoder (content/degradation heads use original latent)
        latent_for_decoder = latent
        if self.refinement_type == "lmrb":
            latent_for_decoder = self.refine_module(latent)

        decoded2 = self.up2(latent_for_decoder)
        decoded2 = self.dec2(self.reduce2(torch.cat([decoded2, level2], dim=1)))
        if decoded_level2 is not None:
            decoded2 = decoded2 * decoded_level2[0][..., None, None] + decoded_level2[1][..., None, None]
        decoded1 = self.up1(decoded2)
        decoded1 = self.dec1(self.reduce1(torch.cat([decoded1, level1], dim=1)))
        if decoded_level1 is not None:
            decoded1 = decoded1 * decoded_level1[0][..., None, None] + decoded_level1[1][..., None, None]

        # Dec1-level refinement (M1, M2, M3, M6): refine dec1 before output head
        refine_dec1 = decoded1
        if self.refinement_type in ("transformer", "naf", "alcrb", "haar"):
            refine_dec1 = self.refine_module(decoded1)

        base_residual = self.output(refine_dec1)

        # RERH: additional correction after base residual
        if self.refinement_type == "rerh":
            correction = self.refine_module(refine_dec1, image, base_residual)
            restored = torch.clamp(image + base_residual + correction, 0.0, 1.0)
        else:
            restored = torch.clamp(image + base_residual, 0.0, 1.0)
        restored = restored[..., :original_height, :original_width]

        identity_logits = None
        identity_feature = None
        if self.identity_mode == "classify":
            identity_logits = self.identity_head(latent.mean(dim=(-2, -1)))
        elif self.identity_mode == "branch":
            # Detach so identity gradients never flow into the shared network.
            identity_logits = self.identity_branch(restored.detach())
        elif self.identity_mode == "contrast":
            identity_feature = self.encode_content(restored)
        return {
            "restored": restored,
            "content": content,
            "degradation_logits": degradation_logits,
            "identity_logits": identity_logits,
            "identity_feature": identity_feature,
        }

    def identity_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Identity representation of an image, L2-normalized.

        classify/contrast reuse the content projector; the detached branch
        exposes its penultimate feature for open-set retrieval.
        """
        if self.identity_mode == "branch":
            features = self.identity_branch[:-1](image)
            return F.normalize(features, dim=-1)
        return self.encode_content(image)

    def _film_modulate(
        self, latent: torch.Tensor, degradation_logits: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Apply per-level FiLM from the predicted degradation distribution."""
        channels_latent, channels_dec2, channels_dec1 = latent.shape[1], self.reduce2.out_channels, self.reduce1.out_channels
        params = self.condition_proj(F.softmax(degradation_logits, dim=-1))
        offset = 0
        gamma, beta = params[:, offset:offset + channels_latent], params[:, offset + channels_latent:offset + 2 * channels_latent]
        latent = latent * gamma[..., None, None] + beta[..., None, None]
        offset += 2 * channels_latent
        gamma2, beta2 = params[:, offset:offset + channels_dec2], params[:, offset + channels_dec2:offset + 2 * channels_dec2]
        gamma1, beta1 = params[:, offset + 2 * channels_dec2:offset + 2 * channels_dec2 + channels_dec1], params[:, offset + 2 * channels_dec2 + channels_dec1:offset + 2 * channels_dec2 + 2 * channels_dec1]
        return latent, (gamma2, beta2), (gamma1, beta1)
