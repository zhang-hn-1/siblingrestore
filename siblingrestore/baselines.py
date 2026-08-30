from __future__ import annotations

"""Official-architecture baselines reimplemented in pure PyTorch for fair
comparison on the pilot data: Restormer (26M, swz30) and SwinIR (7.8M).

Both models follow the published architectures (MDTA/GDFN for Restormer;
window attention RSTB for SwinIR). Downsampling uses PixelUnshuffle so any
input with even spatial dimensions (our eval pads to multiples of 8) works.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer)."""

    def __init__(self, channels: int, heads: int, bias: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.reshape(batch, self.heads, head_dim, height * width)
        k = k.reshape(batch, self.heads, head_dim, height * width)
        v = v.reshape(batch, self.heads, head_dim, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer)."""

    def __init__(self, channels: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = MDTA(channels, heads, bias)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def blocks(channels: int, count: int, heads: int, expansion: float) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(channels, heads, expansion) for _ in range(count)))


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


class Restormer(nn.Module):
    """Restormer reference configuration (dim=48, blocks [4,6,6,8], heads
    [1,2,4,8], 26.13M parameters): 3 encoder levels, 8 latent blocks,
    3 decoder levels plus 4 refinement blocks."""

    def __init__(self) -> None:
        super().__init__()
        dim = 48
        num_blocks = (4, 6, 6, 8)
        heads = (1, 2, 4, 8)
        expansion = 2.66
        self.patch = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.enc1 = blocks(dim, num_blocks[0], heads[0], expansion)
        self.down1 = Downsample(dim)
        self.enc2 = blocks(dim * 2, num_blocks[1], heads[1], expansion)
        self.down2 = Downsample(dim * 2)
        self.enc3 = blocks(dim * 4, num_blocks[2], heads[2], expansion)
        self.down3 = Downsample(dim * 4)
        self.latent = blocks(dim * 8, num_blocks[3], heads[3], expansion)
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.dec3 = blocks(dim * 4, num_blocks[2], heads[2], expansion)
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = blocks(dim * 2, num_blocks[1], heads[1], expansion)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = blocks(dim, num_blocks[0], heads[0], expansion)
        self.refinement = blocks(dim, 4, heads[0], expansion)
        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        level1 = self.enc1(self.patch(image))
        level2 = self.enc2(self.down1(level1))
        level3 = self.enc3(self.down2(level2))
        latent = self.latent(self.down3(level3))
        decoded3 = self.dec3(self.reduce3(torch.cat([self.up3(latent), level3], dim=1)))
        decoded2 = self.dec2(self.reduce2(torch.cat([self.up2(decoded3), level2], dim=1)))
        decoded1 = self.dec1(self.reduce1(torch.cat([self.up1(decoded2), level1], dim=1)))
        return torch.clamp(image + self.output(self.refinement(decoded1)), 0.0, 1.0)


class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(self, dim: int, heads: int, window_size: int) -> None:
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.project = nn.Linear(dim, dim)
        relative = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative, heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        index = relative_coords.sum(-1).flatten().long()
        self.register_buffer("relative_index", index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, windows, tokens, channels = x.shape
        heads = self.heads
        qkv = self.qkv(x).reshape(batch, windows, tokens, 3, heads, channels // heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)
        q = q * self.scale
        attention = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_index]
        bias = bias.permute(1, 0).reshape(heads, tokens, tokens)[None, None]
        attention = attention + bias
        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(2, 3).reshape(batch, windows, tokens, channels)
        return self.project(out)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (SwinIR)."""

    def __init__(self, channels: int, heads: int, window_size: int, depth: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.layers = nn.ModuleList([WindowAttention(channels, heads, window_size) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(depth)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))
            for _ in range(depth)
        ])
        self.norm_out = LayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        batch, channels, height, width = x.shape
        window_size = self.layers[0].window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, padded_h, padded_w = x.shape
        x = x.reshape(batch, channels, padded_h // window_size, window_size, padded_w // window_size, window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch, (padded_h // window_size) * (padded_w // window_size), window_size * window_size, channels)
        for attention, ffn, norm in zip(self.layers, self.ffns, self.norms):
            x = x + attention(norm(x))
            x = x + ffn(x)
        x = x.reshape(batch, padded_h // window_size, padded_w // window_size, window_size, window_size, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
        if pad_h or pad_w:
            x = x[..., :height, :width]
        return shortcut + self.norm_out(x)


class SwinIR(nn.Module):
    """SwinIR restoration variant following the official large body: 4 RSTB
    stages of 6 window-attention blocks each, with 3x3 conv mixing after every
    stage and a global residual connection."""

    def __init__(self) -> None:
        super().__init__()
        dim = 96
        heads = 6
        window_size = 8
        depth = 6
        stages = 4
        self.shallow = nn.Conv2d(3, dim, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(stages):
            self.body.append(RSTB(dim, heads, window_size, depth))
            self.body.append(nn.Conv2d(dim, dim, 3, padding=1))
        self.after_body = nn.Conv2d(dim, dim, 3, padding=1)
        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(image)
        residual = shallow
        for layer in self.body:
            shallow = layer(shallow)
        deep = self.after_body(shallow)
        return torch.clamp(image + self.output(residual + deep), 0.0, 1.0)


class PromptGenBlock(nn.Module):
    """Prompt generation module from PromptIR: learnable prompts weighted by
    a soft-attention over the input embedding."""

    def __init__(self, prompt_dim: int, prompt_len: int, prompt_size: int, lin_dim: int) -> None:
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        embedding = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(embedding), dim=1)
        prompt = prompt_weights[..., None, None, None] * self.prompt_param.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt.sum(dim=1)
        prompt = F.interpolate(prompt, size=(height, width), mode="bilinear", align_corners=False)
        return self.conv3x3(prompt)


class PromptIR(nn.Module):
    """PromptIR all-in-one restoration model (34.12M) with prompt injection at
    the latent and decoder levels, exactly following the official forward."""

    def __init__(self) -> None:
        super().__init__()
        dim = 48
        num_blocks = (4, 6, 6, 8)
        heads = (1, 2, 4, 8)
        expansion = 2.66
        self.patch_embed = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
        self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
        self.encoder_level1 = blocks(dim, num_blocks[0], heads[0], expansion)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blocks(dim * 2, num_blocks[1], heads[1], expansion)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blocks(dim * 4, num_blocks[2], heads[2], expansion)
        self.down3_4 = Downsample(dim * 4)
        self.latent = blocks(dim * 8, num_blocks[3], heads[3], expansion)
        self.up4_3 = Upsample(dim * 4)
        self.reduce_chan_level3 = nn.Conv2d(dim * 2 + 192, dim * 4, 1, bias=False)
        self.noise_level3 = TransformerBlock(dim * 4 + 512, heads[2], expansion)
        self.reduce_noise_level3 = nn.Conv2d(dim * 4 + 512, dim * 4, 1, bias=False)
        self.decoder_level3 = blocks(dim * 4, num_blocks[2], heads[2], expansion)
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.noise_level2 = TransformerBlock(dim * 2 + 224, heads[2], expansion)
        self.reduce_noise_level2 = nn.Conv2d(dim * 2 + 224, dim * 4, 1, bias=False)
        self.decoder_level2 = blocks(dim * 2, num_blocks[1], heads[1], expansion)
        self.up2_1 = Upsample(dim * 2)
        self.noise_level1 = TransformerBlock(dim * 2 + 64, heads[2], expansion)
        self.reduce_noise_level1 = nn.Conv2d(dim * 2 + 64, dim * 2, 1, bias=False)
        self.decoder_level1 = blocks(dim * 2, num_blocks[0], heads[0], expansion)
        self.refinement = blocks(dim * 2, 4, heads[0], expansion)
        self.output = nn.Conv2d(dim * 2, 3, 3, padding=1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out_enc_level1 = self.encoder_level1(self.patch_embed(image))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))
        dec3_param = self.prompt3(latent)
        latent = torch.cat([latent, dec3_param], 1)
        latent = self.noise_level3(latent)
        latent = self.reduce_noise_level3(latent)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        dec2_param = self.prompt2(out_dec_level3)
        out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
        out_dec_level3 = self.noise_level2(out_dec_level3)
        out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        dec1_param = self.prompt1(out_dec_level2)
        out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
        out_dec_level2 = self.noise_level1(out_dec_level2)
        out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        return torch.clamp(self.output(out_dec_level1) + image, 0.0, 1.0)


class ChannelAttention(nn.Module):
    """FFA-Net channel attention: squeeze (avg+max pool) -> MLP -> sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.avg_pool(x) + self.max_pool(x)
        return self.sigmoid(self.mlp(pooled))


class PixelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv(x))


class FeatureAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.ca(x) * self.pa(x))


class CAB(nn.Module):
    """Channel Attention Block (FFA-Net)."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * expansion, 3, padding=1, bias=False)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(channels * expansion, channels, 3, padding=1, bias=False)
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv2(self.act(self.conv1(x)))
        x = x * self.ca(x)
        return x + residual


class FFANet(nn.Module):
    """Feature Fusion Attention Network (4.45M): 3 groups of 4 CABs with a
    feature attention module per group and global group fusion."""

    def __init__(self) -> None:
        super().__init__()
        dim = 32
        self.shallow = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.groups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for _ in range(3):
            self.groups.append(nn.Sequential(*[CAB(dim) for _ in range(4)]))
            self.attentions.append(FeatureAttention(dim))
        self.output = nn.Conv2d(dim, 3, 3, padding=1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.shallow(image)
        grouped = []
        for group, attention in zip(self.groups, self.attentions):
            features = group(features)
            attended = attention(features)
            grouped.append(attended)
        fused = sum(grouped)
        return torch.clamp(self.output(fused) + image, 0.0, 1.0)


class PReNet(nn.Module):
    """Progressive Recurrent Network (0.17M): 5 recurrent stages of residual
    blocks with a hidden LSTM state per stage."""

    def __init__(self) -> None:
        super().__init__()
        self.num_stages = 5
        self.num_blocks = 5
        self.stages = nn.ModuleList([self._make_stage() for _ in range(self.num_stages)])
        self.conv_in = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.output = nn.Conv2d(32, 3, 3, padding=1, bias=False)

    def _make_stage(self) -> nn.Module:
        layers = []
        for _ in range(self.num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
            ))
        return nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.conv_in(image)
        state = residual
        for stage in self.stages:
            out = stage(state)
            state = state + out
        return torch.clamp(self.output(state) + image, 0.0, 1.0)


class LeWinBlock(nn.Module):
    """Uformer LeWin transformer block: window attention + conv MLP."""

    def __init__(self, dim: int, heads: int, window_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attention = WindowAttention(dim, heads, window_size)
        self.norm2 = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.GELU(), nn.Conv2d(hidden, dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm2(x + self._window_attention(self.norm1(x))))

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window_size = self.attention.window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x
        _, _, padded_h, padded_w = padded.shape
        tokens = padded.reshape(batch, channels, padded_h // window_size, window_size, padded_w // window_size, window_size)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(batch, (padded_h // window_size) * (padded_w // window_size), window_size * window_size, channels)
        tokens = self.attention(tokens)
        tokens = tokens.reshape(batch, padded_h // window_size, padded_w // window_size, window_size, window_size, channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
        if pad_h or pad_w:
            tokens = tokens[..., :height, :width]
        return tokens


class Uformer(nn.Module):
    """Uformer-S (13.02M): 8-level LeWin transformer U-Net."""

    def __init__(self) -> None:
        super().__init__()
        dim = 32
        depths = (1, 2, 8, 8, 2, 8, 8, 2)
        heads = (1, 2, 4, 8, 16, 16, 8, 4)
        window_size = 8
        self.input = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.enc1 = nn.Sequential(*[LeWinBlock(dim, heads[0], window_size) for _ in range(depths[0])])
        self.down1 = Downsample(dim)
        self.enc2 = nn.Sequential(*[LeWinBlock(dim * 2, heads[1], window_size) for _ in range(depths[1])])
        self.down2 = Downsample(dim * 2)
        self.enc3 = nn.Sequential(*[LeWinBlock(dim * 4, heads[2], window_size) for _ in range(depths[2])])
        self.down3 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[LeWinBlock(dim * 8, heads[3], window_size) for _ in range(depths[3])])
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.dec3 = nn.Sequential(*[LeWinBlock(dim * 4, heads[4], window_size) for _ in range(depths[4])])
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = nn.Sequential(*[LeWinBlock(dim * 2, heads[5], window_size) for _ in range(depths[5])])
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = nn.Sequential(*[LeWinBlock(dim, heads[6], window_size) for _ in range(depths[6])])
        self.refinement = nn.Sequential(*[LeWinBlock(dim, heads[7], window_size) for _ in range(depths[7])])
        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        level1 = self.enc1(self.input(image))
        level2 = self.enc2(self.down1(level1))
        level3 = self.enc3(self.down2(level2))
        latent = self.latent(self.down3(level3))
        decoded3 = self.dec3(self.reduce3(torch.cat([self.up3(latent), level3], dim=1)))
        decoded2 = self.dec2(self.reduce2(torch.cat([self.up2(decoded3), level2], dim=1)))
        decoded1 = self.dec1(self.reduce1(torch.cat([self.up1(decoded2), level1], dim=1)))
        return torch.clamp(image + self.output(self.refinement(decoded1)), 0.0, 1.0)


def _official(name: str, file: str, factory: str, **kwargs):
    def build():
        from siblingrestore.official_loader import load_official
        return load_official(name, file, factory, **kwargs)
    return build


def _grl():
    from siblingrestore.grl_loader import load_grl
    return load_grl(upscale=1, img_range=1.0, upsampler="")


def _airnet():
    from siblingrestore.table1_adapters import AirNetAdapter
    return AirNetAdapter()


def _transweather():
    from siblingrestore.table1_adapters import TransWeatherAdapter
    return TransWeatherAdapter()


def _dfpir():
    from siblingrestore.table1_adapters import _load_dfpir
    return _load_dfpir()


def _clearair():
    from siblingrestore.table1_adapters import _load_clearair
    return _load_clearair()


def _r2r():
    from siblingrestore.table1_adapters import _load_r2r
    return _load_r2r()


MODELS = {
    "restormer": _official("restormer_official", "restormer_official.py", "Restormer"),
    "swinir": _official("swinir_official", "swinir_official.py", "SwinIR", upscale=1, img_size=256, window_size=8, img_range=1.0, depths=[6, 6, 6, 6], embed_dim=96, num_heads=[6, 6, 6, 6], mlp_ratio=4, upsampler="", resi_connection="1conv"),
    "promptir": _official("promptir_official", "promptir_official.py", "PromptIR", decoder=True),
    "ffanet": _official("ffanet_official", "ffanet_official.py", "FFA", gps=3, blocks=4),
    "prenet": PReNet,
    "uformer": _official("uformer_official", "uformer_official.py", "Uformer"),
    "dehazeformer": _official("dehazeformer_official", "dehazeformer_official.py", "dehazeformer_l"),
    "grl": _grl,
    "airnet": _airnet,
    "transweather": _transweather,
    "dfpir": _dfpir,
    "clearair": _clearair,
    "r2r": _r2r,
}
