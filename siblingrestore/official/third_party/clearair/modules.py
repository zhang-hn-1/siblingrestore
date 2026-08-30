"""
The three conditioning modules of ClearAIR.

QGM  : Quality Guidance Module        (consumes MLLM-IQA score embedding)
SCA  : Semantic Cross-Attention       (consumes SGU semantic feature)
DAM  : Degradation-Aware Module       (consumes content + degradation prompt
                                       from the Task Identifier)

All equations refer to the ClearAIR paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


# ---------------------------------------------------------------------------
# Adapter: maps the MLLM-IQA hidden state Q to feature space (Eq. 3)
# ---------------------------------------------------------------------------
class IQAAdapter(nn.Module):
    """A_adapter in Eq. 3: F_q = A_adapter(Q)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q : (B, in_dim)  ->  F_q : (B, out_dim)
        return self.net(q)


# ---------------------------------------------------------------------------
# Quality Guidance Module — Eq. 4
#   X_out = X_in ⊙ Linear(F_q) + Linear(F_q)
# ---------------------------------------------------------------------------
class QualityGuidanceModule(nn.Module):
    def __init__(self, dim: int, fq_dim: int):
        super().__init__()
        self.scale = nn.Linear(fq_dim, dim)
        self.shift = nn.Linear(fq_dim, dim)
        self.norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor, fq: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W);  fq : (B, fq_dim)
        gamma = self.scale(fq).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.shift(fq).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Mask Average Pooling — Eq. 6, 7
#   pool features inside each binary mask, then broadcast back.
# ---------------------------------------------------------------------------
def mask_average_pool(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    features : (B, C, H, W)
    masks    : (B, Nm, H, W) in {0, 1}
    returns  : (B, C, H, W)  — semantic-aware structural prior F_sem
    """
    b, c, h, w = features.shape
    _, nm, _, _ = masks.shape

    # numerator : (B, Nm, C)   sum of features in each region
    feat_flat = features.view(b, c, -1)              # (B, C, HW)
    mask_flat = masks.view(b, nm, -1)                # (B, Nm, HW)

    region_sum = mask_flat @ feat_flat.transpose(1, 2)  # (B, Nm, C)
    region_area = mask_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, Nm, 1)
    region_mean = region_sum / region_area              # (B, Nm, C)

    # broadcast back: F_sem(h, w) = mean of region the pixel belongs to
    # (B, Nm, C) x (B, Nm, HW) -> (B, C, HW)
    f_sem = region_mean.transpose(1, 2) @ mask_flat     # (B, C, HW)
    f_sem = f_sem.view(b, c, h, w)
    return f_sem


# ---------------------------------------------------------------------------
# Semantic Cross-Attention — Eq. 8, 9
# ---------------------------------------------------------------------------
class SemanticCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = LayerNorm2d(dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, C, H, W)  — F^in_sca
        f_sem : (B, C, H, W)  — semantic features from SGU + MAP

        Windowed variant of the original global HW x HW attention: the input is
        split into windows (default 8x8), attention is computed inside each
        window, and results are stitched back. This keeps the semantic-guided
        cross-attention semantics while making training tractable (the global
        matmul is O(H^2 W^2) and OOMs on 32GB at crop 256).
        """
        import torch.nn.functional as Fn

        b, c, h, w = x.shape
        residual = x
        x_norm = self.norm(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(f_sem)
        v = self.v_proj(f_sem)

        win = 8
        pad_h = (win - h % win) % win
        pad_w = (win - w % win) % win
        if pad_h or pad_w:
            q = Fn.pad(q, (0, pad_w, 0, pad_h))
            k = Fn.pad(k, (0, pad_w, 0, pad_h))
            v = Fn.pad(v, (0, pad_w, 0, pad_h))
        _, _, h2, w2 = q.shape
        nh, nw = h2 // win, w2 // win

        def _to_windows(t: torch.Tensor) -> torch.Tensor:
            # (B, C, H, W) -> (B, nh, nw, win, win, C) -> (B*nh*nw, win*win, C)
            t = t.reshape(b, c, nh, win, nw, win)
            t = t.permute(0, 2, 4, 3, 5, 1).reshape(-1, win * win, c)
            return t

        q = _to_windows(q).reshape(-1, self.num_heads, self.head_dim, win * win)
        k = _to_windows(k).reshape(-1, self.num_heads, self.head_dim, win * win)
        v = _to_windows(v).reshape(-1, self.num_heads, self.head_dim, win * win)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # (nW, h, win*win, win*win)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))             # (nW, h, head_dim, win*win)
        out = out.reshape(-1, win, win, c).permute(0, 3, 1, 2)    # (nW, C, win, win)

        # stitch windows back
        out = out.reshape(b, nh, nw, c, win, win)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h2, w2)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return residual + self.out_proj(out)


# ---------------------------------------------------------------------------
# Degradation Prompt — Eq. 10
#   F_p = MLP(P) ⊙ softmax(MLP(F_d))
# ---------------------------------------------------------------------------
class DegradationPromptGenerator(nn.Module):
    def __init__(self, fd_dim: int = 512, num_prompts: int = 5, prompt_dim: int = 192):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.fd_mlp = nn.Sequential(
            nn.Linear(fd_dim, fd_dim),
            nn.GELU(),
            nn.Linear(fd_dim, num_prompts),
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )

    def forward(self, fd: torch.Tensor) -> torch.Tensor:
        """fd : (B, fd_dim)  ->  F_p : (B, prompt_dim)."""
        weights = self.fd_mlp(fd).softmax(dim=-1)            # (B, num_prompts)
        weighted = weights @ self.prompts                    # (B, prompt_dim)
        return self.prompt_mlp(weighted)


# ---------------------------------------------------------------------------
# Degradation-Aware Module — Eq. 11–14
# ---------------------------------------------------------------------------
class DegradationAwareModule(nn.Module):
    def __init__(
        self,
        dim: int,
        prompt_dim: int = 192,
        fc_dim: int = 512,
        num_heads: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = LayerNorm2d(dim)
        self.in_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # cross-attention with content embedding F_c (Eq. 12)
        self.fc_kv = nn.Linear(fc_dim, dim * 2)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # degradation-prompt modulation (Eq. 13, 14)
        self.fp_mask = nn.Sequential(
            nn.Linear(prompt_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # final fusion (concat -> conv)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        fc: torch.Tensor,
        fp: torch.Tensor,
    ) -> torch.Tensor:
        """
        x  : (B, C, H, W)         — X^in_dam
        fc : (B, fc_dim)          — content embedding from DA-CLIP
        fp : (B, prompt_dim)      — degradation prompt
        """
        b, c, h, w = x.shape
        residual = x

        x_hat = self.in_conv(self.norm(x))                          # X̂^in_dam (Eq. 11)

        # --- cross-attention with F_c (Eq. 12) -----------------------------
        kv = self.fc_kv(fc)                                         # (B, 2C)
        k_vec, v_vec = kv.chunk(2, dim=-1)                          # (B, C) each
        q = self.q_proj(x_hat)                                      # (B, C, H, W)

        q_h = q.reshape(b, self.num_heads, self.head_dim, h * w)    # (B, h, hd, HW)
        k_h = k_vec.reshape(b, self.num_heads, self.head_dim, 1)    # (B, h, hd, 1)
        v_h = v_vec.reshape(b, self.num_heads, self.head_dim, 1)

        # attention over a single key/value (works as gated content broadcast)
        attn = (q_h.transpose(-2, -1) @ k_h) * self.scale           # (B, h, HW, 1)
        attn = attn.softmax(dim=-2)
        out = v_h @ attn.transpose(-2, -1)                          # (B, h, hd, HW)
        x_att = out.reshape(b, c, h, w)
        x_att = self.attn_proj(x_att)

        # --- degradation mask + modulation (Eq. 13, 14) --------------------
        m_d = torch.sigmoid(self.fp_mask(fp))                       # (B, C)
        m_d = m_d.unsqueeze(-1).unsqueeze(-1)                       # (B, C, 1, 1)
        f_m = m_d * x_hat                                           # (B, C, H, W)

        # --- fuse -----------------------------------------------------------
        out = self.fuse(torch.cat([x_att, f_m], dim=1))
        return residual + out
