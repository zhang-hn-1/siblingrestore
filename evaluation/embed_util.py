from __future__ import annotations

import torch
import torch.nn.functional as F


def _positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    result = list(range(0, length - tile_size + 1, stride))
    if result[-1] != length - tile_size:
        result.append(length - tile_size)
    return result


@torch.no_grad()
def embed_tiled(model, image: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
    """Average tile embeddings (each L2-normalized), renormalized, to keep
    the independent verifier memory bounded on large power-line images."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("embed_tiled expects [1, C, H, W]")
    _, _, height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    if padded.shape[-2] <= tile_size and padded.shape[-1] <= tile_size:
        return model.embed(padded)
    embeddings = []
    for top in _positions(padded.shape[-2], tile_size, overlap):
        for left in _positions(padded.shape[-1], tile_size, overlap):
            bottom = top + tile_size
            right = left + tile_size
            embeddings.append(model.embed(padded[..., top:bottom, left:right]))
    mean = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean, dim=-1)
