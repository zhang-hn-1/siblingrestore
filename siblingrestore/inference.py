from __future__ import annotations

import torch
import torch.nn.functional as F

from siblingrestore.model import SiblingRestormer


def _positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


@torch.no_grad()
def restore_tiled(
    model: SiblingRestormer,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    """Restore a B=1 tensor with overlap averaging to cap validation VRAM."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("restore_tiled expects a [1, C, H, W] tensor")
    if tile_size <= 0 or tile_size % 8:
        raise ValueError("tile_size must be a positive multiple of 8")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    _, _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        result = model(image)
        return result["restored"] if isinstance(result, dict) else result

    restored_sum = torch.zeros_like(image)
    weight = torch.zeros_like(image[:, :1])
    tiles = []
    positions = []
    for top in _positions(height, tile_size, overlap):
        for left in _positions(width, tile_size, overlap):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tiles.append(image[..., top:bottom, left:right])
            positions.append((top, left, bottom, right))
    max_h = max(tile.shape[-2] for tile in tiles)
    max_w = max(tile.shape[-1] for tile in tiles)
    pad_h = (8 - max_h % 8) % 8
    pad_w = (8 - max_w % 8) % 8
    padded = [F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect") for tile in tiles]
    chunk_size = 2
    for start in range(0, len(padded), chunk_size):
        chunk = torch.cat(padded[start : start + chunk_size], dim=0)
        result = model(chunk)
        restored_batch = result["restored"] if isinstance(result, dict) else result
        for (top, left, bottom, right), restored in zip(positions[start : start + chunk_size], restored_batch):
            restored_sum[..., top:bottom, left:right] += restored[..., : bottom - top, : right - left]
            weight[..., top:bottom, left:right] += 1
    return restored_sum / weight.clamp_min(1)


@torch.no_grad()
def identity_embed_tiled(
    model: SiblingRestormer,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    """Mean identity embedding over overlapping tiles to cap attention VRAM.

    Uses model.identity_embed so the detached identity branch (or the content
    projector for classify/contrast modes) is applied consistently. Tile
    embeddings are L2-normalized; the mean is re-normalized for cosine
    retrieval.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("identity_embed_tiled expects a [1, C, H, W] tensor")
    _, _, height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    if height <= tile_size and width <= tile_size:
        return model.identity_embed(padded)
    embeddings = []
    padded_height, padded_width = padded.shape[-2:]
    for top in _positions(padded_height, tile_size, overlap):
        for left in _positions(padded_width, tile_size, overlap):
            bottom = min(top + tile_size, padded_height)
            right = min(left + tile_size, padded_width)
            tile = padded[..., top:bottom, left:right]
            embeddings.append(model.identity_embed(tile))
    mean = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean, dim=-1)


@torch.no_grad()
def encode_content_tiled(
    model: SiblingRestormer,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    """Backward-compatible alias extracting the content embedding per tile."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("encode_content_tiled expects a [1, C, H, W] tensor")
    _, _, height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    if height <= tile_size and width <= tile_size:
        return model.encode_content(padded)
    embeddings = []
    padded_height, padded_width = padded.shape[-2:]
    for top in _positions(padded_height, tile_size, overlap):
        for left in _positions(padded_width, tile_size, overlap):
            bottom = min(top + tile_size, padded_height)
            right = min(left + tile_size, padded_width)
            tile = padded[..., top:bottom, left:right]
            embeddings.append(model.encode_content(tile))
    mean = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean, dim=-1)
