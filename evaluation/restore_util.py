from __future__ import annotations

import torch
import torch.nn.functional as F

from siblingrestore.inference import restore_tiled


def restore_image_padded(model, image: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
    """Restore a [1,C,H,W] image after padding to an 8-aligned canvas.

    The official Restormer and DehazeFormer backbones require spatial
    dimensions divisible by the pixel-unshuffle factor, while tiny native
    images short-circuit the tiled path. We therefore always run on a padded
    canvas and crop the restored output back to the original input size.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("expected a [1, C, H, W] image")
    height, width = image.shape[-2:]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h == 0 and pad_w == 0:
        padded = image
    else:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    restored = restore_tiled(model, padded, tile_size, overlap)
    return restored[..., :height, :width]
