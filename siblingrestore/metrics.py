from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    return ((prediction - target).pow(2) * expanded).sum() / expanded.sum().clamp_min(1.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mse = float(masked_mse(prediction, target, mask))
    return -10.0 * math.log10(max(mse, 1e-12))


def ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Global masked SSIM for fast pilot diagnostics; formal work should use windowed SSIM."""
    expanded = mask.expand_as(prediction)
    count = expanded.sum().clamp_min(1.0)
    mean_x = (prediction * expanded).sum() / count
    mean_y = (target * expanded).sum() / count
    var_x = (((prediction - mean_x) ** 2) * expanded).sum() / count
    var_y = (((target - mean_y) ** 2) * expanded).sum() / count
    covariance = (((prediction - mean_x) * (target - mean_y)) * expanded).sum() / count
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(value)
