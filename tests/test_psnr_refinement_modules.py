from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siblingrestore.model import SiblingRestormer
from siblingrestore.refinement import _HaarTransform


TYPES = ("none", "transformer", "naf", "alcrb", "rerh", "lmrb", "haar", "alcrb_hfrb")


def make_model(refinement_type: str) -> SiblingRestormer:
    return SiblingRestormer(
        dim=48,
        blocks_per_level=(1, 1, 1, 1, 1),
        heads=(1, 2, 4),
        projection_dim=16,
        refinement_type=refinement_type,
    )


def test_all_refinements_cpu_forward_backward():
    for refinement_type in TYPES:
        torch.manual_seed(13)
        model = make_model(refinement_type)
        image = torch.rand(1, 3, 32, 32, requires_grad=True)
        output = model(image)["restored"]
        assert output.shape == image.shape
        assert torch.isfinite(output).all()
        output.mean().backward()
        assert image.grad is not None and torch.isfinite(image.grad).all()
        if refinement_type != "none":
            refinement_parameters = [p for n, p in model.named_parameters() if n.startswith("refine_module.")]
            assert refinement_parameters
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in refinement_parameters)


def test_zero_initialized_refinements_are_identity():
    torch.manual_seed(13)
    x = torch.randn(1, 48, 16, 16)
    for refinement_type in ("transformer", "naf", "alcrb", "haar", "alcrb_hfrb"):
        model = make_model(refinement_type)
        y = model.refine_module(x)
        assert torch.equal(x, y), refinement_type

    model = make_model("lmrb")
    latent = torch.randn(1, 192, 4, 4)
    assert torch.equal(latent, model.refine_module(latent))


def test_rerh_zero_initialized_correction():
    model = make_model("rerh")
    decoded = torch.randn(1, 48, 16, 16)
    image = torch.rand(1, 3, 16, 16)
    residual = torch.randn(1, 3, 16, 16)
    correction = model.refine_module(decoded, image, residual)
    assert torch.equal(correction, torch.zeros_like(correction))


def test_haar_inverse_reconstructs_even_and_odd_sizes():
    transform = _HaarTransform(scale=0.5)
    for height, width in ((32, 32), (31, 29), (32, 30)):
        image = torch.randn(1, 3, height, width)
        reconstructed = transform.inverse(transform(image), original_size=(height, width))
        assert reconstructed.shape == image.shape
        assert float((image - reconstructed).abs().max()) < 1e-6
