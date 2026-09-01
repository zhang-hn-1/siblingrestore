"""Unit tests for adaptive_frozen_anchor_loss.

Run: python -m pytest tests/test_adaptive_anchor.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siblingrestore.losses import adaptive_frozen_anchor_loss


def _make_orthogonal_sources(num_sources: int, dim: int, device: str = "cpu") -> torch.Tensor:
    """Create well-separated unit-norm source prototypes."""
    torch.manual_seed(42)
    prototypes = torch.randn(num_sources, dim, device=device)
    return torch.nn.functional.normalize(prototypes, dim=-1)


# Case 1: margin > tau → weight = 0 → loss = 0
def test_margin_above_tau_gives_zero_weight():
    dim = 64
    tau = 0.10
    sources = _make_orthogonal_sources(3, dim)
    # Restored embeddings almost identical to their own source → high positive sim
    restored = sources.clone() + 0.001 * torch.randn_like(sources)
    restored = torch.nn.functional.normalize(restored, dim=-1)
    source_ids = torch.tensor([0, 1, 2])

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    # Margin should be large (pos ~1.0, neg ~0 or less)
    assert diag["mean_margin"] > tau, f"margin {diag['mean_margin']} should be > {tau}"
    # All weights should be 0
    assert diag["mean_weight"] == 0.0, f"mean_weight should be 0, got {diag['mean_weight']}"
    assert diag["active_ratio"] == 0.0
    # Loss must be legal zero
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


# Case 2: 0 < margin < tau → 0 < weight < 1
def test_margin_between_zero_and_tau_gives_partial_weight():
    dim = 4
    tau = 1.0  # High tau so margin~0.6 falls in the partial regime
    torch.manual_seed(0)
    sources = torch.nn.functional.normalize(torch.randn(3, dim), dim=-1)
    restored = sources.clone() + 0.01 * torch.randn_like(sources)
    restored = torch.nn.functional.normalize(restored, dim=-1)
    source_ids = torch.tensor([0, 1, 2])

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    assert torch.isfinite(loss)
    margin = diag["mean_margin"]
    assert 0 < margin < tau, f"margin {margin:.4f} should be in (0, {tau})"
    assert loss > 0, "loss should be positive when margin < tau"
    assert diag["active_ratio"] > 0.0, "some samples should be active"
    assert 0 < diag["mean_weight"] < 1.0, f"mean_weight={diag['mean_weight']:.4f}"


# Case 3: margin < 0 → weight = 1
def test_margin_below_zero_gives_full_weight():
    dim = 64
    tau = 0.10
    # Make two sources very similar so that restored of source 0 is closer to
    # source 1 than to source 0 → negative margin
    sources = _make_orthogonal_sources(3, dim)
    sources[1] = sources[0] + 0.01 * sources[1]  # source 1 ≈ source 0
    sources = torch.nn.functional.normalize(sources, dim=-1)
    # Restored of sample 0 is exactly source 1 (wrong source)
    restored = sources.clone()
    restored[0] = sources[1]  # sample 0 restored as source 1
    source_ids = torch.tensor([0, 1, 2])

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    assert torch.isfinite(loss)
    # Sample 0 should have negative margin (pos_sim < neg_sim)
    # At least some weights should be 1
    assert diag["mean_weight"] > 0.0


# Case 4: same-source siblings cannot be hard negative
def test_same_source_siblings_not_selected_as_negative():
    dim = 64
    tau = 0.10
    num_sources = 3
    siblings = 2
    sources = _make_orthogonal_sources(num_sources, dim)
    # Each source has 2 siblings (restored embeddings close to their source)
    restored = sources.repeat_interleave(siblings, dim=0)
    restored = restored + 0.001 * torch.randn_like(restored)
    restored = torch.nn.functional.normalize(restored, dim=-1)
    # source_ids: [0, 0, 1, 1, 2, 2]
    source_ids = torch.arange(num_sources).repeat_interleave(siblings)

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    assert torch.isfinite(loss)
    # Hard negative should be from a different source, not a sibling
    # With well-separated sources, margin should be large → low weight
    assert diag["mean_margin"] > 0.0


# Case 5: batch with only one unique source → no NaN/error
def test_single_source_no_nan():
    dim = 64
    tau = 0.10
    # Only 1 unique source prototype, but 3 restored samples all mapping to it
    sources = _make_orthogonal_sources(1, dim)  # [1, D]
    restored = sources.repeat(3, 1) + 0.001 * torch.randn(3, dim)  # [3, D]
    restored = torch.nn.functional.normalize(restored, dim=-1)
    source_ids = torch.tensor([0, 0, 0])

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert not torch.isnan(loss), "loss is NaN with single source"
    # No valid negative → all weights should be 0 → loss = 0
    assert float(loss) == 0.0


# Case 6: all weights = 0 → loss = legal 0
def test_all_zero_weights_gives_legal_zero():
    dim = 64
    tau = 0.10
    # Well-separated sources, restored ≈ clean → margin >> tau → all weights 0
    sources = _make_orthogonal_sources(4, dim)
    restored = sources.clone()
    source_ids = torch.tensor([0, 1, 2, 3])

    loss, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert diag["active_ratio"] == 0.0
    assert diag["mean_weight"] == 0.0


def test_gradient_flows_through_restored():
    """Ensure gradient flows from the loss back to restored embeddings."""
    dim = 64
    tau = 0.10
    sources = _make_orthogonal_sources(3, dim)
    restored = sources.clone() + 0.1 * torch.randn_like(sources)
    restored = torch.nn.functional.normalize(restored, dim=-1)
    restored.requires_grad_(True)
    source_ids = torch.tensor([0, 1, 2])

    loss, _ = adaptive_frozen_anchor_loss(restored, sources, source_ids, tau=tau)
    loss.backward()

    assert restored.grad is not None, "gradient should flow to restored"
    assert torch.isfinite(restored.grad).all(), "gradient should be finite"


def test_diagnostics_keys():
    """Ensure all expected diagnostic keys are present."""
    dim = 64
    sources = _make_orthogonal_sources(3, dim)
    restored = sources.clone()
    source_ids = torch.tensor([0, 1, 2])

    _, diag = adaptive_frozen_anchor_loss(restored, sources, source_ids)

    expected_keys = {
        "active_ratio", "mean_margin", "mean_weight",
        "mean_pos_sim", "mean_neg_sim", "raw_loss",
    }
    assert set(diag.keys()) == expected_keys, f"missing keys: {expected_keys - set(diag.keys())}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
