from __future__ import annotations

import torch
import torch.nn.functional as F


def optional_loss(weight: float, factory):
    """Lazily build an auxiliary loss so disabled terms stay out of the graph."""
    if weight <= 0.0:
        return None
    return factory()


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + epsilon**2).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, true_x) + F.l1_loss(pred_y, true_y)


def source_contrastive_loss(
    content: torch.Tensor,
    clean_anchor: torch.Tensor,
    class_ids: torch.Tensor,
    temperature: float = 0.1,
    same_class_negative_weight: float = 2.0,
) -> torch.Tensor:
    """Match every degraded sibling to its own clean-source anchor.

    content: [B, K, D], clean_anchor: [B, D], class_ids: [B]
    """
    batch, siblings, _ = content.shape
    if batch < 2:
        raise ValueError("source contrastive loss requires at least two source groups per batch")
    queries = F.normalize(content.reshape(batch * siblings, -1), dim=-1)
    anchors = F.normalize(clean_anchor, dim=-1)
    logits = queries @ anchors.transpose(0, 1) / temperature
    query_classes = class_ids[:, None].expand(batch, siblings).reshape(-1)
    same_class = query_classes[:, None].eq(class_ids[None, :])
    own_source = torch.arange(batch, device=content.device).repeat_interleave(siblings)
    hard_negative = same_class & ~F.one_hot(own_source, num_classes=batch).bool()
    logits = logits + hard_negative.float() * torch.log(
        torch.tensor(same_class_negative_weight, device=content.device)
    )
    return F.cross_entropy(logits, own_source)


def sibling_output_consistency(restored: torch.Tensor) -> torch.Tensor:
    """Pairwise L1 consistency among aligned outputs [B, K, C, H, W]."""
    siblings = restored.shape[1]
    terms = [
        F.l1_loss(restored[:, first], restored[:, second])
        for first in range(siblings)
        for second in range(first + 1, siblings)
    ]
    return torch.stack(terms).mean()




def frozen_anchor_loss(restored_embedding: torch.Tensor, clean_embedding: torch.Tensor) -> torch.Tensor:
    """Match restored embeddings to detached clean-source verifier anchors."""
    if restored_embedding.shape != clean_embedding.shape:
        raise ValueError(f"anchor shape mismatch: {restored_embedding.shape} vs {clean_embedding.shape}")
    restored_embedding = F.normalize(restored_embedding.float(), dim=-1)
    clean_embedding = F.normalize(clean_embedding.float().detach(), dim=-1)
    return (1.0 - F.cosine_similarity(restored_embedding, clean_embedding, dim=-1)).mean()


def verifier_sibling_consensus(
    features: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """Same-source sibling pull + cross-source push in frozen verifier space.

    Unlike pulling restored outputs toward the clean anchor (which competes
    with pixel reconstruction), this only requires that restoration outputs of
    the SAME source under DIFFERENT degradations share a consistent identity
    embedding, and that different sources stay separated. This is the
    SiblingRestore core assumption and does not fight the reconstruction
    objective.

    features: [B, K, D] frozen-verifier embeddings of restored siblings.
    """
    batch, siblings, dim = features.shape
    normalized = F.normalize(features.float(), dim=-1)
    pulls = [
        F.mse_loss(normalized[batch_index, first], normalized[batch_index, second])
        for batch_index in range(batch)
        for first in range(siblings)
        for second in range(first + 1, siblings)
    ]
    pushes = []
    for first_batch in range(batch):
        for second_batch in range(first_batch + 1, batch):
            cosine = (normalized[first_batch] * normalized[second_batch]).sum(dim=-1)
            pushes.append(torch.clamp(cosine - margin, min=0.0).mean())
    loss = torch.stack(pulls).mean() if pulls else normalized.sum() * 0.0
    if pushes:
        loss = loss + torch.stack(pushes).mean()
    return loss


def multi_scale_anchor_loss(
    restored_features: list[torch.Tensor],
    clean_features: list[torch.Tensor],
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-scale verifier anchor: per-channel L2-normalized spatial feature
    matching between restored and clean images across the first three verifier
    stages. Prevents the global-pooling shortcut that lets the model satisfy
    identity constraints by smoothing local structure."""
    if len(restored_features) != len(clean_features) or len(restored_features) != len(weights):
        raise ValueError("MSA requires aligned feature lists and per-stage weights")
    total = 0.0
    count = 0
    for restored, clean, weight in zip(restored_features, clean_features, weights):
        restored = restored.float()
        clean = clean.float().detach()
        if restored.shape != clean.shape:
            raise ValueError(f"MSA stage shape mismatch: {restored.shape} vs {clean.shape}")
        batch, channels, height, width = restored.shape
        restored_norm = torch.sqrt(restored.pow(2).sum(dim=(-2, -1), keepdim=True) + eps)
        clean_norm = torch.sqrt(clean.pow(2).sum(dim=(-2, -1), keepdim=True) + eps)
        restored_bar = restored / restored_norm
        clean_bar = clean / clean_norm
        stage_loss = F.l1_loss(restored_bar, clean_bar)
        total = total + float(weight) * stage_loss
        count += 1
    return total


def project_conflicting_gradient(
    anchor: torch.Tensor, reconstruction: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, bool, float, float]:
    """Project anchor gradient away from reconstruction only on conflict."""
    dot = torch.dot(anchor, reconstruction)
    rec_sq = torch.dot(reconstruction, reconstruction).clamp_min(eps)
    before = float(dot.detach().cpu())
    if before < 0.0:
        projected = anchor - dot / rec_sq * reconstruction
        return projected, True, before, float(torch.dot(projected, reconstruction).detach().cpu())
    return anchor, False, before, before


def gentle_identity_loss(
    feature: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """No-exponential identity pull/push on [B, K, D] normalized features.

    Same-source siblings are pulled together with MSE (monotone in cosine);
    different sources are pushed apart with a margin hinge. Unlike InfoNCE at
    low temperature this stays gentle, matching the degradation-CE recipe that
    proved stable at extreme gradient ratios.
    """
    batch, siblings, _ = feature.shape
    pulls = [
        F.mse_loss(feature[batch_index, first], feature[batch_index, second])
        for batch_index in range(batch)
        for first in range(siblings)
        for second in range(first + 1, siblings)
    ]
    pushes = []
    for first_batch in range(batch):
        for second_batch in range(first_batch + 1, batch):
            cosine = (feature[first_batch] * feature[second_batch]).sum(dim=-1)
            pushes.append(torch.clamp(cosine - margin, min=0.0).mean())
    loss = torch.stack(pulls).mean() if pulls else feature.sum() * 0.0
    if pushes:
        loss = loss + torch.stack(pushes).mean()
    return loss
