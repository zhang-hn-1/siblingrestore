from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SourceVerifier(nn.Module):
    """Compact frozen source verifier used outside the restoration backbone."""

    def __init__(self, source_count: int = 71, embedding_dim: int = 128) -> None:
        super().__init__()
        self.source_count = int(source_count)
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, self.embedding_dim)
        )
        self.classifier = nn.Linear(self.embedding_dim, self.source_count)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(image)
        embedding = F.normalize(self.projector(hidden), dim=-1)
        return {"embedding": embedding, "logits": self.classifier(embedding)}

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        return self(image)["embedding"]

    def forward_stages(self, image: torch.Tensor, num_stages: int = 3) -> list[torch.Tensor]:
        """Return spatial feature maps from the first `num_stages` encoder
        stages without touching the projector/classifier heads.

        Adds forward logic only; parameter structure and state dict keys are
        unchanged, so existing verifier checkpoints load exactly as before.
        """
        if not 1 <= num_stages <= 4:
            raise ValueError("num_stages must be between 1 and 4")
        features = []
        hidden = image
        for index, stage in enumerate(self.encoder):
            hidden = stage(hidden)
            if index < num_stages:
                features.append(hidden)
        return features


def freeze_verifier(verifier: SourceVerifier) -> SourceVerifier:
    verifier.eval()
    for parameter in verifier.parameters():
        parameter.requires_grad_(False)
    return verifier


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verifier_checkpoint(
    path: Path, device: torch.device, expected_sha256: str | None = None
) -> tuple[SourceVerifier, dict[str, object]]:
    path = Path(path)
    actual = checkpoint_fingerprint(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"verifier fingerprint mismatch: expected {expected_sha256}, got {actual}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", config)
    verifier = SourceVerifier(
        source_count=int(model_config.get("source_count", 71)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
    )
    verifier.load_state_dict(checkpoint["model"])
    freeze_verifier(verifier.to(device))
    metadata = {
        "path": str(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
        "source_count": verifier.source_count,
        "embedding_dim": verifier.embedding_dim,
    }
    return verifier, metadata


def write_fingerprint(path: Path, role: str, config: dict[str, object]) -> dict[str, object]:
    record = {
        "role": role,
        "checkpoint": str(path),
        "sha256": checkpoint_fingerprint(path),
        "bytes": Path(path).stat().st_size,
        "config": config,
    }
    Path(path).with_name("checkpoint_fingerprint.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record
