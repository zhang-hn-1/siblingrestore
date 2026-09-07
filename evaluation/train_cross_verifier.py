from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siblingrestore.data import VerifierImageDataset


class CrossVerifier(nn.Module):
    """ResNet18 identity verifier trained from scratch on SFR v1."""

    def __init__(self, source_count: int, embedding_dim: int = 128):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Linear(in_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, source_count)

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.projector(self.backbone(image)), dim=-1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.embed(image)
        return {"embedding": embedding, "logits": self.classifier(embedding)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/plamd_sfr_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/identity_evaluation/cross_verifier"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    groups = json.loads((args.data_root / "metadata" / "source_groups.json").read_text())
    source_ids = sorted(g["source_id"] for g in groups if g["split"] == "train")
    dataset = VerifierImageDataset(args.data_root, source_ids, crop_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    model = CrossVerifier(len(source_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"data_root": str(args.data_root), "split": "train", "source_count": len(source_ids), "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "architecture": "torchvision.resnet18(weights=None)+128d projector", "device": str(device)}
    (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.output_dir / "train.log").open("w", encoding="utf-8") as log:
        iterator = iter(loader)
        model.train()
        for step in range(1, args.steps + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            image = batch["image"].to(device)
            target = batch["source_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(image)
            loss = nn.functional.cross_entropy(result["logits"], target)
            loss.backward(); optimizer.step()
            if step == 1 or step % 100 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach())}
                log.write(json.dumps(record) + "\n"); log.flush(); print(record, flush=True)
    checkpoint = {"model": model.state_dict(), "config": config, "source_ids": source_ids}
    checkpoint_path = args.output_dir / "cross_verifier_resnet18.pt"
    torch.save(checkpoint, checkpoint_path)
    fingerprint = {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}
    (args.output_dir / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
    print(json.dumps(fingerprint, indent=2))


if __name__ == "__main__":
    main()
