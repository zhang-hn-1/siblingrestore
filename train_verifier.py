from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from siblingrestore.data import VerifierImageDataset
from siblingrestore.verifier import SourceVerifier, checkpoint_fingerprint, write_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_groups(root: Path) -> list[dict]:
    return json.loads((root / "metadata" / "source_groups.json").read_text(encoding="utf-8"))


def metadata_hash(root: Path, name: str) -> str:
    return hashlib.sha256((root / "metadata" / name).read_bytes()).hexdigest()


def make_split(groups: list[dict], seed: int, selection_fraction: float) -> dict[str, object]:
    train = [group for group in groups if group["split"] == "train"]
    by_class: dict[str, list[str]] = defaultdict(list)
    for group in train:
        by_class[group["subcategory"]].append(group["source_id"])
    fit, selection = [], []
    for subcategory, source_ids in sorted(by_class.items()):
        ids = sorted(source_ids)
        random.Random(seed + sum(map(ord, subcategory))).shuffle(ids)
        count = max(1, round(len(ids) * selection_fraction))
        selection.extend(sorted(ids[:count]))
        fit.extend(sorted(ids[count:]))
    all_train = {group["source_id"] for group in train}
    if set(fit) & set(selection) or set(fit) | set(selection) != all_train:
        raise ValueError("invalid source selection split")
    return {"algorithm": "stratified_source_split_v1", "seed": seed, "selection_fraction": selection_fraction, "fit_source_ids": sorted(fit), "selection_source_ids": sorted(selection), "fit_count": len(fit), "selection_count": len(selection)}


def loader(root: Path, source_ids: list[str], crop_size: int, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(VerifierImageDataset(root, source_ids, crop_size), batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def retrieval(model: SourceVerifier, root: Path, source_ids: list[str], device: torch.device, tile_size: int = 256) -> dict[str, float]:
    groups = {group["source_id"]: group for group in read_groups(root)}
    from siblingrestore.data import _read_rgb
    anchors, views = [], []
    for source_id in sorted(source_ids):
        group = groups[source_id]
        clean = _read_rgb(root / group["clean_path"]).unsqueeze(0).to(device)
        anchors.append(model.embed(F.interpolate(clean, size=(tile_size, tile_size), mode="bilinear", align_corners=False)).cpu())
        for path in group["degraded"].values():
            image = _read_rgb(root / path).unsqueeze(0).to(device)
            views.append((source_id, model.embed(F.interpolate(image, size=(tile_size, tile_size), mode="bilinear", align_corners=False)).cpu()))
    bank = torch.cat(anchors)
    hits, own, cross = 0, [], []
    for source_id, embedding in views:
        scores = (embedding @ bank.T).squeeze(0)
        index = sorted(source_ids).index(source_id)
        hits += int(scores.argmax() == index)
        own.append(float(scores[index]))
        cross.extend(float(value) for i, value in enumerate(scores) if i != index)
    pairwise = bank @ bank.T
    offdiag = pairwise[~torch.eye(len(bank), dtype=torch.bool)]
    return {"retrieval_top1": hits / len(views), "own_anchor_cos": float(np.mean(own)), "cross_anchor_cos": float(np.mean(cross)), "anchor_margin": float(np.mean(own) - np.mean(cross)), "clean_anchor_pairwise_cos": float(offdiag.mean())}


def train_phase(model: SourceVerifier, train_loader: DataLoader, device: torch.device, steps: int, learning_rate: float) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    step = 0
    while step < steps:
        for batch in train_loader:
            image = batch["image"].to(device)
            labels = batch["source_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image)["logits"], labels)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if config.get("device", "auto") == "cuda" and torch.cuda.is_available() else "cpu")
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    groups = read_groups(root)
    split = make_split(groups, int(config.get("selection_seed", seed)), float(config.get("selection_fraction", 0.2)))
    split["metadata_hashes"] = {name: metadata_hash(root, name) for name in ("source_groups.json", "pilot_index.csv")}
    split["excluded_split_source_ids"] = sorted(group["source_id"] for group in groups if group["split"] != "train")
    (output_dir / "selection_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    model_config = config["model"]
    selection_model = SourceVerifier(source_count=len(split["fit_source_ids"]), embedding_dim=int(model_config.get("embedding_dim", 128))).to(device)
    train_phase(selection_model, loader(root, split["fit_source_ids"], int(config.get("crop_size", 256)), int(config.get("batch_size", 16)), int(config.get("num_workers", 0)), True), device, int(config.get("selection_steps", 200)), float(config.get("learning_rate", 2e-4)))
    selection_model.eval()
    selection_metrics = retrieval(selection_model, root, split["selection_source_ids"], device)
    all_train = sorted(group["source_id"] for group in groups if group["split"] == "train")
    final_model = SourceVerifier(source_count=len(all_train), embedding_dim=int(model_config.get("embedding_dim", 128))).to(device)
    start = time.time()
    train_phase(final_model, loader(root, all_train, int(config.get("crop_size", 256)), int(config.get("batch_size", 16)), int(config.get("num_workers", 0)), True), device, int(config.get("final_steps", 500)), float(config.get("learning_rate", 2e-4)))
    final_model.eval()
    checkpoint = {"model": final_model.state_dict(), "config": config, "selection_metrics": selection_metrics, "train_source_ids": all_train}
    torch.save(checkpoint, output_dir / "best.pt")
    torch.save(checkpoint, output_dir / "last.pt")
    diagnostics = {"selection": selection_metrics, "status": "passed" if selection_metrics["anchor_margin"] > 0.0 else "failed_no_retrieval_separation"}
    (output_dir / "verifier_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "verifier_diagnostics.csv").write_text("metric,value\n" + "\n".join(f"{key},{value}" for key, value in selection_metrics.items()) + "\n", encoding="utf-8")
    resolved = dict(config); resolved["selection_split"] = split
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    summary = {"status": diagnostics["status"], "selection_metrics": selection_metrics, "steps": int(config.get("final_steps", 500)), "elapsed_seconds": time.time() - start, "parameters": sum(p.numel() for p in final_model.parameters())}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fingerprint = write_fingerprint(output_dir / "best.pt", str(config.get("role", "verifier")), resolved)
    print(json.dumps({"summary": summary, "fingerprint": fingerprint}, ensure_ascii=False))
    if diagnostics["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
