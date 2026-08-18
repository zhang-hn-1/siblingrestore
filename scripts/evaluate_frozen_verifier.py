from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siblingrestore.data import paired_eval_pad
from siblingrestore.inference import restore_tiled
from siblingrestore.metrics import psnr, ssim
from siblingrestore.verifier import load_verifier_checkpoint


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    result = list(range(0, length - tile_size + 1, stride))
    if result[-1] != length - tile_size:
        result.append(length - tile_size)
    return result


def verifier_embed_tiled(verifier, image: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("verifier_embed_tiled expects [1,C,H,W]")
    if tile_size <= 0 or tile_size % 8 or overlap < 0 or overlap >= tile_size:
        raise ValueError("invalid tile size or overlap")
    _, _, height, width = image.shape
    pad_h, pad_w = (8 - height % 8) % 8, (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    tiles = []
    for top in positions(padded.shape[-2], tile_size, overlap):
        for left in positions(padded.shape[-1], tile_size, overlap):
            tiles.append(padded[..., top : top + tile_size, left : left + tile_size])
    batch = torch.cat(tiles, dim=0)
    embeddings = verifier.embed(batch)
    return F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)


def binary_auc(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def eer(scores: list[float], labels: list[int]) -> float:
    thresholds = sorted(set(scores), reverse=True)
    best = 1.0
    for threshold in thresholds:
        predicted = [score >= threshold for score in scores]
        positives = sum(labels); negatives = len(labels) - positives
        fpr = sum(p and not label for p, label in zip(predicted, labels)) / max(negatives, 1)
        fnr = sum(not p and label for p, label in zip(predicted, labels)) / max(positives, 1)
        best = min(best, abs(fpr - fnr) / 2 + min(fpr, fnr))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    restoration = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from train import make_model
    model = make_model(restoration["config"]).to(device).eval()
    model.load_state_dict(restoration["model"])
    verifier, fingerprint = load_verifier_checkpoint(args.verifier, device)
    groups = json.loads((args.data_root / "metadata" / "source_groups.json").read_text())
    groups = [group for group in groups if group["split"] == args.split]
    if args.max_sources:
        groups = groups[: args.max_sources]
    anchors = []
    with torch.no_grad():
        for group in groups:
            anchors.append(verifier_embed_tiled(verifier, read_rgb(args.data_root / group["clean_path"]).unsqueeze(0).to(device), args.tile_size, args.tile_overlap).cpu())
    bank = torch.cat(anchors)
    rows, pair_rows, input_pair_rows, restored_by_source = [], [], [], []
    with torch.no_grad():
        for source_index, group in enumerate(groups):
            clean_original = read_rgb(args.data_root / group["clean_path"])
            views = []
            for degradation in sorted(group["degraded"]):
                degraded_original = read_rgb(args.data_root / group["degraded"][degradation])
                degraded, clean, mask = paired_eval_pad(degraded_original, clean_original)
                restored_padded = restore_tiled(model, degraded.unsqueeze(0).to(device), args.tile_size, args.tile_overlap).cpu()
                restored = restored_padded[..., : clean_original.shape[-2], : clean_original.shape[-1]]
                input_embedding = verifier_embed_tiled(verifier, degraded_original.unsqueeze(0).to(device), args.tile_size, args.tile_overlap).cpu()
                restored_embedding = verifier_embed_tiled(verifier, restored.to(device), args.tile_size, args.tile_overlap).cpu()
                input_scores = (input_embedding @ bank.T).squeeze(0)
                restored_scores = (restored_embedding @ bank.T).squeeze(0)
                input_own = float(input_scores[source_index])
                restored_own = float(restored_scores[source_index])
                input_impostor = max((float(input_scores[index]) for index in range(len(groups)) if index != source_index), default=float("nan"))
                restored_impostor = max((float(restored_scores[index]) for index in range(len(groups)) if index != source_index), default=float("nan"))
                row = {"method": args.method, "seed": args.seed, "source_id": group["source_id"], "subcategory": group["subcategory"], "degradation": degradation, "psnr": psnr(restored_padded, clean.unsqueeze(0), mask.unsqueeze(0)), "ssim": ssim(restored_padded, clean.unsqueeze(0), mask.unsqueeze(0)), "input_to_clean_top1": int(input_scores.argmax() == source_index), "restored_to_clean_top1": int(restored_scores.argmax() == source_index), "input_own_anchor_cos": input_own, "restored_own_anchor_cos": restored_own, "input_nearest_impostor_cos": input_impostor, "restored_nearest_impostor_cos": restored_impostor, "input_margin": input_own - input_impostor, "restored_margin": restored_own - restored_impostor, "identity_gain_margin": (restored_own - input_own) - (restored_impostor - input_impostor)}
                rows.append(row)
                for query_type, scores in (("input", input_scores), ("restored", restored_scores)):
                    for anchor_index, score in enumerate(scores):
                        (input_pair_rows if query_type == "input" else pair_rows).append({"method": args.method, "seed": args.seed, "query_type": query_type, "source_id": group["source_id"], "degradation": degradation, "anchor_source_id": groups[anchor_index]["source_id"], "label": int(anchor_index == source_index), "score": float(score)})
                views.append(restored)
            restored_by_source.append(views)
    pair_scores = [row["score"] for row in pair_rows]
    pair_labels = [row["label"] for row in pair_rows]
    input_pair_scores = [row["score"] for row in input_pair_rows]
    input_pair_labels = [row["label"] for row in input_pair_rows]
    sibling = [float(F.l1_loss(a, b)) for views in restored_by_source for a, b in combinations(views, 2)]
    means = [F.interpolate(torch.stack(views).mean(dim=0), size=(256, 256), mode="bilinear", align_corners=False) for views in restored_by_source]
    cross = [float(F.l1_loss(a, b)) for a, b in combinations(means, 2)]
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in ("psnr", "ssim", "input_to_clean_top1", "restored_to_clean_top1", "input_own_anchor_cos", "restored_own_anchor_cos", "input_nearest_impostor_cos", "restored_nearest_impostor_cos", "input_margin", "restored_margin", "identity_gain_margin")}
    aggregate.update({"restored_roc_auc": binary_auc(pair_scores, pair_labels), "restored_eer": eer(pair_scores, pair_labels), "input_roc_auc": binary_auc(input_pair_scores, input_pair_labels), "input_eer": eer(input_pair_scores, input_pair_labels), "roc_auc_gain": binary_auc(pair_scores, pair_labels) - binary_auc(input_pair_scores, input_pair_labels), "eer_gain": eer(input_pair_scores, input_pair_labels) - eer(pair_scores, pair_labels), "sibling_output_l1": float(np.mean(sibling)), "cross_source_output_l1": float(np.mean(cross)), "intra_inter_output_ratio": float(np.mean(sibling) / np.mean(cross))})
    report = {"status": "independent_verifier_evaluation", "split": args.split, "sources": len(groups), "restored_views": len(rows), "verifier_fingerprint": fingerprint, "aggregate": aggregate, "notes": ["Headline identity metrics use an independent frozen verifier.", "Test remains sealed unless explicitly requested after validation selection."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    all_pairs = input_pair_rows + pair_rows
    with args.output.with_name(args.output.stem + "_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_pairs[0])); writer.writeheader(); writer.writerows(all_pairs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
