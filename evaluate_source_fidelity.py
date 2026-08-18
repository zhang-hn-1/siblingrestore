from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from siblingrestore.data import CLASS_TO_ID, paired_eval_pad
from siblingrestore.inference import identity_embed_tiled, restore_tiled
from siblingrestore.metrics import psnr, ssim
from siblingrestore.model import SiblingRestormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def descriptor(image: torch.Tensor) -> torch.Tensor:
    pooled = F.interpolate(image, size=(16, 16), mode="bilinear", align_corners=False)
    gray = 0.299 * pooled[:, :1] + 0.587 * pooled[:, 1:2] + 0.114 * pooled[:, 2:3]
    grad_x = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    vector = torch.cat([pooled, grad_x, grad_y], dim=1).flatten(1)
    return F.normalize(vector, dim=-1)


def make_model(checkpoint: dict[str, object], device: torch.device) -> SiblingRestormer:
    model_config = checkpoint["config"]["model"]
    model = SiblingRestormer(
        dim=int(model_config["dim"]),
        blocks_per_level=tuple(model_config["blocks_per_level"]),
        heads=tuple(model_config["heads"]),
        projection_dim=int(model_config["projection_dim"]),
        degradation_conditioned=bool(model_config.get("degradation_conditioned", False)),
        identity_mode=model_config.get("identity_mode", None),
        identity_source_count=int(model_config.get("identity_source_count", 71)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(checkpoint, device)
    groups = json.loads(
        (args.data_root / "metadata" / "source_groups.json").read_text(encoding="utf-8")
    )
    groups = [group for group in groups if group["split"] == args.split]
    if args.max_sources:
        groups = groups[: args.max_sources]

    clean_descriptors: list[torch.Tensor] = []
    clean_classes: list[int] = []
    for group in groups:
        clean = read_rgb(args.data_root / group["clean_path"]).unsqueeze(0).to(device)
        clean_descriptors.append(descriptor(clean).cpu())
        clean_classes.append(CLASS_TO_ID[group["subcategory"]])
    clean_bank = torch.cat(clean_descriptors, dim=0)

    clean_embeddings = [
        identity_embed_tiled(
            model,
            read_rgb(args.data_root / group["clean_path"]).unsqueeze(0).to(device),
            args.tile_size,
            args.tile_overlap,
        ).cpu()
        for group in groups
    ]
    clean_embed_bank = torch.cat(clean_embeddings, dim=0)

    rows: list[dict[str, object]] = []
    restored_by_source: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for source_index, group in enumerate(groups):
            clean_original = read_rgb(args.data_root / group["clean_path"])
            restored_views: list[torch.Tensor] = []
            for degradation, relative in sorted(group["degraded"].items()):
                degraded_original = read_rgb(args.data_root / relative)
                height, width = clean_original.shape[-2:]
                degraded, clean, mask = paired_eval_pad(degraded_original, clean_original)
                restored = restore_tiled(
                    model,
                    degraded.unsqueeze(0).to(device),
                    args.tile_size,
                    args.tile_overlap,
                ).cpu()
                restored_views.append(restored[..., :height, :width])
                restored_descriptor = descriptor(restored[..., :height, :width])
                similarities = restored_descriptor @ clean_bank.transpose(0, 1)
                predicted_source = int(similarities.argmax(dim=1))
                same_class_indices = [
                    index for index, class_id in enumerate(clean_classes)
                    if class_id == clean_classes[source_index]
                ]
                same_class_scores = similarities[:, same_class_indices]
                predicted_same_class = same_class_indices[int(same_class_scores.argmax(dim=1))]
                embed = identity_embed_tiled(
                    model,
                    restored[..., :height, :width].to(device),
                    args.tile_size,
                    args.tile_overlap,
                ).cpu()
                embed_scores = embed @ clean_embed_bank.transpose(0, 1)
                embed_own_clean_top1 = int(embed_scores.argmax(dim=1) == source_index)
                rows.append(
                    {
                        "source_id": group["source_id"],
                        "subcategory": group["subcategory"],
                        "degradation": degradation,
                        "psnr": psnr(restored, clean.unsqueeze(0), mask.unsqueeze(0)),
                        "ssim": ssim(restored, clean.unsqueeze(0), mask.unsqueeze(0)),
                        "own_clean_top1": int(predicted_source == source_index),
                        "own_clean_same_class_top1": int(predicted_same_class == source_index),
                        "embed_own_clean_top1": embed_own_clean_top1,
                    }
                )
            restored_by_source.append(restored_views)

    source_consistency = [
        float(np.mean([float(F.l1_loss(a, b)) for a, b in combinations(views, 2)]))
        for views in restored_by_source
    ]
    # Sources have different native sizes; resize source means to a common
    # reference before computing the cross-source distance.
    mean_restored_per_source = [
        F.interpolate(
            torch.stack(views).mean(dim=0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        for views in restored_by_source
    ]
    cross_source_terms = [
        float(F.l1_loss(mean_restored_per_source[first], mean_restored_per_source[second]))
        for first in range(len(mean_restored_per_source))
        for second in range(first + 1, len(mean_restored_per_source))
    ]
    cross_source_output_l1 = float(np.mean(cross_source_terms))
    sibling_output_l1 = float(np.mean(source_consistency))
    intra_inter_output_ratio = sibling_output_l1 / cross_source_output_l1 if cross_source_output_l1 > 0 else None

    def mean(selected: list[dict[str, object]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in selected]))

    by_degradation: dict[str, dict[str, float]] = {}
    for degradation in sorted({str(row["degradation"]) for row in rows}):
        selected = [row for row in rows if row["degradation"] == degradation]
        by_degradation[degradation] = {
            key: mean(selected, key)
            for key in ("psnr", "ssim", "own_clean_top1", "own_clean_same_class_top1", "embed_own_clean_top1")
        }
    report = {
        "status": "diagnostic_not_publication_ready",
        "split": args.split,
        "sources": len(groups),
        "restored_views": len(rows),
        "aggregate": {
            "psnr": mean(rows, "psnr"),
            "ssim": mean(rows, "ssim"),
            "own_clean_top1": mean(rows, "own_clean_top1"),
            "own_clean_same_class_top1": mean(rows, "own_clean_same_class_top1"),
            "embed_own_clean_top1": mean(rows, "embed_own_clean_top1"),
            "sibling_output_l1": sibling_output_l1,
            "cross_source_output_l1": cross_source_output_l1,
            "intra_inter_output_ratio": intra_inter_output_ratio,
        },
        "by_degradation": by_degradation,
        "notes": [
            "Retrieval uses a fixed color/gradient descriptor and is a pilot diagnostic.",
            "embed_own_clean_top1 uses the model's own content embedding; for identity-trained methods this is circular and serves as a representation check, not a PSNR claim.",
            "intra_inter_output_ratio = sibling_output_l1 / cross_source_output_l1; lower means identity is more compact relative to inter-source distance.",
            "Lower sibling_output_l1 is useful only when clean-reference metrics do not regress.",
            "Replace global SSIM and fixed retrieval with formal metrics before publication.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
