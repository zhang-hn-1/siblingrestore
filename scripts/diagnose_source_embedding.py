from __future__ import annotations

"""B1 diagnosis: does the content embedding actually separate sources?

Zero-training-cost analysis of existing checkpoints (e.g. source_0003,
independent, degradation_001). Extracts the L2-normalized content embedding
(tiled, to cap attention VRAM) for every val source's clean image and its six
degraded views, then reports:

- intra_sibling_cos  : mean pairwise cosine among a source's six degraded views
- inter_sibling_cos  : mean cosine between degraded views of different sources
- own_anchor_cos     : mean cosine from degraded views to their own clean anchor
- cross_anchor_cos   : mean cosine from degraded views to other clean anchors
- retrieval_top1     : degraded-view -> nearest clean anchor, own-source hit rate
- clean_anchor_pairwise_cos : mean pairwise cosine among clean anchors (uniqueness)
- global_pairwise_cos: mean cosine across ALL degraded embeddings (collapse probe)

Run: python scripts/diagnose_source_embedding.py --checkpoint A.pt --checkpoint B.pt ...
"""

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siblingrestore.inference import encode_content_tiled
from siblingrestore.model import SiblingRestormer

DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", help="optional display label per checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("data/plamd_vari_grip_pilot"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("runs/embedding_diagnosis/diagnosis.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


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


def pairwise_cosine_mean(vectors: torch.Tensor) -> float:
    if vectors.shape[0] < 2:
        return float("nan")
    normalized = F.normalize(vectors, dim=-1)
    matrix = normalized @ normalized.transpose(0, 1)
    count = matrix.shape[0]
    off_diagonal = matrix[~torch.eye(count, dtype=torch.bool)]
    return float(off_diagonal.mean())


def diagnose(
    model: SiblingRestormer,
    groups: list[dict],
    data_root: Path,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
) -> dict[str, object]:
    clean_embeddings: list[torch.Tensor] = []
    degraded_embeddings: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for group in groups:
            clean = read_rgb(data_root / group["clean_path"]).unsqueeze(0).to(device)
            clean_embeddings.append(
                encode_content_tiled(model, clean, tile_size, tile_overlap).cpu().squeeze(0)
            )
            views = []
            for degradation in DEGRADATIONS:
                degraded = read_rgb(data_root / group["degraded"][degradation]).unsqueeze(0).to(device)
                views.append(
                    encode_content_tiled(model, degraded, tile_size, tile_overlap).cpu().squeeze(0)
                )
            degraded_embeddings.append(views)

    clean_bank = torch.stack(clean_embeddings)          # [S, D]
    all_degraded = torch.stack([e for views in degraded_embeddings for e in views])  # [S*6, D]
    counts = len(groups)

    intra = [pairwise_cosine_mean(torch.stack(views)) for views in degraded_embeddings]
    intra_sibling_cos = float(np.mean(intra))

    inter_terms = []
    for first, second in combinations(range(counts), 2):
        for first_view, second_view in zip(degraded_embeddings[first], degraded_embeddings[second]):
            inter_terms.append(float(F.cosine_similarity(first_view, second_view, dim=-1)))
    inter_sibling_cos = float(np.mean(inter_terms))

    own_anchor_cos: list[float] = []
    cross_anchor_cos: list[float] = []
    retrieval_hits = 0
    total = 0
    for source_index, views in enumerate(degraded_embeddings):
        for view in views:
            scores = view @ clean_bank.transpose(0, 1)
            own = float(scores[source_index])
            own_anchor_cos.append(own)
            cross = [float(scores[j]) for j in range(counts) if j != source_index]
            cross_anchor_cos.extend(cross)
            if int(scores.argmax()) == source_index:
                retrieval_hits += 1
            total += 1
    retrieval_top1 = retrieval_hits / total

    clean_hits = 0
    clean_anchor_pairwise_cos = pairwise_cosine_mean(clean_bank)

    global_pairwise_cos = pairwise_cosine_mean(all_degraded)

    return {
        "intra_sibling_cos": intra_sibling_cos,
        "inter_sibling_cos": inter_sibling_cos,
        "own_anchor_cos": float(np.mean(own_anchor_cos)),
        "cross_anchor_cos": float(np.mean(cross_anchor_cos)),
        "anchor_margin": float(np.mean(own_anchor_cos)) - float(np.mean(cross_anchor_cos)),
        "retrieval_top1": retrieval_top1,
        "clean_anchor_pairwise_cos": clean_anchor_pairwise_cos,
        "global_pairwise_cos": global_pairwise_cos,
        "embedding_dim": int(clean_bank.shape[1]),
    }


def main() -> None:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.checkpoint):
        raise SystemExit("--label count must match --checkpoint count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    groups = json.loads((args.data_root / "metadata" / "source_groups.json").read_text(encoding="utf-8"))
    groups = [group for group in groups if group["split"] == args.split]

    records = []
    for index, checkpoint_path in enumerate(args.checkpoint):
        label = args.label[index] if args.label else checkpoint_path.parent.name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = make_model(checkpoint, device)
        metrics = diagnose(model, groups, args.data_root, device, args.tile_size, args.tile_overlap)
        records.append({"label": label, "checkpoint": str(checkpoint_path), **metrics})
        print(json.dumps({key: value for key, value in records[-1].items() if not isinstance(value, dict)}))

    report = {
        "metadata": {
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "sources": len(groups),
            "degradations_per_source": len(DEGRADATIONS),
            "tile_size": args.tile_size,
            "embedding_source": "encode_content (content projector), tiled mean, L2-normalized",
            "note": "Diagnostic only; retrieval uses the model's own embedding and is circular for identity-trained methods.",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    keys = [key for key in records[0] if key not in ("label", "checkpoint")]
    lines = [
        "# Embedding Diagnosis (B1)",
        "",
        f"- split: {args.split}, sources: {len(groups)}, degradations per source: {len(DEGRADATIONS)}",
        f"- analysis time (UTC): {report['metadata']['analysis_timestamp_utc']}",
        "",
        "| label | intra_sibling_cos | inter_sibling_cos | own_anchor_cos | cross_anchor_cos | anchor_margin | retrieval_top1 | clean_anchor_pairwise_cos | global_pairwise_cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {label} | {intra_sibling_cos:.4f} | {inter_sibling_cos:.4f} | {own_anchor_cos:.4f} | {cross_anchor_cos:.4f} | {anchor_margin:.4f} | {retrieval_top1:.3f} | {clean_anchor_pairwise_cos:.4f} | {global_pairwise_cos:.4f} |".format(**record)
        )
    lines += [
        "",
        "Reading:",
        "- retrieval_top1 near 1.0 + large positive anchor_margin => the representation separates sources.",
        "- global_pairwise_cos near 1.0 => embedding collapse; the contrastive term is not learning.",
        "- clean_anchor_pairwise_cos near 1.0 => clean anchors are not mutually separated.",
        "- If retrieval_top1 is high but PSNR regressed (source_0003), gradients from contrastive conflict with reconstruction.",
        "- If retrieval_top1 is low even for source_0003, the contrastive loss never formed a useful embedding (temperature/negative design issue).",
    ]
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
