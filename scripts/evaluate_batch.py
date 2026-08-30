from __future__ import annotations

"""Batch evaluation pipeline: parallel decode + bucketed batched restore +
tiled verifier embedding. Cuts 480-view evaluation from 20-40 min to ~3-5 min.

Usage: python scripts/evaluate_batch.py --checkpoint X --verifier Y \
       --data-root data/plamd_vari_grip_500 --output out.json --split val \
       --device cuda --method NAME --seed 13 [--tile-size 512]
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from siblingrestore.metrics import psnr, ssim
from siblingrestore.verifier import load_verifier_checkpoint
from scripts.evaluate_frozen_verifier import binary_auc, eer, positions, read_rgb, verifier_embed_tiled

BUCKETS = ((0, 384), (384, 512), (512, 640), (640, 768), (768, 896), (896, 2048))


def bucket_for(h: int, w: int) -> int:
    for index, (low, high) in enumerate(BUCKETS):
        if low < max(h, w) <= high:
            return index
    return len(BUCKETS) - 1


def bucket_batch(index: int) -> int:
    if index >= 3:
        return 2
    if index == 2:
        return 4
    return 8


def decode_pair(args) -> tuple[torch.Tensor, torch.Tensor]:
    degraded_full, clean_full = args
    degraded = read_rgb(degraded_full)
    clean = read_rgb(clean_full)
    return degraded, clean


def embed_batch(verifier, tensors, tile_size: int, overlap: int, device, batch: int = 16) -> list[torch.Tensor]:
    """Batched verifier embedding preserving tiled semantics for large images.
    Requests are grouped by spatial shape so each chunk stacks identically-sized
    tiles (small images stay unpadded; tiled images yield full-size tiles)."""
    requests: dict[tuple[int, int], list] = {}
    for index, image in enumerate(tensors):
        if image.ndim == 4:
            image = image.squeeze(0)  # inputs arrive as (1, C, H, W)
        _, height, width = image.shape
        pad_h, pad_w = (8 - height % 8) % 8, (8 - width % 8) % 8
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
        if padded.shape[-2] <= tile_size and padded.shape[-1] <= tile_size:
            requests.setdefault(padded.shape, []).append((index, padded))
        else:
            for top in positions(padded.shape[-2], tile_size, overlap):
                for left in positions(padded.shape[-1], tile_size, overlap):
                    tile = padded[..., top : top + tile_size, left : left + tile_size]
                    requests.setdefault(tile.shape, []).append((index, tile))
    results: dict[int, list[torch.Tensor]] = {}
    for _, chunk in requests.items():
        for start in range(0, len(chunk), batch):
            items = chunk[start : start + batch]
            images = torch.stack([item[1] for item in items]).to(device)
            embeddings = verifier.embed(images)
            for (index, _), embedding in zip(items, embeddings):
                results.setdefault(index, []).append(embedding.cpu())
    return [F.normalize(torch.stack(results[index]).mean(dim=0, keepdim=True), dim=-1) for index in range(len(tensors))]


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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=0)
    args = parser.parse_args()
    t0 = time.time()
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

    # 1. Parallel decode all pairs.
    views = []
    for source_index, group in enumerate(groups):
        for degradation in sorted(group["degraded"]):
            views.append((args.data_root / group["degraded"][degradation], args.data_root / group["clean_path"], source_index, degradation))
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        decoded = list(executor.map(lambda v: (decode_pair(v[:2]), v[2], v[3]), views))
    print(f"[timing] decode {len(views)} views: {time.time() - t0:.1f}s", flush=True)
    # 2. Bucketed batched restore.
    buckets: dict[int, list] = {}
    decoded_map = {}
    for (degraded, clean), source_index, degradation in decoded:
        decoded_map[(source_index, degradation)] = degraded
        degraded_pad, clean_pad, mask = paired_eval_pad(degraded, clean)
        size_index = bucket_for(degraded_pad.shape[-2], degraded_pad.shape[-1])
        buckets.setdefault(size_index, []).append((degraded_pad, clean_pad, mask, source_index, degradation))
    restored_map = {}
    clean_pad_map = {}
    mask_map = {}
    with torch.no_grad():
        for size_index, items in buckets.items():
            low, high = BUCKETS[size_index]
            target = high
            batch = bucket_batch(size_index)
            for start in range(0, len(items), batch):
                chunk = items[start : start + batch]
                padded_batch = []
                original_sizes = []
                for degraded_pad, *_ in chunk:
                    h, w = degraded_pad.shape[-2:]
                    # Reflect-pad to the bucket target: zero padding would drag
                    # GroupNorm statistics and hurt restoration fidelity.
                    # Reflect requires pad < dim; fall back to replicate otherwise.
                    dh, dw = target - h, target - w
                    mode_h = "reflect" if 0 < dh < h else ("replicate" if dh else "constant")
                    mode_w = "reflect" if 0 < dw < w else ("replicate" if dw else "constant")
                    if dh and dw:
                        padded = F.pad(degraded_pad, (0, dw, 0, dh), mode=mode_h if mode_h == mode_w else "replicate")
                    elif dh:
                        padded = F.pad(degraded_pad, (0, 0, 0, dh), mode=mode_h)
                    elif dw:
                        padded = F.pad(degraded_pad, (0, dw, 0, 0), mode=mode_w)
                    else:
                        padded = degraded_pad
                    padded_batch.append(padded)
                    original_sizes.append((h, w))
                out = model(torch.stack(padded_batch).to(device))
                restored_batch = out["restored"] if isinstance(out, dict) else out
                for (degraded_pad, clean_pad, mask, source_index, degradation), restored, (h, w) in zip(chunk, restored_batch, original_sizes):
                    restored_map[(source_index, degradation)] = restored[..., :h, :w].cpu()
                    clean_pad_map[(source_index, degradation)] = clean_pad
                    mask_map[(source_index, degradation)] = mask

    # 3. Batched verifier embedding (clean bank + all views at once).
    clean_tensors = [read_rgb(args.data_root / group["clean_path"]).unsqueeze(0) for group in groups]
    bank = torch.cat(embed_batch(verifier, clean_tensors, args.tile_size, args.tile_overlap, device))
    view_pairs = []
    for source_index, group in enumerate(groups):
        for degradation in sorted(group["degraded"]):
            degraded = decoded_map[(source_index, degradation)].unsqueeze(0)
            restored = restored_map[(source_index, degradation)].unsqueeze(0)
            view_pairs.append((source_index, degradation, degraded, restored))
    input_embeddings = embed_batch(verifier, [pair[2] for pair in view_pairs], args.tile_size, args.tile_overlap, device)
    restored_embeddings = embed_batch(verifier, [pair[3] for pair in view_pairs], args.tile_size, args.tile_overlap, device)
    view_embeddings = {(source_index, degradation): (input_embedding, restored_embedding) for (source_index, degradation, _, _), input_embedding, restored_embedding in zip(view_pairs, input_embeddings, restored_embeddings)}

    rows, pair_rows, input_pair_rows, restored_by_source = [], [], [], []
    with torch.no_grad():
        for source_index, group in enumerate(groups):
            views_for_source = []
            for degradation in sorted(group["degraded"]):
                restored = restored_map[(source_index, degradation)]
                input_embedding, restored_embedding = view_embeddings[(source_index, degradation)]
                input_scores = (input_embedding @ bank.T).squeeze(0)
                restored_scores = (restored_embedding @ bank.T).squeeze(0)
                input_own = float(input_scores[source_index])
                restored_own = float(restored_scores[source_index])
                input_impostor = max((float(input_scores[index]) for index in range(len(groups)) if index != source_index), default=float("nan"))
                restored_impostor = max((float(restored_scores[index]) for index in range(len(groups)) if index != source_index), default=float("nan"))
                row = {"method": args.method, "seed": args.seed, "source_id": group["source_id"], "subcategory": group["subcategory"], "degradation": degradation, "psnr": psnr(restored.unsqueeze(0), clean_pad_map[(source_index, degradation)].unsqueeze(0), mask_map[(source_index, degradation)].unsqueeze(0)), "ssim": ssim(restored.unsqueeze(0), clean_pad_map[(source_index, degradation)].unsqueeze(0), mask_map[(source_index, degradation)].unsqueeze(0)), "input_to_clean_top1": int(input_scores.argmax() == source_index), "restored_to_clean_top1": int(restored_scores.argmax() == source_index), "input_own_anchor_cos": input_own, "restored_own_anchor_cos": restored_own, "input_nearest_impostor_cos": input_impostor, "restored_nearest_impostor_cos": restored_impostor, "input_margin": input_own - input_impostor, "restored_margin": restored_own - restored_impostor, "identity_gain_margin": (restored_own - input_own) - (restored_impostor - input_impostor)}
                rows.append(row)
                for query_type, scores in (("input", input_scores), ("restored", restored_scores)):
                    for anchor_index, score in enumerate(scores):
                        (input_pair_rows if query_type == "input" else pair_rows).append({"method": args.method, "seed": args.seed, "query_type": query_type, "source_id": group["source_id"], "degradation": degradation, "anchor_source_id": groups[anchor_index]["source_id"], "label": int(anchor_index == source_index), "score": float(score)})
                views_for_source.append(restored)
            restored_by_source.append(views_for_source)

    pair_scores = [row["score"] for row in pair_rows]
    pair_labels = [row["label"] for row in pair_rows]
    input_pair_scores = [row["score"] for row in input_pair_rows]
    input_pair_labels = [row["label"] for row in input_pair_rows]
    sibling = [float(F.l1_loss(a, b)) for views in restored_by_source for a, b in combinations(views, 2)]
    means = [F.interpolate(torch.stack(views).mean(dim=0, keepdim=True), size=(256, 256), mode="bilinear", align_corners=False) for views in restored_by_source]
    cross = [float(F.l1_loss(a, b)) for a, b in combinations(means, 2)]
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in ("psnr", "ssim", "input_to_clean_top1", "restored_to_clean_top1", "input_own_anchor_cos", "restored_own_anchor_cos", "input_nearest_impostor_cos", "restored_nearest_impostor_cos", "input_margin", "restored_margin", "identity_gain_margin")}
    aggregate.update({"restored_roc_auc": binary_auc(pair_scores, pair_labels), "restored_eer": eer(pair_scores, pair_labels), "input_roc_auc": binary_auc(input_pair_scores, input_pair_labels), "input_eer": eer(input_pair_scores, input_pair_labels), "roc_auc_gain": binary_auc(pair_scores, pair_labels) - binary_auc(input_pair_scores, input_pair_labels), "eer_gain": eer(input_pair_scores, input_pair_labels) - eer(pair_scores, pair_labels), "sibling_output_l1": float(np.mean(sibling)), "cross_source_output_l1": float(np.mean(cross)), "intra_inter_output_ratio": float(np.mean(sibling) / np.mean(cross))})
    report = {"status": "independent_verifier_evaluation", "split": args.split, "sources": len(groups), "restored_views": len(rows), "verifier_fingerprint": fingerprint, "aggregate": aggregate, "notes": ["Headline identity metrics use an independent frozen verifier.", "Batch evaluation pipeline."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_pairs = input_pair_rows + pair_rows
    with args.output.with_name(args.output.stem + "_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_pairs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
