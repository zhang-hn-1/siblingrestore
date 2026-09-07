from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siblingrestore.data import _read_rgb
from siblingrestore.inference import restore_tiled
from restore_util import restore_image_padded
from train_cross_verifier import CrossVerifier

DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")
METHODS = {
    "Degraded": None,
    "Restormer": ROOT / "runs/campaigns/c005_official_group2/restormer/13/best.pt",
    "DehazeFormer": ROOT / "runs/campaigns/c005_official_group1/dehazeformer/13/best.pt",
    "A5": ROOT / "runs/ablation_core/A5_no_sibling_seed13/best.pt",
}


def binary_auc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, 1): ranks[index] = rank
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives: return float("nan")
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def eer(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives: return float("nan")
    tp = fp = 0; best = 1.0; previous = None
    for index in order:
        score = scores[index]
        if previous is not None and score != previous:
            best = min(best, (fp / negatives + (positives - tp) / positives) / 2)
        previous = score
        if labels[index]: tp += 1
        else: fp += 1
    return min(best, (fp / negatives + (positives - tp) / positives) / 2)


def evaluate_views(verifier, source_ids, clean_images, views, device):
    anchors = torch.cat([verifier.embed(clean_images[sid].to(device)) for sid in source_ids])
    rows, scores, labels = [], [], []
    for sid in source_ids:
        own_index = source_ids.index(sid)
        for degradation in DEGRADATIONS:
            embedding = verifier.embed(views[degradation][sid].to(device))
            cosine = (embedding @ anchors.T).squeeze(0)
            rows.append({"source_id": sid, "degradation": degradation, "own_cosine": float(cosine[own_index]), "top1": int(int(cosine.argmax()) == own_index)})
            scores.extend(float(value) for value in cosine)
            labels.extend([int(index == own_index) for index in range(len(source_ids))])
    return rows, {"roc_auc": binary_auc(scores, labels), "eer": eer(scores, labels)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/identity_evaluation/cross_verifier/cross_verifier_resnet18.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/identity_evaluation/cross_verifier"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    verifier = CrossVerifier(len(checkpoint["source_ids"]))
    verifier.load_state_dict(checkpoint["model"]); verifier.to(device).eval()
    groups = {g["source_id"]: g for g in json.loads((ROOT / "data/plamd_sfr_v1/metadata/source_groups.json").read_text()) if g["split"] == "test"}
    source_ids = sorted(groups)
    clean_images = {sid: _read_rgb(ROOT / "data/plamd_sfr_v1" / groups[sid]["clean_path"]).unsqueeze(0) for sid in source_ids}
    degraded = {d: {sid: _read_rgb(ROOT / "data/plamd_sfr_v1" / groups[sid]["degraded"][d]).unsqueeze(0) for sid in source_ids} for d in DEGRADATIONS}
    restoration_models = {}
    for method, path in METHODS.items():
        if path is None: continue
        checkpoint_model = torch.load(path, map_location="cpu", weights_only=False)
        from train import make_model
        model = make_model(checkpoint_model["config"]); model.load_state_dict(checkpoint_model["model"])
        restoration_models[method] = model.to(device).eval()

    all_rows, summary = [], []
    for method in METHODS:
        views = degraded
        if method in restoration_models:
            views = {d: {} for d in DEGRADATIONS}
            for d in DEGRADATIONS:
                for sid in source_ids:
                    views[d][sid] = restore_image_padded(restoration_models[method], degraded[d][sid].to(device), 512, 32).cpu()
        rows, verification = evaluate_views(verifier, source_ids, clean_images, views, device)
        for row in rows: row["method"] = method
        all_rows.extend(rows)
        summary.append({"method": method, "cosine": sum(r["own_cosine"] for r in rows) / len(rows), "top1": sum(r["top1"] for r in rows) / len(rows), "roc_auc": verification["roc_auc"], "eer": verification["eer"], "num_views": len(rows)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cross_verifier_per_view.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "source_id", "degradation", "own_cosine", "top1"]); writer.writeheader(); writer.writerows(all_rows)
    with (args.output_dir / "cross_verifier_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (args.output_dir / "cross_verifier_results.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "test_sources": len(source_ids), "results": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
