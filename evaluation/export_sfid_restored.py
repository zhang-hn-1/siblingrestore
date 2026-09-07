"""Export SFID foggy test images restored by a frozen restoration method.

Usage (run from repo root):
  python evaluation/export_sfid_restored.py --method A5 --gpu 0
  python evaluation/export_sfid_restored.py --method Degraded   # copy only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SFID = ROOT / "data/external/SFID_extracted/SFID"
OUT = ROOT / "data/external/sfid_detection"
CHECKPOINTS = {
    "A5": ROOT / "runs/ablation_core/A5_no_sibling_seed13/best.pt",
    "DehazeFormer": ROOT / "runs/campaigns/c005_official_group1/dehazeformer/13/best.pt",
    "Restormer": ROOT / "runs/campaigns/c005_official_group2/restormer/13/best.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["Degraded", "A5", "DehazeFormer", "Restormer"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    source_images = SFID / "images/test"
    source_labels = SFID / "labels/test"
    fogged = sorted(p.name for p in source_images.glob("fogged_*.jpg"))
    print(json.dumps({"method": args.method, "fog_images": len(fogged)}))

    target_root = OUT / args.method
    target_images = target_root / "images"
    target_labels = target_root / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    # Ground-truth labels are identical for every method (copy once).
    for name in fogged:
        label = source_labels / (name[:-4] + ".txt")
        if label.exists():
            shutil.copy(label, target_labels / label.name)

    if args.method == "Degraded":
        for name in fogged:
            dst = target_images / name
            if not dst.exists():
                shutil.copy(source_images / name, dst)
        print(json.dumps({"method": args.method, "status": "copied", "images": len(fogged)}))
        return

    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = torch.load(CHECKPOINTS[args.method], map_location="cpu", weights_only=False)
    from train import make_model
    model = make_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    from evaluation.restore_util import restore_image_padded
    from PIL import Image
    import numpy as np
    from siblingrestore.data import _read_rgb

    for i, name in enumerate(fogged):
        out_path = target_images / (name[:-4] + ".png")
        if out_path.exists():
            continue
        degraded = _read_rgb(source_images / name).unsqueeze(0).to(device)
        with torch.no_grad():
            restored = restore_image_padded(model, degraded, 512, 32)
        array = restored[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.asarray(array)).save(out_path)
        if (i + 1) % 100 == 0:
            print(json.dumps({"method": args.method, "progress": i + 1, "of": len(fogged)}), flush=True)
    print(json.dumps({"method": args.method, "status": "done", "images": len(fogged)}))


if __name__ == "__main__":
    main()
