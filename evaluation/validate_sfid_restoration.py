"""Dataset-only SFID restoration validation: fog vs clear twin quality.

No detector, no FINet code. Uses SFID's internal paired fog/clear images:
every fogged_*.jpg in the dataset has its clear original (verified 6859/6859).
Restored outputs were exported beforehand by evaluation/export_sfid_restored.py.

Metrics follow the project convention: masked PSNR/SSIM at native resolution
(clean aligned to the degraded/restored size with Lanczos, PLAMD Method-B
style) and AlexNet LPIPS at the 256x256 protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SFID = ROOT / "data/external/SFID_extracted/SFID"
SFID_DET = ROOT / "data/external/sfid_detection"
DEGRADED_DIR = SFID_DET / "Degraded/images"       # copies of fog jpg
METHOD_DIRS = {
    "Fog (input)": SFID_DET / "Degraded/images",
    "A5": SFID_DET / "A5/images",
    "DehazeFormer": SFID_DET / "DehazeFormer/images",
    "Restormer": SFID_DET / "Restormer/images",
}


def load_rgb(path: Path):
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def lanczos_to(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    array = image.permute(1, 2, 0).mul(255.0).round().clamp_(0, 255).byte().numpy()
    pil = Image.fromarray(array).resize((target_w, target_h), Image.LANCZOS)
    return torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"

    train_clear = {p.stem: p for p in (SFID / "images/train").glob("*.jpg") if not p.name.startswith("fogged_")}
    test_clear = {p.stem: p for p in (SFID / "images/test").glob("*.jpg") if not p.name.startswith("fogged_")}
    fog_files = sorted((SFID / "images/test").glob("fogged_*.jpg"))
    if args.max_images:
        fog_files = fog_files[: args.max_images]
    # locate clear twin (test first, then train)
    pairs = []
    for fog in fog_files:
        base = fog.name[len("fogged_"):-4]
        clear = test_clear.get(base) or train_clear.get(base)
        if clear is not None:
            pairs.append((fog, clear))
    print(json.dumps({"device": device, "paired_images": len(pairs), "of": len(fog_files)}))

    from siblingrestore.metrics import psnr, ssim
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        print(f"lpips unavailable: {exc}")

    rows = []
    for method, method_dir in METHOD_DIRS.items():
        per_image = {name: [] for name in ("psnr", "ssim", "lpips")}
        for fog, clear in pairs:
            target = method_dir / (fog.stem + (".png" if method != "Fog (input)" else ".jpg"))
            if not target.exists():
                continue
            restored = load_rgb(target)
            reference = load_rgb(clear)
            if reference.shape != restored.shape:
                reference = lanczos_to(reference, int(restored.shape[-2]), int(restored.shape[-1]))
            mask = torch.ones((1, reference.shape[-2], reference.shape[-1]), dtype=torch.float32)
            per_image["psnr"].append(float(psnr(restored, reference, mask)))
            per_image["ssim"].append(float(ssim(restored, reference, mask)))
            if lpips_fn is not None:
                with torch.no_grad():
                    a = torch.nn.functional.interpolate(restored.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).to(device) * 2 - 1
                    b = torch.nn.functional.interpolate(reference.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).to(device) * 2 - 1
                    per_image["lpips"].append(float(lpips_fn(a, b)))
        row = {"method": method, "num_images": len(per_image["psnr"])}
        for name in ("psnr", "ssim", "lpips"):
            values = per_image[name]
            row[f"{name}_mean"] = statistics.mean(values) if values else None
            row[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = ROOT / "artifacts/sfid_detection"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sfid_restoration_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "sfid_restoration_quality.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
