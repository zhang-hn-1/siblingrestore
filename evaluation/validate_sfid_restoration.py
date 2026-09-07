"""Fast parallel SFID restoration-quality validation (dataset only, no detector).

Phase 1: PSNR/SSIM per pair across methods using multiprocessing (CPU decode).
Phase 2: LPIPS (AlexNet @256) streaming on GPU, reusing on-disk images.
Outputs artifacts/sfid_detection/sfid_restoration_quality.{csv,json}
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SFID = ROOT / "data/external/SFID_extracted/SFID"
SFID_DET = ROOT / "data/external/sfid_detection"
METHOD_DIRS = {
    "Fog (input)": SFID_DET / "Degraded/images",
    "A5": SFID_DET / "A5/images",
    "DehazeFormer": SFID_DET / "DehazeFormer/images",
    "Restormer": SFID_DET / "Restormer/images",
}


def _load_rgb(path: Path):
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _lanczos_to(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    array = image.permute(1, 2, 0).mul(255.0).round().clamp_(0, 255).byte().numpy()
    pil = Image.fromarray(array).resize((target_w, target_h), Image.LANCZOS)
    return torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)


def _pair_metrics(args):
    method, target_path, clear_path = args
    from siblingrestore.metrics import psnr, ssim
    restored = _load_rgb(Path(target_path))
    reference = _load_rgb(Path(clear_path))
    if reference.shape != restored.shape:
        reference = _lanczos_to(reference, int(restored.shape[-2]), int(restored.shape[-1]))
    mask = torch.ones((1, reference.shape[-2], reference.shape[-1]), dtype=torch.float32)
    return method, float(psnr(restored, reference, mask)), float(ssim(restored, reference, mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    train_clear = {p.stem: p for p in (SFID / "images/train").glob("*.jpg") if not p.name.startswith("fogged_")}
    test_clear = {p.stem: p for p in (SFID / "images/test").glob("*.jpg") if not p.name.startswith("fogged_")}
    fog_files = sorted((SFID / "images/test").glob("fogged_*.jpg"))
    if args.max_images:
        fog_files = fog_files[: args.max_images]

    tasks = []
    for method, method_dir in METHOD_DIRS.items():
        for fog in fog_files:
            base = fog.name[len("fogged_"):-4]
            clear = test_clear.get(base) or train_clear.get(base)
            if clear is None:
                continue
            target = method_dir / (fog.stem + (".png" if method != "Fog (input)" else ".jpg"))
            if target.exists():
                tasks.append((method, str(target), str(clear)))

    start = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_pair_metrics, tasks, chunksize=8)
    print(json.dumps({"phase1_elapsed_seconds": round(time.time() - start), "pairs": len(results)}))

    by_method = {}
    for method, p, s in results:
        by_method.setdefault(method, {"psnr": [], "ssim": []})
        by_method[method]["psnr"].append(p)
        by_method[method]["ssim"].append(s)

    # Phase 2: LPIPS streaming on GPU.
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    import lpips
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for method in by_method:
        by_method[method]["lpips"] = []
    for method, method_dir in METHOD_DIRS.items():
        for fog in fog_files:
            base = fog.name[len("fogged_"):-4]
            clear = test_clear.get(base) or train_clear.get(base)
            if clear is None:
                continue
            target = method_dir / (fog.stem + (".png" if method != "Fog (input)" else ".jpg"))
            if not target.exists():
                continue
            with torch.no_grad():
                a = _lanczos_to(_load_rgb(target), 256, 256).unsqueeze(0) * 2 - 1
                b = _lanczos_to(_load_rgb(clear), 256, 256).unsqueeze(0) * 2 - 1
                value = float(lpips_model(a.to(device), b.to(device)))
            by_method[method]["lpips"].append(value)
    print(json.dumps({"phase2_elapsed_seconds": round(time.time() - start)}))

    rows = []
    for method, values in by_method.items():
        row = {"method": method, "num_images": len(values["psnr"])}
        for name in ("psnr", "ssim", "lpips"):
            v = values[name]
            row[f"{name}_mean"] = statistics.mean(v) if v else None
            row[f"{name}_std"] = statistics.stdev(v) if len(v) > 1 else None
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = ROOT / "artifacts/sfid_detection"
    with (out / "sfid_restoration_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "sfid_restoration_quality.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
