"""Build the AdaIR-style qualitative comparison for the selected best-PSNR case.

The script only runs inference for the two missing official baseline renders
(Restormer and PromptIR). Existing A5/Ours and DehazeFormer renders are reused.
All displayed PSNR values come from the frozen unified per-image table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "paper_figures_v1"))
from common import METHOD_BY_ID, OUT  # noqa: E402


DEST = OUT / "sample_restored"
FIG_DIR = ROOT / "paper_assets" / "main"
METRIC_PATH = OUT / "unified_per_image.csv"
DATA_ROOT = ROOT / "data" / "plamd_sfr_v1"

# Two complementary low-light cases from the existing visualization plan.
# The ROI is shared across all seven columns within each row.
CASES = [
    {
        "source_id": "vari-grip/rust/Fotos 23-10-2020_DJI_0043_vari_grip_759",
        "degradation": "lowlight",
        "row_label": "Rust / low-light",
        "roi": (158, 186, 330, 368),
    },
    {
        "source_id": "vari-grip/good/Fotos 10-12-2020_DJI_0258_vari_grip_282",
        "degradation": "lowlight",
        "row_label": "Good / low-light",
        "roi": (300, 430, 650, 780),
    },
]


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def to_png(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def restore_one(model, degraded_path: Path, clean_path: Path, device: torch.device):
    from siblingrestore.data import align_pair_shape, paired_eval_pad
    from siblingrestore.inference import restore_tiled

    with torch.no_grad():
        degraded_orig = read_rgb(degraded_path)
        clean_orig = read_rgb(clean_path)
        clean_h, clean_w = int(clean_orig.shape[-2]), int(clean_orig.shape[-1])
        degraded, clean = align_pair_shape(degraded_orig, clean_orig)
        degraded_pad, clean_pad, _ = paired_eval_pad(degraded, clean)
        del clean_pad
        # Match the repository's official inference call signature.
        restored_pad = restore_tiled(model, degraded_pad.unsqueeze(0).to(device), 512, 32)
        restored = restored_pad[..., : degraded.shape[-2], : degraded.shape[-1]].squeeze(0).cpu()
        if tuple(restored.shape[-2:]) != (clean_h, clean_w):
            restored = F.interpolate(
                restored.unsqueeze(0), size=(clean_h, clean_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return restored


def load_model(method_id: str, device: torch.device):
    from train import make_model

    meta = METHOD_BY_ID[method_id]
    checkpoint = torch.load(ROOT / meta["ckpt"], map_location="cpu", weights_only=False)
    model = make_model(checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def input_and_clean_paths(source_id: str, degradation: str) -> tuple[Path, Path]:
    groups = json.loads((DATA_ROOT / "metadata" / "source_groups.json").read_text())
    group = next(g for g in groups if g["source_id"] == source_id)
    input_path = DATA_ROOT / group["degraded"][degradation]
    clean_path = DATA_ROOT / group["clean_path"]
    return input_path, clean_path


def ensure_baseline_renders(device_name: str) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    models = {}
    for case in CASES:
        source_id = case["source_id"]
        degradation = case["degradation"]
        safe_source = source_id.replace("/", "_").replace(" ", "_")
        input_path, clean_path = input_and_clean_paths(source_id, degradation)
        # Cache the exact input/GT display frames used by the comparison.
        input_tensor = read_rgb(input_path)
        clean_tensor = read_rgb(clean_path)
        if tuple(input_tensor.shape[-2:]) != tuple(clean_tensor.shape[-2:]):
            input_tensor = F.interpolate(
                input_tensor.unsqueeze(0), size=clean_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        to_png(input_tensor, DEST / "_ref" / f"{safe_source}__{degradation}__input.png")
        to_png(clean_tensor, DEST / "_ref" / f"{safe_source}__clean.png")
        for method_id in ("A5", "dehazeformer", "restormer", "promptir", "dfpir"):
            out_path = DEST / method_id / f"{safe_source}__{degradation}.png"
            if out_path.exists():
                continue
            if method_id not in models:
                print(f"loading {method_id} on {device} ...", flush=True)
                models[method_id] = load_model(method_id, device)
            restored = restore_one(models[method_id], input_path, clean_path, device)
            to_png(restored, out_path)
            print(f"wrote {out_path}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def psnr(arr: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((arr.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def render_figure() -> dict:
    metric = pd.read_csv(METRIC_PATH)
    rendered_cases = []
    for case in CASES:
        source_id = case["source_id"]
        degradation = case["degradation"]
        safe_source = source_id.replace("/", "_").replace(" ", "_")
        row = metric[
            (metric["source_id"] == source_id)
            & (metric["degradation"] == degradation)
        ].set_index("method")
        paths = {
            "Input": DEST / "_ref" / f"{safe_source}__{degradation}__input.png",
            "Restormer": DEST / "restormer" / f"{safe_source}__{degradation}.png",
            "PromptIR": DEST / "promptir" / f"{safe_source}__{degradation}.png",
            "DehazeFormer": DEST / "dehazeformer" / f"{safe_source}__{degradation}.png",
            "DFPIR": DEST / "dfpir" / f"{safe_source}__{degradation}.png",
            "Ours": DEST / "A5" / f"{safe_source}__{degradation}.png",
            "Reference": DEST / "_ref" / f"{safe_source}__clean.png",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing render(s):\n" + "\n".join(missing))
        images = {label: load_image(path) for label, path in paths.items()}
        clean = images["Reference"]
        scores = {
            "Input": psnr(images["Input"], clean),
            "Restormer": float(row.loc["restormer", "psnr"]),
            "PromptIR": float(row.loc["promptir", "psnr"]),
            "DehazeFormer": float(row.loc["dehazeformer", "psnr"]),
            "DFPIR": float(row.loc["dfpir", "psnr"]),
            "Ours": float(row.loc["A5", "psnr"]),
            "Reference": None,
        }
        rendered_cases.append({"case": case, "images": images, "scores": scores})

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 7, figsize=(21.0, 8.35), dpi=300,
                             squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.988, top=0.985, bottom=0.09,
                        hspace=0.34, wspace=0.035)

    for r, rendered in enumerate(rendered_cases):
        case = rendered["case"]
        images = rendered["images"]
        scores = rendered["scores"]
        x0, y0, x1, y1 = case["roi"]
        for ax, (label, arr) in zip(axes[r], images.items()):
            ax.imshow(arr, interpolation="lanczos")
            ax.set_aspect("equal")
            ax.set_xlim(0, arr.shape[1])
            ax.set_ylim(arr.shape[0], 0)
            ax.axis("off")
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor="#e31a1c", linewidth=1.25,
                transform=ax.transData, zorder=5,
            ))
            inset = ax.inset_axes([0.585, 0.045, 0.375, 0.375])
            inset.imshow(arr[y0:y1, x0:x1], interpolation="lanczos")
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("#e31a1c")
                spine.set_linewidth(1.15)

            weight = "bold" if label == "Ours" else "normal"
            # Method labels are shown once on the first row, as in the reference.
            if r == 0:
                ax.text(0.5, -0.055, label, transform=ax.transAxes,
                        ha="center", va="top", fontsize=11.5, fontweight=weight)
            score_text = "Reference" if scores[label] is None else f"{scores[label]:.2f} dB"
            ax.text(0.5, -0.055 if r == 1 else -0.135, score_text,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10.0, fontweight=weight)
        axes[r, 0].text(-0.18, 0.5, case["row_label"], transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10.0, rotation=90)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / "fig3_visual_comparison.png"
    pdf_path = FIG_DIR / "fig3_visual_comparison.pdf"
    svg_path = FIG_DIR / "fig3_visual_comparison.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    metadata = {
        "selection_rule": "first row is the previously selected highest-A5-PSNR case; second row is an additional pre-selected fig2 source",
        "same_input_and_gt_within_each_row": True,
        "psnr_source": str(METRIC_PATH),
        "cases": [
            {
                "source_id": item["case"]["source_id"],
                "degradation": item["case"]["degradation"],
                "row_label": item["case"]["row_label"],
                "roi_xyxy": list(item["case"]["roi"]),
                "input_psnr_display_db": item["scores"]["Input"],
                "method_psnr_db": item["scores"],
            }
            for item in rendered_cases
        ],
        "outputs": [str(png_path), str(pdf_path), str(svg_path)],
    }
    (FIG_DIR / "fig3_visual_comparison_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    ensure_baseline_renders(args.device)
    metadata = render_figure()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
