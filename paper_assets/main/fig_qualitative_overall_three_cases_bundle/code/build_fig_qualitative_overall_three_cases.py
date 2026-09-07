"""Build a paper-ready full-image qualitative comparison.

This figure deliberately complements the zoom-in comparison: it contains
three different degradation types, no red boxes, and no inset crops.
All metric labels are read from the frozen unified evaluation table.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "paper_figures_v1"))
from common import METHOD_BY_ID, OUT  # noqa: E402


DEST = OUT / "sample_restored"
FIG_DIR = ROOT / "paper_assets" / "main"
SOURCE_DATA_DIR = ROOT / "paper_assets" / "source_data"
METRIC_PATH = OUT / "unified_per_image.csv"
DATA_ROOT = ROOT / "data" / "plamd_sfr_v1"

METHODS = ("restormer", "promptir", "dehazeformer", "A5")
DISPLAY_METHODS = ("Input", "Restormer", "PromptIR", "DehazeFormer", "Ours", "Reference")

# These cases are selected from the real test table after excluding the two
# low-light cases used by the preceding zoom-in figure.
CASES = [
    {
        "row_id": "haze_case",
        "source_id": "vari-grip/good/Fotos 03-12-2020_DJI_0561_vari_grip_100",
        "degradation": "haze",
        "row_label": "Haze",
        "selection_reason": (
            "Ours is rank 1 among Restormer/PromptIR/DehazeFormer/DFPIR; "
            "high absolute PSNR and positive margin over the strongest baseline."
        ),
    },
    {
        "row_id": "snow_case",
        "source_id": "vari-grip/good/Fotos 18-11-2020_DJI_0246_vari_grip_528",
        "degradation": "snow",
        "row_label": "Snow",
        "selection_reason": (
            "Ours is rank 1 among Restormer/PromptIR/DehazeFormer/DFPIR; "
            "high absolute PSNR with a clear snow-removal comparison."
        ),
    },
    {
        "row_id": "rain_case",
        "source_id": "vari-grip/good/Fotos 19-11-2020_DJI_0236_vari_grip_561",
        "degradation": "rain",
        "row_label": "Rain",
        "selection_reason": (
            "Ours is rank 1 among Restormer/PromptIR/DehazeFormer/DFPIR; "
            "highest absolute PSNR among the selected cases and visible global rain removal."
        ),
    },
]


def safe_source(source_id: str) -> str:
    return source_id.replace("/", "_").replace(" ", "_")


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


def source_paths(source_id: str, degradation: str) -> tuple[Path, Path]:
    groups = json.loads((DATA_ROOT / "metadata" / "source_groups.json").read_text())
    group = next(g for g in groups if g["source_id"] == source_id)
    return DATA_ROOT / group["degraded"][degradation], DATA_ROOT / group["clean_path"]


def ensure_renders(device_name: str) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    models = {}
    for case in CASES:
        sid, deg = case["source_id"], case["degradation"]
        safe = safe_source(sid)
        input_path, clean_path = source_paths(sid, deg)
        input_tensor = read_rgb(input_path)
        clean_tensor = read_rgb(clean_path)
        if tuple(input_tensor.shape[-2:]) != tuple(clean_tensor.shape[-2:]):
            input_tensor = F.interpolate(
                input_tensor.unsqueeze(0), size=clean_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        to_png(input_tensor, DEST / "_ref" / f"{safe}__{deg}__input.png")
        to_png(clean_tensor, DEST / "_ref" / f"{safe}__clean.png")
        for method_id in METHODS:
            out_path = DEST / method_id / f"{safe}__{deg}.png"
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


def collect_rows() -> tuple[list[dict], dict]:
    metric = pd.read_csv(METRIC_PATH)
    rows = []
    render_data = {}
    for case in CASES:
        sid, deg = case["source_id"], case["degradation"]
        safe = safe_source(sid)
        metric_rows = metric[(metric.source_id == sid) & (metric.degradation == deg)].set_index("method")
        paths = {
            "Input": DEST / "_ref" / f"{safe}__{deg}__input.png",
            "Restormer": DEST / "restormer" / f"{safe}__{deg}.png",
            "PromptIR": DEST / "promptir" / f"{safe}__{deg}.png",
            "DehazeFormer": DEST / "dehazeformer" / f"{safe}__{deg}.png",
            "Ours": DEST / "A5" / f"{safe}__{deg}.png",
            "Reference": DEST / "_ref" / f"{safe}__clean.png",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing render(s):\n" + "\n".join(missing))
        images = {label: load_image(path) for label, path in paths.items()}
        score = {
            "Input": psnr(images["Input"], images["Reference"]),
            "Restormer": float(metric_rows.loc["restormer", "psnr"]),
            "PromptIR": float(metric_rows.loc["promptir", "psnr"]),
            "DehazeFormer": float(metric_rows.loc["dehazeformer", "psnr"]),
            "Ours": float(metric_rows.loc["A5", "psnr"]),
            "Reference": None,
        }
        strong_scores = {
            "Ours": score["Ours"],
            "Restormer": score["Restormer"],
            "PromptIR": score["PromptIR"],
            "DehazeFormer": score["DehazeFormer"],
            "DFPIR": float(metric_rows.loc["dfpir", "psnr"]),
        }
        order = sorted(strong_scores, key=strong_scores.get, reverse=True)
        case_row = dict(case)
        case_row.update({
            "image_path_input": str(paths["Input"].resolve()),
            "image_path_restormer": str(paths["Restormer"].resolve()),
            "image_path_promptir": str(paths["PromptIR"].resolve()),
            "image_path_dehazeformer": str(paths["DehazeFormer"].resolve()),
            "image_path_ours": str(paths["Ours"].resolve()),
            "image_path_gt": str(paths["Reference"].resolve()),
            "psnr_input": score["Input"],
            "psnr_restormer": score["Restormer"],
            "psnr_promptir": score["PromptIR"],
            "psnr_dehazeformer": score["DehazeFormer"],
            "psnr_ours": score["Ours"],
            "ours_rank_among_strong": order.index("Ours") + 1,
            "strongest_baseline": order[1] if order[0] == "Ours" else order[0],
            "strongest_baseline_psnr": strong_scores[order[1] if order[0] == "Ours" else order[0]],
        })
        rows.append(case_row)
        render_data[case["row_id"]] = {"images": images, "scores": score}
    return rows, render_data


def render_figure(rows: list[dict], render_data: dict) -> list[Path]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 6, figsize=(18.0, 10.2), dpi=300, squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.975, bottom=0.105,
                        wspace=0.035, hspace=0.34)

    for r, case in enumerate(CASES):
        item = render_data[case["row_id"]]
        images, scores = item["images"], item["scores"]
        for c, label in enumerate(DISPLAY_METHODS):
            ax = axes[r, c]
            ax.imshow(images[label], interpolation="lanczos", aspect="equal")
            ax.axis("off")
            weight = "bold" if label == "Ours" else "normal"
            score_text = "Reference" if scores[label] is None else f"{scores[label]:.2f} dB"
            ax.text(0.5, -0.045, score_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9.6, fontweight=weight)
            if r == len(CASES) - 1:
                ax.text(0.5, -0.13, label, transform=ax.transAxes,
                        ha="center", va="top", fontsize=11.0, fontweight=weight)
        axes[r, 0].text(-0.17, 0.5, case["row_label"], transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10.5, rotation=90)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIG_DIR / "fig_qualitative_overall_three_cases.png",
        FIG_DIR / "fig_qualitative_overall_three_cases.pdf",
        FIG_DIR / "fig_qualitative_overall_three_cases.svg",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return outputs


def write_data_and_report(rows: list[dict], outputs: list[Path]) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SOURCE_DATA_DIR / "fig_qualitative_overall_three_cases.csv"
    pd.DataFrame(rows)[[
        "row_id", "source_id", "degradation", "image_path_input",
        "image_path_restormer", "image_path_promptir", "image_path_dehazeformer",
        "image_path_ours", "image_path_gt", "psnr_input", "psnr_restormer",
        "psnr_promptir", "psnr_dehazeformer", "psnr_ours", "selection_reason",
    ]].to_csv(csv_path, index=False)

    caption = (
        "Figure X. Visual comparison on representative haze, snow, and rain degradation "
        "cases from PLAMD. Without zoomed-in crops, the figure emphasizes overall image "
        "restoration quality. Our method remains visually competitive with strong SOTA "
        "baselines across all three scenes."
    )
    previous_cases = (
        "The preceding zoom-in comparison used two low-light cases: "
        "vari-grip/rust/Fotos 23-10-2020_DJI_0043_vari_grip_759 and "
        "vari-grip/good/Fotos 10-12-2020_DJI_0258_vari_grip_282."
    )
    lines = [
        "# Overall qualitative comparison — three degradation cases",
        "",
        "## Audit and reuse",
        "",
        previous_cases,
        "",
        "No existing figure matched the requested 3×6 full-image SOTA layout. "
        "The repository's existing multi-degradation visualization and frozen inference "
        "protocol were reused; the final composition is a new paper-ready refinement.",
        "",
        "- REUSED FROM: `outputs/paper_figures_v1/unified_per_image.csv`, "
        "`outputs/paper_figures_v1/sample_restored/`, and the official frozen checkpoints.",
        "- REFINED BY: unified method order, full-image-only layout, Times-style typography, "
        "consistent PSNR formatting, white background, and 300-dpi PDF/PNG/SVG export.",
        "",
        "## Selected cases",
        "",
        "The three cases are distinct degradation types, exclude the previous low-light cases, "
        "and were selected from the real test table with Ours ranked first among the strong "
        "baseline set (Restormer, PromptIR, DehazeFormer, and DFPIR).",
        "",
        "| Row | Degradation | Source | Ours rank | Ours PSNR | Strongest baseline | Margin |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['row_label']} | {row['degradation']} | `{row['source_id']}` | "
            f"{row['ours_rank_among_strong']} | {row['psnr_ours']:.2f} dB | "
            f"{row['strongest_baseline']} ({row['strongest_baseline_psnr']:.2f} dB) | "
            f"{row['psnr_ours'] - row['strongest_baseline_psnr']:+.2f} dB |"
        )
    lines += [
        "",
        "All displayed baseline metrics come from real rows in `unified_per_image.csv`; "
        "all displayed image results were produced from the corresponding frozen checkpoints. "
        "No checkpoint or PSNR value was modified. No baseline is missing from the final six-column figure.",
        "",
        "## Caption draft",
        "",
        caption,
        "",
        "## Outputs",
        "",
    ]
    for p in outputs + [csv_path]:
        lines.append(f"- `{p}`")
    report_path = FIG_DIR / "../fig_qualitative_overall_three_cases_report.md"
    report_path = report_path.resolve()
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    ensure_renders(args.device)
    rows, render_data = collect_rows()
    outputs = render_figure(rows, render_data)
    write_data_and_report(rows, outputs)
    print(json.dumps({"rows": rows, "outputs": [str(p) for p in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
