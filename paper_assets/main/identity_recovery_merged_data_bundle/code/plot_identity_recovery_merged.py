"""Render the merged per-degradation identity-recovery figure.

This is a minimal composition change: the source CSV is read unchanged,
Degraded is moved into the same grouped-bar axis as the restored methods,
and A5 is displayed as Ours for the paper-facing label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/identity_evaluation/identity_cosine/identity_cosine_per_degradation.csv"
OUT_DIR = ROOT / "artifacts/identity_evaluation/paper_figures"
OUT_STEM = OUT_DIR / "identity_recovery_merged"

DEGRADATIONS = ["blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow"]
DISPLAY_DEGRADATIONS = ["Blur", "Haze", "Inpainting", "Low-light", "Noise", "Rain", "Snow"]


def load_values() -> dict[str, list[float]]:
    df = pd.read_csv(SOURCE)
    lookup = df.set_index(["method", "degradation"])["restored_cosine"]
    degraded = df[df.method == "A5"].set_index("degradation")["degraded_cosine"]
    return {
        "Degraded": [float(degraded.loc[d]) for d in DEGRADATIONS],
        "Restormer": [float(lookup.loc[("Restormer", d)]) for d in DEGRADATIONS],
        "DehazeFormer": [float(lookup.loc[("DehazeFormer", d)]) for d in DEGRADATIONS],
        "Ours": [float(lookup.loc[("A5", d)]) for d in DEGRADATIONS],
    }


def render() -> list[Path]:
    values = load_values()
    methods = ["Degraded", "Restormer", "DehazeFormer", "Ours"]
    colors = {
        "Degraded": "#8c8c8c",
        "Restormer": "#ff7f0e",
        "DehazeFormer": "#2ca02c",
        "Ours": "#1f77b4",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=300)
    y_min, y_max = 0.55, 1.01
    x = np.arange(len(DEGRADATIONS), dtype=float)
    width = 0.18
    offsets = (np.arange(len(methods)) - 1.5) * width

    for i, method in enumerate(methods):
        vals = np.asarray(values[method], dtype=float)
        bars = ax.bar(
            x + offsets[i], vals - y_min, width=width,
            bottom=y_min, color=colors[method],
            edgecolor="#333333" if method == "Ours" else "none",
            linewidth=1.0 if method == "Ours" else 0.0,
            label=method, zorder=3,
        )
        if method == "Ours":
            for bar in bars:
                bar.set_hatch(None)

    ax.set_ylabel("Identity cosine to clean source anchor")
    ax.set_xticks(x, DISPLAY_DEGRADATIONS)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.58, len(DEGRADATIONS) - 0.42)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=3, color="#444444")
    leg = fig.legend(
        handles=[ax.containers[i] for i in range(len(methods))],
        labels=methods, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.955), frameon=False,
        handlelength=1.2, columnspacing=1.1,
    )
    for text in leg.get_texts():
        if text.get_text() == "Ours":
            text.set_fontweight("bold")

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.84)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [OUT_STEM.with_suffix(".png"), OUT_STEM.with_suffix(".pdf"), OUT_STEM.with_suffix(".svg")]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for output in render():
        print(output)
