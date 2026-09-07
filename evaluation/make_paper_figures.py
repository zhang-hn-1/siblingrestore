from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEGS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")
METHODS = ("A5", "Restormer", "DehazeFormer")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, outdir: Path, name: str):
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = ROOT / "artifacts/identity_evaluation/paper_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    gain_rows = read_csv(ROOT / "artifacts/identity_evaluation/identity_cosine/identity_recovery_gain.csv")
    bootstrap_rows = read_csv(ROOT / "artifacts/identity_evaluation/bootstrap/bootstrap_significance_table.csv")

    # Figure A: degraded vs restored cosine per degradation (A5 highlighted).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    import numpy as np
    width = 0.25
    x = np.arange(len(DEGS))
    for offset, method in enumerate(METHODS):
        degraded = [float(r["degraded_cosine"]) for r in gain_rows if r["method"] == method]
        restored = [float(r["restored_cosine"]) for r in gain_rows if r["method"] == method]
        axes[0].bar(x + offset * width, degraded, width, label=method)
        axes[1].bar(x + offset * width, restored, width, label=method)
    for ax in axes:
        ax.set_xticks(x + width); ax.set_xticklabels(DEGS, rotation=30)
        ax.set_ylim(0.5, 1.0); ax.legend()
    axes[0].set_ylabel("Degraded cosine"); axes[1].set_ylabel("Restored cosine")
    fig.suptitle("Identity cosine to clean source anchor")
    save(fig, outdir, "identity_cosine_per_degradation")

    # Figure B: identity recovery gain.
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, method in enumerate(METHODS):
        gains = [float(r["gain"]) for r in gain_rows if r["method"] == method]
        ax.bar(x + offset * width, gains, width, label=method)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x + width); ax.set_xticklabels(DEGS, rotation=30)
    ax.set_ylabel("Restored cosine gain"); ax.legend(); fig.suptitle("Identity recovery gain across degradations")
    save(fig, outdir, "identity_recovery_gain")

    # Figure D: bootstrap CI for headline metrics of both comparisons.
    comparisons = ("A5_vs_Restormer", "A5_vs_DehazeFormer")
    metrics = ("psnr", "ssim", "lpips", "restored_to_clean_top1", "restored_margin")
    labels = [f"{comparison.split('_vs_')[1]}\n{metric}" for comparison in comparisons for metric in metrics]
    means = [float(r["mean"]) for r in bootstrap_rows if r["comparison"] in comparisons and r["metric"] in metrics]
    lows = [float(r["lower95"]) for r in bootstrap_rows if r["comparison"] in comparisons and r["metric"] in metrics]
    highs = [float(r["upper95"]) for r in bootstrap_rows if r["comparison"] in comparisons and r["metric"] in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    xerr = np.stack([[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]])
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Mean paired source difference (A5 - baseline)"); fig.suptitle("Source-level bootstrap 95% CI")
    save(fig, outdir, "bootstrap_confidence_intervals")

    print(json_dumps({"figures": ["identity_cosine_per_degradation", "identity_recovery_gain", "bootstrap_confidence_intervals"], "output": str(outdir)}))


def json_dumps(value):
    import json
    return json.dumps(value, indent=2)


if __name__ == "__main__":
    main()
