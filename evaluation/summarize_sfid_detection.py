"""Summarize SFID downstream detection mAP matrix and write artifacts."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/sfid_detection"

DETECTORS = {
    "without_fog": "YOLOv5m (clear-trained)",
    "fogged": "YOLOv5m (fog-trained)",
    "se_fogged": "SE-YOLOv5m (fog-trained)",
}


def main():
    metrics = []
    for log in sorted((ART / "metrics").glob("*__*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
        if not match:
            continue
        method, variant = log.stem.split("__")
        row = {
            "method": method,
            "detector_variant": variant,
            "detector": DETECTORS.get(variant, variant),
            "images": int(match.group(1)),
            "targets": int(match.group(2)),
            "precision": float(match.group(3)),
            "recall": float(match.group(4)),
            "mAP50": float(match.group(5)),
            "mAP50_95": float(match.group(6)),
        }
        # per-class lines
        for line in text.splitlines():
            cm = re.match(r"\s*(insulator|broken_piece)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if cm:
                row[f"mAP50_{cm.group(1)}"] = float(cm.group(6))
                row[f"mAP50_95_{cm.group(1)}"] = float(cm.group(7))
        metrics.append(row)

    metrics.sort(key=lambda r: (r["method"], r["detector_variant"]))
    with (ART / "sfid_detection_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    (ART / "sfid_detection_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Table for report (mAP50 per method x detector).
    methods = ["Clear", "Degraded", "A5", "DehazeFormer", "Restormer"]
    variants = ["without_fog", "fogged", "se_fogged"]
    lines = ["# SFID Downstream Detection Benchmark", "",
             "Frozen FINet detectors (SE-YOLOv5) evaluated on SFID foggy test images (1404 fog views).",
             "Clear = non-fog test images (1339) provided as in-domain reference.",
             "Restoration models are frozen; detector weights frozen; no retraining.",
             "", "## mAP@0.5", "", "| Method | " + " | ".join(DETECTORS[v] for v in variants) + " |", "|---|---" + "---|" * len(variants)]
    def value(method, variant, key="mAP50"):
        row = next((r for r in metrics if r["method"] == method and r["detector_variant"] == variant), None)
        return f"{row[key]:.3f}" if row else "NA"
    for method in methods:
        lines.append(f"| {method} | " + " | ".join(value(method, v) for v in variants) + " |")
    lines += ["", "## mAP@0.5:0.95", "", "| Method | " + " | ".join(DETECTORS[v] for v in variants) + " |", "|---|---" + "---|" * len(variants)]
    for method in methods:
        lines.append(f"| {method} | " + " | ".join(value(method, v, "mAP50_95") for v in variants) + " |")
    (ART / "SFID_DETECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: mAP50 grouped bars.
    x = list(range(len(methods)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, variant in enumerate(variants):
        vals = [float(value(m, variant)) if value(m, variant) != "NA" else 0 for m in methods]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=DETECTORS[variant])
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("mAP@0.5"); ax.set_ylim(0.85, 1.0); ax.legend()
    ax.set_title("SFID foggy-test detection mAP after restoration (frozen detectors)")
    fig.tight_layout()
    fig.savefig(ART / "sfid_detection_mAP50.png", dpi=300)
    fig.savefig(ART / "sfid_detection_mAP50.pdf")
    plt.close(fig)
    print(json.dumps({"metrics": len(metrics), "outputs": ["sfid_detection_metrics.csv", "SFID_DETECTION_REPORT.md", "sfid_detection_mAP50.png/pdf"]}, indent=2))


if __name__ == "__main__":
    main()
