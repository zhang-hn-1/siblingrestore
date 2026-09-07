"""Run FINet (SE-YOLOv5) detection evaluation on an exported SFID method dir.

Usage (from repo root):
  python evaluation/run_sfid_detection.py --method Degraded --variant without_fog --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINET = ROOT / "data/external/FINet_src"
OUT = ROOT / "data/external/sfid_detection"
ART = ROOT / "artifacts/sfid_detection"
WEIGHTS = {
    "without_fog": ROOT / "data/external/finet_runs/runs/m_ep99_without_fog/weights/best.pt",
    "fogged": ROOT / "data/external/finet_runs/runs/m_ep99_fogged/weights/best.pt",
    "se_fogged": ROOT / "data/external/finet_runs/runs/se_m_ep99_fogged/weights/best.pt",
}
METHODS = ("Degraded", "A5", "DehazeFormer", "Restormer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--variant", choices=tuple(WEIGHTS), default="without_fog")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    yaml_dir = ART / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = yaml_dir / f"{args.method}.yaml"
    val_path = OUT / args.method / "images"
    data_yaml.write_text(
        f"val: {val_path}\nnc: 2\nnames: ['insulator', 'broken_piece']\n", encoding="utf-8"
    )
    metrics_dir = ART / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir / f"{args.method}__{args.variant}.log"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [
        sys.executable, str(FINET / "test.py"),
        "--weights", str(WEIGHTS[args.variant]),
        "--data", str(data_yaml),
        "--img-size", "640",
        "--batch-size", "1",
        "--conf-thres", "0.001",
        "--iou-thres", "0.65",
        "--device", str(args.gpu),
        "--verbose",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, cwd=str(FINET), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(json.dumps({"method": args.method, "variant": args.variant, "status": "failed", "exit": result.returncode}))
        raise SystemExit(result.returncode)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = {"method": args.method, "variant": args.variant, "detector": str(WEIGHTS[args.variant])}
    for line in text.splitlines():
        m = re.search(r"all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            summary.update({
                "images": int(m.group(1)), "targets": int(m.group(2)),
                "precision": float(m.group(3)), "recall": float(m.group(4)),
                "mAP50": float(m.group(5)), "mAP50_95": float(m.group(6)),
            })
    (metrics_dir / f"{args.method}__{args.variant}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
