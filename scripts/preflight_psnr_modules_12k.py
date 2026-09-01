#!/usr/bin/env python3
"""Preflight forward/backward and efficiency checks for c007 candidates."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_ROOT = ROOT / "configs" / "psnr_modules_12k"
OUT_ROOT = ROOT / "results" / "psnr_modules_12k"
MODELS = [
    "M0_deg_fixed_baseline", "M1_transformer_refine", "M2_naf_refine",
    "M3_alcrb", "M4_rerh", "M5_lmrb", "M6_hfrb",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for preflight")
    device = torch.device("cuda")
    os.environ.setdefault("PYTHONHASHSEED", "13")
    from train import make_model
    from thop import profile

    rows = []
    for name in MODELS:
        config = json.loads((CONFIG_ROOT / name / "seed13.json").read_text(encoding="utf-8"))
        torch.manual_seed(13)
        model = make_model(config).to(device)
        model.train()
        params = sum(p.numel() for p in model.parameters())
        dummy = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = model(dummy)
        forward_seconds = time.perf_counter() - start
        restored = output["restored"] if isinstance(output, dict) else output
        forward_finite = bool(torch.isfinite(restored).all())
        loss = restored.mean()
        loss.backward()
        torch.cuda.synchronize()
        backward_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        with torch.no_grad():
            macs, _ = profile(model.eval(), inputs=(torch.rand(1, 3, 256, 256, device=device),), verbose=False)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.append({
            "model": name,
            "refinement_type": config["model"].get("refinement_type", "none"),
            "parameters": params,
            "parameters_M": params / 1e6,
            "macs": int(macs),
            "macs_G": macs / 1e9,
            "output_shape": list(restored.shape),
            "forward_finite": forward_finite,
            "backward_finite": backward_finite,
            "forward_seconds": forward_seconds,
            "peak_vram_GB": peak_vram_gb,
            "device": torch.cuda.get_device_name(device),
            "config_path": str(CONFIG_ROOT / name / "seed13.json"),
        })
        del model, dummy, output, restored, loss
        torch.cuda.empty_cache()
        print(json.dumps(rows[-1]), flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "preflight.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PSNR Modules 12k Preflight", "", "| Model | Params (M) | MACs (G) | Output | Forward | Backward | Peak VRAM (GB) |", "|---|---:|---:|---|---|---|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['parameters_M']:.3f} | {row['macs_G']:.3f} | `{row['output_shape']}` | {row['forward_finite']} | {row['backward_finite']} | {row['peak_vram_GB']:.3f} |")
    (OUT_ROOT / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
