"""Measure lightweight metrics (Params, MACs, Latency, Peak VRAM, Throughput)
for all evaluated models at 256x256 input.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/measure_model_metrics.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siblingrestore.baselines import MODELS
from train import make_model

ROOT = Path(__file__).resolve().parents[1]

# model name -> checkpoint path (to build config)
MODEL_CKPT = {
    "ours_dim48_anchor": "runs/campaigns/c005_official_group3/ours_dim48_anchor/13/best.pt",
    "ours_dim64_anchor": "runs/campaigns/c005_official_group1/ours_dim64_anchor/13/best.pt",
}


def load_model(name: str) -> torch.nn.Module:
    if name in MODEL_CKPT:
        ck = torch.load(MODEL_CKPT[name], map_location="cpu", weights_only=False)
        # Ensure config includes refinement_type for backward compat
        model_config = ck["config"].get("model", ck["config"])
        if "refinement_type" not in model_config:
            model_config["refinement_type"] = "none"
        return make_model(ck["config"])
    return MODELS[name]()


def measure(name: str, device: torch.device, reps: int = 30, warmup: int = 5) -> dict:
    model = load_model(name).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # MACs via thop
    from thop import profile
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    # Latency / Throughput / Peak VRAM (batch=1, 256x256)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            model(dummy)
        torch.cuda.synchronize()
        total = time.time() - t0
    latency_ms = total / reps * 1000
    throughput = reps / total
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3

    return {
        "model": name,
        "params_M": round(params, 2),
        "macs_G": round(macs / 1e9, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_vram_GB": round(peak_vram, 2),
        "throughput_ips": round(throughput, 2),
    }


def main() -> None:
    device = torch.device("cuda")
    names = [
        "ours_dim48_anchor", "dehazeformer", "ours_dim64_anchor", "restormer",
        "promptir", "dfpir", "ffanet", "r2r", "uformer", "airnet",
        "swinir", "prenet", "transweather", "clearair", "grl",
    ]
    results = []
    for name in names:
        try:
            r = measure(name, device)
            print(json.dumps(r), flush=True)
            results.append(r)
        except Exception as e:
            print(f"{name}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
    out = ROOT / "results" / "model_metrics.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
