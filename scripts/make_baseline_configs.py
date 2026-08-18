from __future__ import annotations

"""Generate baseline training configs (Restormer / SwinIR / PromptIR) on the
pilot data using the same protocol as the independent control: batch 8,
crop 256, 5000 steps, AdamW 2e-4 cosine, Charbonnier + gradient loss.

Run: python scripts/make_baseline_configs.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "baselines"
SEEDS = (13,)

BASE = {
    "data_root": "data/plamd_vari_grip_pilot",
    "mode": "independent",
    "optimizer": "adamw",
    "scheduler": "cosine",
    "device": "cuda",
    "crop_size": 256,
    "batch_size": 8,
    "num_workers": 8,
    "epochs": 700,
    "max_steps": 5000,
    "validation_interval_steps": 500,
    "validation_split": "val",
    "gradient_diagnostic_interval_steps": 200,
    "eval_tile_size": 512,
    "eval_tile_overlap": 32,
    "learning_rate": 0.0002,
    "min_learning_rate": 0.000001,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "amp": True,
    "loss_weights": {
        "gradient": 0.05,
        "source": 0.0,
        "output": 0.0,
        "degradation": 0.0,
        "temperature": 0.1,
        "same_class_negative_weight": 2.0,
    },
}

VARIANTS = {
    "restormer": {"model_family": "restormer", "batch_size": 8},
    "swinir": {"model_family": "swinir", "batch_size": 8},
    "promptir": {"model_family": "promptir", "batch_size": 4},
    "ffanet": {"model_family": "ffanet", "batch_size": 8},
    "prenet": {"model_family": "prenet", "batch_size": 8},
    "uformer": {"model_family": "uformer", "batch_size": 4},
    "dehazeformer": {"model_family": "dehazeformer", "batch_size": 4},
}


def main() -> None:
    written = 0
    for name, overrides in VARIANTS.items():
        for seed in SEEDS:
            config = json.loads(json.dumps(BASE))
            config.update(overrides)
            config["seed"] = seed
            config["output_dir"] = f"runs/baselines/{name}/seed{seed}"
            path = CONFIG_ROOT / name / f"seed{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written += 1
    print(f"wrote {written} baseline configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
