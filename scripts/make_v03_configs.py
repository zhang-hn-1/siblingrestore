from __future__ import annotations

"""Generate the v0.3 degradation-line configs (A1/A2/A3/A4 x seeds 13/37/73).

Run: python scripts/make_v03_configs.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "v03_degradation"
SEEDS = (13, 37, 73)

BASE = {
    "data_root": "data/plamd_vari_grip_pilot",
    "mode": "sibling",
    "optimizer": "adamw",
    "scheduler": "cosine",
    "device": "cuda",
    "crop_size": 256,
    "sibling_count": 2,
    "batch_size": 4,
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
    "model": {"dim": 32, "blocks_per_level": [2, 2, 3, 2, 2], "heads": [1, 2, 4], "projection_dim": 128},
}

# name -> (loss_weights overrides, model overrides)
VARIANTS = {
    "degradation_003": ({"degradation": 0.03}, {}),
    "degradation_005": ({"degradation": 0.05}, {}),
    "degradation_001_output_001": ({"degradation": 0.01, "output": 0.01}, {}),
    "degradation_cond_001": ({"degradation": 0.01}, {"degradation_conditioned": True}),
}


def main() -> None:
    written = 0
    for name, (weight_overrides, model_overrides) in VARIANTS.items():
        weights = {
            "gradient": 0.05,
            "source": 0.0,
            "output": 0.0,
            "degradation": 0.01,
            "temperature": 0.1,
            "same_class_negative_weight": 2.0,
        }
        weights.update(weight_overrides)
        for seed in SEEDS:
            model = dict(BASE["model"])
            model.update(model_overrides)
            config = dict(BASE)
            config["seed"] = seed
            config["output_dir"] = f"runs/v03_degradation/{name}/seed{seed}"
            config["model"] = model
            config["loss_weights"] = weights
            path = CONFIG_ROOT / name / f"seed{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written += 1
    print(f"wrote {written} configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
