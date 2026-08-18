from __future__ import annotations

"""Generate v0.5 configs: teacher/evaluator verifiers and the two frozen-anchor
restoration variants across seeds 13/37/73.

Verifier configs are written immediately. Restoration configs require the
teacher checkpoint fingerprint, so their sha256 is injected later by
scripts/run_v05_all.sh once the teacher is trained and audited.

Run: python scripts/make_v05_configs.py [--teacher-sha <hex>]
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "v05_frozen_verifier"
SEEDS = (13, 37, 73)

VERIFIER_BASE = {
    "data_root": "data/plamd_vari_grip_pilot",
    "device": "cuda",
    "selection_fraction": 0.2,
    "selection_steps": 800,
    "final_steps": 1200,
    "crop_size": 256,
    "batch_size": 32,
    "num_workers": 8,
    "learning_rate": 0.0002,
    "model": {"source_count": 71, "embedding_dim": 128},
}

RESTORATION_BASE = {
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
    "loss_weights": {
        "gradient": 0.05,
        "source": 0.0,
        "output": 0.01,
        "degradation": 0.01,
        "temperature": 0.1,
        "same_class_negative_weight": 2.0,
        "anchor": 0.01,
    },
}

VERIFIER_ROLES = {
    "teacher": {"role": "teacher", "seed": 101, "selection_seed": 101011},
    "evaluator": {"role": "evaluator", "seed": 202, "selection_seed": 202022},
}

RESTORATION_VARIANTS = {
    "frozen_anchor_001": {"pcgrad": False},
    "frozen_anchor_pcgrad_001": {"pcgrad": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-sha", type=str, default="")
    args = parser.parse_args()
    written = 0
    for role, overrides in VERIFIER_ROLES.items():
        config = dict(VERIFIER_BASE)
        config.update(overrides)
        config["output_dir"] = f"runs/v05_frozen_verifier/{role}"
        path = CONFIG_ROOT / f"verifier_{role}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
    for name, variant in RESTORATION_VARIANTS.items():
        for seed in SEEDS:
            config = json.loads(json.dumps(RESTORATION_BASE))
            config["seed"] = seed
            config["output_dir"] = f"runs/v05_frozen_verifier/{name}/seed{seed}"
            config["frozen_verifier"] = {
                "checkpoint": "runs/v05_frozen_verifier/teacher/best.pt",
                "sha256": args.teacher_sha,
                "pcgrad": variant["pcgrad"],
            }
            path = CONFIG_ROOT / name / f"seed{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written += 1
    print(f"wrote {written} configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
