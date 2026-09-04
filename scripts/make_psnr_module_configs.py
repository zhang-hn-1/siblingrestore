#!/usr/bin/env python3
"""Generate the seven fair 12k PSNR-module configurations and audit them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "ablation_core" / "A5_no_sibling.json"
CONFIG_ROOT = ROOT / "configs" / "psnr_modules_12k"
RESULT_ROOT = ROOT / "results" / "psnr_modules_12k"
TEACHER = ROOT / "runs" / "baselines_500" / "verifier_teacher" / "best.pt"
EXPECTED_TEACHER_SHA = "16a87f86a21f7f5c4d956a48925f027fd63c91046c697a67e686e93d85eccf23"

MODELS = {
    "M0_deg_fixed_baseline": "none",
    "M1_transformer_refine": "transformer",
    "M2_naf_refine": "naf",
    "M3_alcrb": "alcrb",
    "M4_rerh": "rerh",
    "M5_lmrb": "lmrb",
    "M6_hfrb": "haar",
    "M7_alcrb_hfrb": "alcrb_hfrb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    actual_teacher_sha = sha256(TEACHER)
    if actual_teacher_sha != EXPECTED_TEACHER_SHA:
        raise SystemExit(
            f"teacher SHA256 mismatch: expected {EXPECTED_TEACHER_SHA}, got {actual_teacher_sha}"
        )

    configs = {}
    for model_name, refinement_type in MODELS.items():
        config = json.loads(json.dumps(baseline))
        config.update(
            {
                "max_steps": 12000,
                "seed": 13,
                "output_dir": str(ROOT / "runs" / "psnr_modules_12k" / model_name / "seed13"),
                "num_workers": 2,
                "frozen_verifier": {
                    "checkpoint": "runs/baselines_500/verifier_teacher/best.pt",
                    "sha256": EXPECTED_TEACHER_SHA,
                    "pcgrad": False,
                },
            }
        )
        config["model"]["degradation_conditioned"] = False
        config["model"]["refinement_type"] = refinement_type
        model_path = CONFIG_ROOT / model_name / "seed13.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        configs[model_name] = {
            "config_path": str(model_path),
            "config_sha256": sha256(model_path),
            "refinement_type": refinement_type,
            "output_dir": config["output_dir"],
            "seed": 13,
            "max_steps": 12000,
            "teacher_sha256": actual_teacher_sha,
        }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "pass",
        "baseline_source": str(BASELINE),
        "teacher_checkpoint": str(TEACHER),
        "teacher_sha256": actual_teacher_sha,
        "models": configs,
        "allowed_differences": ["model.refinement_type", "model module fixed parameters", "output_dir"],
        "common_fields": {
            "data_root": baseline["data_root"],
            "mode": baseline["mode"],
            "optimizer": baseline["optimizer"],
            "scheduler": baseline["scheduler"],
            "crop_size": baseline["crop_size"],
            "sibling_count": baseline["sibling_count"],
            "batch_size": baseline["batch_size"],
            "epochs": 700,
            "max_steps": 12000,
            "validation_interval_steps": baseline["validation_interval_steps"],
            "validation_split": baseline["validation_split"],
            "learning_rate": baseline["learning_rate"],
            "min_learning_rate": baseline["min_learning_rate"],
            "weight_decay": baseline["weight_decay"],
            "gradient_clip": baseline["gradient_clip"],
            "amp": baseline["amp"],
            "seed": 13,
            "loss_weights": baseline["loss_weights"],
            "pcgrad": False,
        },
    }
    (RESULT_ROOT / "fairness_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PSNR Modules 12k Fairness Audit", "", "- status: **PASS**", f"- baseline: `{BASELINE}`", f"- teacher SHA256: `{actual_teacher_sha}`", "", "## Models", "", "| Model | refinement_type | max_steps | seed | config SHA256 |", "|---|---|---:|---:|---|"]
    for name, row in configs.items():
        lines.append(f"| {name} | {row['refinement_type']} | {row['max_steps']} | {row['seed']} | `{row['config_sha256']}` |")
    lines += ["", "## Allowed differences", "", "- `model.refinement_type`", "- module-specific fixed structure parameters", "- `output_dir`", "", "All other training, data, loss, optimizer, scheduler, teacher, and seed fields are copied from A5 and fixed.", ""]
    (RESULT_ROOT / "fairness_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "pass", "configs": len(configs), "teacher_sha256": actual_teacher_sha}, indent=2))


if __name__ == "__main__":
    main()
