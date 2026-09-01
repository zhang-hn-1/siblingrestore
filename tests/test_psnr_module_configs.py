from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "psnr_modules_12k"
MODELS = {
    "M0_deg_fixed_baseline": "none",
    "M1_transformer_refine": "transformer",
    "M2_naf_refine": "naf",
    "M3_alcrb": "alcrb",
    "M4_rerh": "rerh",
    "M5_lmrb": "lmrb",
    "M6_hfrb": "haar",
}


def test_all_configs_have_strict_common_budget():
    configs = []
    for model_name, refinement_type in MODELS.items():
        path = CONFIG_ROOT / model_name / "seed13.json"
        assert path.exists()
        config = json.loads(path.read_text(encoding="utf-8"))
        configs.append(config)
        assert config["model"]["refinement_type"] == refinement_type
        assert config["max_steps"] == 12000
        assert config["seed"] == 13
        assert config["loss_weights"]["output"] == 0.0
        assert config["loss_weights"]["degradation"] == 0.01
        assert config["loss_weights"]["anchor"] == 0.01
        assert config["frozen_verifier"]["pcgrad"] is False
        assert config["frozen_verifier"]["sha256"] == "16a87f86a21f7f5c4d956a48925f027fd63c91046c697a67e686e93d85eccf23"

    base = configs[0]
    for config in configs[1:]:
        for key in ("data_root", "mode", "optimizer", "scheduler", "device", "crop_size", "sibling_count", "batch_size", "epochs", "validation_interval_steps", "validation_split", "gradient_diagnostic_interval_steps", "eval_tile_size", "eval_tile_overlap", "learning_rate", "min_learning_rate", "weight_decay", "gradient_clip", "amp", "seed", "loss_weights", "frozen_verifier"):
            assert config[key] == base[key], key
        for key in ("dim", "blocks_per_level", "heads", "projection_dim", "degradation_conditioned"):
            assert config["model"][key] == base["model"][key], key


def test_fairness_audit_exists_and_passes():
    audit = json.loads((ROOT / "results" / "psnr_modules_12k" / "fairness_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert len(audit["models"]) == 7
