#!/usr/bin/env python3
"""Run a two-optimizer-step GPU smoke test for each c007 candidate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "psnr_modules_12k"
SMOKE_ROOT = ROOT / "runs" / "psnr_modules_12k_smoke"
MODELS = [
    "M0_deg_fixed_baseline", "M1_transformer_refine", "M2_naf_refine",
    "M3_alcrb", "M4_rerh", "M5_lmrb", "M6_hfrb", "M7_alcrb_hfrb",
]


def main() -> None:
    for index, name in enumerate(MODELS):
        gpu = [0, 1, 3][index % 3]
        config = json.loads((CONFIG_ROOT / name / "seed13.json").read_text(encoding="utf-8"))
        config.update({"max_steps": 2, "smoke_train_sources": 8, "smoke_eval_items": 2, "num_workers": 0, "validation_interval_steps": 2})
        config["output_dir"] = str(SMOKE_ROOT / name / "seed13")
        smoke_config = SMOKE_ROOT / "configs" / f"{name}.json"
        smoke_config.parent.mkdir(parents=True, exist_ok=True)
        smoke_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log = SMOKE_ROOT / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "train.py"), "--config", str(smoke_config), "--device", "cuda"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        print(f"start {name} on GPU {gpu}; log={log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"smoke failed for {name}; inspect {log}")
        summary = json.loads((Path(config["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
        if summary.get("steps") != 2:
            raise SystemExit(f"smoke for {name} did not complete 2 steps: {summary}")
    print("all seven 2-step GPU smoke tests passed")


if __name__ == "__main__":
    main()
