from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "v02_backbone_decision"
DATA = ROOT / "data" / "plamd_vari_grip_pilot"
BASELINE = ROOT / "runs" / "ablation_v02_multiseed"
SEEDS = (13, 37, 73)
METHODS = ("independent", "group", "degradation_001")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def signature(config: dict, validation_count: int | None = None) -> dict:
    mode = str(config.get("mode"))
    batch = int(config.get("batch_size", 0))
    siblings = int(config.get("sibling_count", 1))
    return {
        "data_root": config.get("data_root"),
        "crop_size": config.get("crop_size"),
        "max_steps": config.get("max_steps"),
        "validation_interval_steps": config.get("validation_interval_steps"),
        "validation_split": config.get("validation_split", "val"),
        "learning_rate": config.get("learning_rate"),
        "min_learning_rate": config.get("min_learning_rate"),
        "optimizer": config.get("optimizer", "adamw"),
        "scheduler": config.get("scheduler", "cosine"),
        "weight_decay": config.get("weight_decay"),
        "gradient_clip": config.get("gradient_clip"),
        "amp": config.get("amp"),
        "num_workers": config.get("num_workers"),
        "eval_tile_size": config.get("eval_tile_size"),
        "eval_tile_overlap": config.get("eval_tile_overlap"),
        "model": config.get("model"),
        "sibling_count": siblings,
        "degraded_images_per_step": batch if mode == "independent" else batch * siblings,
        "validation_count": validation_count,
    }


def server_profile() -> dict:
    profile = {
        "linux_distribution": None,
        "kernel": platform.release(),
        "cpu_model": None,
        "cpu_cores": os.cpu_count(),
        "cpu_threads": os.cpu_count(),
        "total_ram": None,
        "gpus": [],
        "nvidia_driver": None,
        "cuda_runtime": None,
        "python_version": sys.version,
        "pytorch_version": None,
        "torchvision_version": None,
        "cudnn_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "disk_free_project": None,
        "num_workers": 8,
        "amp": True,
        "model_parameters": 1193121,
        "unknown_fields": [],
    }
    os_release = Path("/etc/os-release")
    if os_release.exists():
        values = {}
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip('"')
        profile["linux_distribution"] = values.get("PRETTY_NAME")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                profile["cpu_model"] = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                profile["total_ram"] = line.split(":", 1)[1].strip()
                break
    usage = shutil.disk_usage(ROOT)
    profile["disk_free_project"] = {"bytes": usage.free, "human": f"{usage.free / (1024 ** 4):.2f} TiB"}
    try:
        import torch

        profile["pytorch_version"] = torch.__version__
        profile["cuda_runtime"] = torch.version.cuda
        profile["cudnn_version"] = torch.backends.cudnn.version()
        try:
            import torchvision

            profile["torchvision_version"] = torchvision.__version__
        except Exception as exc:
            profile["unknown_fields"].append(f"torchvision_version: {exc}")
    except Exception as exc:
        profile["unknown_fields"].append(f"torch: {exc}")
    query = command("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader")
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            profile["gpus"].append({"name": parts[0], "uuid": parts[1], "memory_total": parts[2], "driver_version": parts[3]})
            profile["nvidia_driver"] = parts[3]
    if not profile["gpus"]:
        profile["unknown_fields"].append("GPU inventory unavailable")
    return profile


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_groups = read_json(DATA / "metadata" / "source_groups.json")
    with (DATA / "metadata" / "pilot_index.csv").open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    split_sources = {
        split: sorted({row["source_id"] for row in dataset_groups if row["split"] == split})
        for split in ("train", "val", "test")
    }
    manifest = {
        "root": str(DATA),
        "files": {
            "metadata/audit.json": sha256(DATA / "metadata" / "audit.json"),
            "metadata/pilot_index.csv": sha256(DATA / "metadata" / "pilot_index.csv"),
            "metadata/source_groups.json": sha256(DATA / "metadata" / "source_groups.json"),
        },
        "source_counts": {split: len(ids) for split, ids in split_sources.items()},
        "index_counts": {split: sum(row["split"] == split for row in index_rows) for split in ("train", "val", "test")},
        "validation_degradations": sorted({row["degradation"] for row in index_rows if row["split"] == "val"}),
    }
    config_paths = {
        "independent": [ROOT / "configs" / "v02_backbone_decision" / "independent" / f"seed{seed}.json" for seed in SEEDS],
        "group": [BASELINE / f"seed{seed}" / "group" / "resolved_config.json" for seed in SEEDS],
        "degradation_001": [BASELINE / f"seed{seed}" / "degradation_001" / "resolved_config.json" for seed in SEEDS],
    }
    config_signatures = {}
    for method, paths in config_paths.items():
        config_signatures[method] = []
        for seed, path in zip(SEEDS, paths):
            config = read_json(path)
            config_signatures[method].append({"seed": seed, "path": str(path), "signature": signature(config, 96)})
    reference = config_signatures["independent"][0]["signature"]
    differences = []
    for method, entries in config_signatures.items():
        for entry in entries:
            for key, expected in reference.items():
                if entry["signature"].get(key) != expected:
                    differences.append({"method": method, "seed": entry["seed"], "field": key, "reference": expected, "actual": entry["signature"].get(key)})
    old_independent = ROOT / "runs" / "pilot_independent"
    old_config = read_json(old_independent / "resolved_config.json")
    old_log_first = json.loads((old_independent / "train_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    reuse_reasons = []
    if not (old_independent / "best_validation.json").exists() or not (old_independent / "last_validation.json").exists():
        reuse_reasons.append("old independent has no best_validation.json and last_validation.json")
    for field in ("optimizer", "scheduler", "validation_split", "gradient_diagnostic_interval_steps"):
        if field not in old_config or old_config.get(field) is None:
            reuse_reasons.append(f"old independent resolved_config does not explicitly record {field}")
    if "reconstruction" not in old_log_first:
        reuse_reasons.append("old independent train_log uses legacy loss fields instead of current v0.2 fields")
    if old_config.get("output_dir") == "runs/pilot_independent":
        reuse_reasons.append("old independent belongs to pilot output namespace, not the current v0.2 decision namespace")
    fairness = {
        "passed": not differences,
        "differences": differences,
        "signatures": config_signatures,
        "independent_seed13_reuse_allowed": False,
        "independent_seed13_reuse_reasons": reuse_reasons,
        "old_independent_config": old_config,
    }
    report = {
        "status": "independent_seed13_retrain_required" if reuse_reasons else "independent_seed13_reuse_candidate",
        "goal": "fairly compare independent, group, and degradation_001 on fixed validation split",
        "dataset_manifest": manifest,
        "source_split_check": {split: len(ids) for split, ids in split_sources.items()},
        "method_scope": {"methods": ["independent", "group", "degradation_001"], "seeds": list(SEEDS), "test_split_used": False},
        "fairness_config_diff": fairness,
        "checkpoint_selection": {"required": "best_validation.json selected by validation PSNR; last_validation.json retained for audit", "old_independent_verifiable": False},
        "evaluator": {"script": "evaluate_source_fidelity.py", "sha256": sha256(ROOT / "evaluate_source_fidelity.py"), "split": "val", "tile_size": 512, "tile_overlap": 32},
        "server_profile": server_profile(),
    }
    (OUT / "fairness_config_diff.json").write_text(json.dumps(fairness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "server_profile.json").write_text(json.dumps(report["server_profile"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SiblingRestore v0.2 Backbone Decision Preflight",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Dataset and split",
        "",
        f"- manifest root: `{DATA}`",
        f"- source counts: train={len(split_sources['train'])}, val={len(split_sources['val'])}, test={len(split_sources['test'])}",
        f"- validation views: {manifest['index_counts']['val']}",
        f"- validation degradations: {', '.join(manifest['validation_degradations'])}",
        "- test split used: false",
        "",
        "## Fairness",
        "",
        f"- current decision configurations equal on shared fields: `{fairness['passed']}`",
        f"- configuration differences: `{len(differences)}`",
        f"- independent seed13 reuse allowed: `{fairness['independent_seed13_reuse_allowed']}`",
        "",
        "### Why the old independent seed13 is not reused",
        "",
    ]
    lines.extend(f"- {reason}" for reason in reuse_reasons)
    lines += ["", "## Plan", "", "- Train independent seed13, seed37, and seed73 in a new output namespace.", "- Reuse only the complete v0.2 group and degradation_001 artifacts after integrity checks.", "- Run 10,000-iteration source-level cluster bootstrap on validation source_id clusters.", "", "## Server", "", f"- GPU count recorded: {len(report['server_profile']['gpus'])}", f"- PyTorch: `{report['server_profile']['pytorch_version']}`", f"- CUDA runtime: `{report['server_profile']['cuda_runtime']}`", f"- project free disk: `{report['server_profile']['disk_free_project']}`", ""]
    (OUT / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "fairness_passed": fairness["passed"], "reuse_independent_seed13": False, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
