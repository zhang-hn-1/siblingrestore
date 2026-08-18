from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution as `python scripts/analyze_v02_backbone_decision.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_loss_ablation import cluster_bootstrap, finite_scalars

METHODS = ("independent", "group", "degradation_001")
SEEDS = (13, 37, 73)
DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")
PAIRS = (
    ("degradation_001_vs_independent", "degradation_001", "independent"),
    ("group_vs_independent", "group", "independent"),
    ("degradation_001_vs_group", "degradation_001", "group"),
)
METRICS = ("psnr", "ssim")
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260814


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_fidelity(path: Path) -> dict[str, dict[str, dict]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(str(row["source_id"]), {})[str(row["degradation"])] = row
    return result


def config_signature(config: dict, validation_count: int | None) -> dict:
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


def source_mean(record: dict, source_id: str, metric: str, degradation: str | None = None) -> float:
    selected = [degradation] if degradation is not None else list(DEGRADATIONS)
    return statistics.mean(float(record["fidelity_map"][source_id][name][metric]) for name in selected)


def load_record(root: Path, baseline_root: Path, method: str, seed: int, integrity: dict) -> dict | None:
    directory = root / "independent" / f"seed{seed}" if method == "independent" else baseline_root / f"seed{seed}" / method
    required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "source_fidelity_val.json", "source_fidelity_val.csv")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        integrity["missing_artifacts"].append({"method": method, "seed": seed, "path": str(directory), "missing": missing})
        return None
    config = read_json(directory / "resolved_config.json")
    best = read_json(directory / "best_validation.json")
    last = read_json(directory / "last_validation.json")
    summary = read_json(directory / "summary.json")
    fidelity = read_json(directory / "source_fidelity_val.json")
    train_rows = [json.loads(x) for x in (directory / "train_log.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    history_rows = [json.loads(x) for x in (directory / "validation_history.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    for name, payload in (("summary", summary), ("best_validation", best), ("last_validation", last), ("source_fidelity", fidelity)):
        bad = finite_scalars(payload)
        if bad:
            integrity["warnings"].append({"method": method, "seed": seed, "kind": "nonfinite", "file": name, "fields": bad})
    rows = list(csv.DictReader((directory / "source_fidelity_val.csv").open("r", encoding="utf-8-sig", newline="")))
    keys = [(row.get("source_id"), row.get("degradation")) for row in rows]
    if len(rows) != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "view_count", "value": len(rows)})
    if len(set(row.get("source_id") for row in rows)) != 16:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "source_count"})
    if len(keys) != len(set(keys)):
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "duplicate_source_degradation"})
    by_source = {}
    for row in rows:
        by_source.setdefault(row.get("source_id"), []).append(row.get("degradation"))
        for key in ("psnr", "ssim"):
            try:
                if not math.isfinite(float(row[key])):
                    integrity["warnings"].append({"method": method, "seed": seed, "kind": "nonfinite_csv", "field": key})
            except (KeyError, TypeError, ValueError):
                integrity["warnings"].append({"method": method, "seed": seed, "kind": "invalid_csv", "field": key})
    incomplete = {source: sorted(values) for source, values in by_source.items() if len(values) != 6 or set(values) != set(DEGRADATIONS)}
    if incomplete:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "source_degradation_completeness", "value": incomplete})
    if fidelity.get("split") != "val" or fidelity.get("sources") != 16 or fidelity.get("restored_views") != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "validation_count_or_split"})
    history_steps = {int(row.get("step", -1)) for row in history_rows}
    best_step = int(best.get("step", -1))
    if (
        int(summary.get("steps", 0)) != 5000
        or int(last.get("step", -1)) != 5000
        or best_step < 0
        or best_step > 5000
        or best_step not in history_steps
    ):
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "step_or_checkpoint_selection"})
    if abs(float(best["aggregate"]["psnr"]) - float(fidelity["aggregate"]["psnr"])) > 1e-4:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "best_checkpoint_fidelity_mismatch"})
    return {"method": method, "seed": seed, "directory": directory, "config": config, "best": best, "last": last, "summary": summary, "fidelity": fidelity, "fidelity_rows": rows, "fidelity_map": load_fidelity(directory / "source_fidelity_val.csv"), "train_rows": train_rows, "history_rows": history_rows}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    root, baseline_root, seeds = args.root, args.baseline_root, tuple(args.seeds)
    root.mkdir(parents=True, exist_ok=True)
    integrity = {"missing_artifacts": [], "warnings": [], "expected_methods": list(METHODS), "expected_seeds": list(seeds), "expected_sources": 16, "expected_views_per_method_seed": 96, "expected_degradations_per_source": 6, "test_split_used": False}
    records = {}
    for method in METHODS:
        for seed in seeds:
            record = load_record(root, baseline_root, method, seed, integrity)
            if record is not None:
                records[(method, seed)] = record
    signatures = {f"{method}/seed{seed}": config_signature(record["config"], record["best"].get("count")) for (method, seed), record in records.items()}
    reference = signatures.get("independent/seed13")
    differences = []
    if reference is None:
        differences.append({"field": "reference", "reason": "independent seed13 missing"})
    else:
        for label, current in signatures.items():
            for key, expected in reference.items():
                if current.get(key) != expected:
                    differences.append({"label": label, "field": key, "reference": expected, "actual": current.get(key)})
    fairness = {"passed": not differences, "differences": differences, "signatures": signatures, "expected_batch_semantics": {"independent": {"batch_size": 8, "degraded_images_per_step": 8}, "group": {"batch_size": 4, "sibling_count": 2, "degraded_images_per_step": 8}, "degradation_001": {"batch_size": 4, "sibling_count": 2, "degraded_images_per_step": 8}}}
    (root / "fairness_config_diff.json").write_text(json.dumps(fairness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_seed_rows = []
    for (method, seed), record in sorted(records.items()):
        per_seed_rows.append({"method": method, "seed": seed, "psnr": record["best"]["aggregate"]["psnr"], "ssim": record["best"]["aggregate"]["ssim"], "best_step": record["best"].get("step"), "best_epoch": record["best"].get("epoch"), "last_step": record["last"].get("step"), "last_epoch": record["last"].get("epoch"), "elapsed_seconds": record["summary"].get("elapsed_seconds"), "parameters": record["summary"].get("parameters"), "peak_gpu_memory_mib": None, "peak_gpu_memory_note": "not recorded by train.py"})
    write_csv(root / "per_seed_results.csv", per_seed_rows)

    paired_rows, bootstrap_rows, per_degradation_rows = [], [], []
    cluster_report = {}
    for comparison, model_a, model_b in PAIRS:
        cluster_report[comparison] = {}
        for metric in METRICS:
            per_seed_source = {}
            per_seed_pairs = []
            for seed in seeds:
                if (model_a, seed) not in records or (model_b, seed) not in records:
                    continue
                a, b = records[(model_a, seed)], records[(model_b, seed)]
                common = sorted(set(a["fidelity_map"]) & set(b["fidelity_map"]))
                deltas = {source: source_mean(a, source, metric) - source_mean(b, source, metric) for source in common}
                per_seed_source[seed] = deltas
                per_seed_pairs.append({"seed": seed, "delta": a["best"]["aggregate"][metric] - b["best"]["aggregate"][metric]})
                ci = cluster_bootstrap(deltas, BOOTSTRAP_SEED + seed, BOOTSTRAP_ITERATIONS)
                bootstrap_rows.append({"comparison": comparison, "metric": metric, "seed": seed, "mean_delta": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "n_sources": ci["source_count"], "n_seeds": 1, "bootstrap_unit": "source_id", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED + seed})
                paired_rows.append({"comparison": comparison, "seed": seed, "metric": metric, "model_a": model_a, "model_b": model_b, "model_a_value": a["best"]["aggregate"][metric], "model_b_value": b["best"]["aggregate"][metric], "paired_delta": a["best"]["aggregate"][metric] - b["best"]["aggregate"][metric]})
            aggregate_source = {}
            for source in sorted({source for values in per_seed_source.values() for source in values}):
                values = [per_seed_source[seed][source] for seed in seeds if source in per_seed_source.get(seed, {})]
                if len(values) == len(seeds):
                    aggregate_source[source] = statistics.mean(values)
            ci = cluster_bootstrap(aggregate_source, BOOTSTRAP_SEED + len(bootstrap_rows), BOOTSTRAP_ITERATIONS)
            aggregate_row = {"comparison": comparison, "metric": metric, "seed": "aggregate", "mean_delta": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "n_sources": ci["source_count"], "n_seeds": len(seeds), "bootstrap_unit": "source_id", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED + len(bootstrap_rows)}
            bootstrap_rows.append(aggregate_row)
            cluster_report[comparison][metric] = {"per_seed": per_seed_pairs, "aggregate": aggregate_row}
        if comparison in ("degradation_001_vs_independent", "group_vs_independent"):
            for degradation in DEGRADATIONS:
                for metric in METRICS:
                    per_seed_source = {}
                    for seed in seeds:
                        if (model_a, seed) not in records or (model_b, seed) not in records:
                            continue
                        a, b = records[(model_a, seed)], records[(model_b, seed)]
                        common = sorted(set(a["fidelity_map"]) & set(b["fidelity_map"]))
                        deltas = {source: source_mean(a, source, metric, degradation) - source_mean(b, source, metric, degradation) for source in common}
                        per_seed_source[seed] = deltas
                        ci = cluster_bootstrap(deltas, BOOTSTRAP_SEED + len(per_degradation_rows), BOOTSTRAP_ITERATIONS)
                        per_degradation_rows.append({"comparison": comparison, "seed": seed, "degradation": degradation, "metric": metric, "n_sources": ci["source_count"], "n_seeds": 1, "paired_delta_mean": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "bootstrap_unit": "source_id"})
                    aggregate_source = {}
                    for source in sorted({source for values in per_seed_source.values() for source in values}):
                        values = [per_seed_source[seed][source] for seed in seeds if source in per_seed_source.get(seed, {})]
                        if len(values) == len(seeds):
                            aggregate_source[source] = statistics.mean(values)
                    ci = cluster_bootstrap(aggregate_source, BOOTSTRAP_SEED + len(per_degradation_rows), BOOTSTRAP_ITERATIONS)
                    per_degradation_rows.append({"comparison": comparison, "seed": "aggregate", "degradation": degradation, "metric": metric, "n_sources": ci["source_count"], "n_seeds": len(seeds), "paired_delta_mean": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "bootstrap_unit": "source_id"})
    write_csv(root / "paired_comparisons.csv", paired_rows)
    write_csv(root / "bootstrap_results.csv", bootstrap_rows)
    write_csv(root / "per_degradation_results.csv", per_degradation_rows)

    integrity["fairness_differences"] = differences
    integrity["fairness_passed"] = fairness["passed"]
    integrity["missing_artifacts_count"] = len(integrity["missing_artifacts"])
    integrity["warnings_count"] = len(integrity["warnings"])
    integrity["complete"] = not integrity["missing_artifacts"] and not integrity["warnings"] and fairness["passed"] and len(records) == len(METHODS) * len(seeds)
    integrity["evaluator"] = {"script": "evaluate_source_fidelity.py", "validation_split": "val", "schema_checked": True}
    (root / "data_integrity_report.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {}
    for method in METHODS:
        metrics[method] = {}
        for metric in METRICS:
            values = [float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records]
            metrics[method][metric] = {"mean": statistics.mean(values) if values else None, "std": statistics.stdev(values) if len(values) >= 2 else None, "per_seed": {str(seed): float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records}}
    decision = {"status": "incomplete", "final_paper_main_model": False, "summary": "数据不完整或公平性检查失败，禁止主干判定。"}
    if integrity["complete"]:
        psnr = [row["paired_delta"] for row in paired_rows if row["comparison"] == "degradation_001_vs_independent" and row["metric"] == "psnr"]
        ssim = [row["paired_delta"] for row in paired_rows if row["comparison"] == "degradation_001_vs_independent" and row["metric"] == "ssim"]
        ci = cluster_report["degradation_001_vs_independent"]["psnr"]["aggregate"]
        criteria = {"mean_psnr_delta_gt_zero": statistics.mean(psnr) > 0, "mean_ssim_delta_ge_zero": statistics.mean(ssim) >= 0, "at_least_two_of_three_psnr_deltas_gt_zero": sum(x > 0 for x in psnr) >= 2, "no_seed_psnr_delta_below_minus_0_15": min(psnr) >= -0.15, "source_cluster_psnr_ci_lower_gt_zero": float(ci["ci_lower"]) > 0, "fairness_and_integrity_passed": True}
        if all(criteria.values()):
            status, summary = "provisional_backbone_pass", "degradation_001 satisfies the preregistered provisional engineering backbone rules; this is not a final paper main-model claim."
        elif statistics.mean(psnr) > 0 and float(ci["ci_lower"]) <= 0:
            status, summary = "promising_but_inconclusive", "degradation_001 has a positive mean but the source-level CI95 includes zero."
        else:
            status, summary = "reject_as_backbone", "degradation_001 does not satisfy the preregistered stable backbone rules."
        decision = {"status": status, "final_paper_main_model": False, "summary": summary, "criteria": criteria, "main_comparison": "degradation_001_vs_independent"}

    report = {"metadata": {"analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds), "methods": list(METHODS), "bootstrap_unit": "source_id", "bootstrap_description": "Each source carries all 6 degradations and all 3 seed records.", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED, "ci_level": 0.95, "validation_sources": 16, "validation_views": 96, "test_split_used": False}, "metrics": metrics, "paired_comparisons": paired_rows, "bootstrap_results": bootstrap_rows, "per_degradation_results": per_degradation_rows, "training_config": {"fairness": fairness, "num_workers": 8, "amp": True, "model_parameters": 1193121}, "data_integrity": integrity, "decision": decision, "limitations": ["Only fixed validation was used; test remains sealed.", "own-clean top-1 is saturated and appendix-only.", "Peak GPU memory was not recorded by train.py and is null in per_seed_results.csv.", "provisional engineering backbone is not a final paper main-model claim."]}
    (root / "multiseed_backbone_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# SiblingRestore v0.2 Backbone Decision Report", "", f"Decision status: {decision['status']}", "", "## 1. Experimental Setup", "", "- Methods: independent, group, degradation_001.", f"- Seeds: {', '.join(str(seed) for seed in seeds)}.", "- Validation only: 16 sources × 6 degradations = 96 views.", "- Test split was not used.", "- All methods use the same backbone and 5,000 optimization steps.", "- Independent uses batch_size=8; grouped methods use batch_size=4 and sibling_count=2; all process 8 degraded images per step.", "", "## 2. Server Configuration", ""]
    server_path = root / "server_profile.json"
    if server_path.exists():
        server = read_json(server_path)
        lines += [f"- Linux: {server.get('linux_distribution')}", f"- Kernel: {server.get('kernel')}", f"- CPU: {server.get('cpu_model')} ({server.get('cpu_cores')} logical CPUs)", f"- RAM: {server.get('total_ram')}", f"- GPUs: {len(server.get('gpus', []))}", f"- PyTorch/CUDA/cuDNN: {server.get('pytorch_version')} / {server.get('cuda_runtime')} / {server.get('cudnn_version')}", f"- num_workers: {server.get('num_workers')}; AMP: {server.get('amp')}", ""]
    lines += ["## 3. Fairness Check", "", f"- passed: {fairness['passed']}", f"- differences: {len(differences)}", f"- data integrity complete: {integrity['complete']}", "", "## 4. Three-Seed Main Metrics", "", "| method | PSNR mean | PSNR sample std | SSIM mean | SSIM sample std |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | {metrics[method]['psnr']['mean']} | {metrics[method]['psnr']['std']} | {metrics[method]['ssim']['mean']} | {metrics[method]['ssim']['std']} |")
    lines += ["", "## 5. Per-Seed Paired Deltas", ""]
    for comparison, _, _ in PAIRS:
        lines.append(f"### {comparison}")
        for metric in METRICS:
            entries = [row for row in paired_rows if row["comparison"] == comparison and row["metric"] == metric]
            lines.append("- " + metric + ": " + ", ".join(f"seed{row['seed']}={row['paired_delta']:.6f}" for row in entries))
    lines += ["", "## 6. Source-Level Cluster Bootstrap", "", "- Bootstrap unit: source_id.", "- Each sampled source carries all six degradations and all three seed records.", "- 10,000 iterations, fixed bootstrap seed 20260814.", "", "| comparison | metric | mean delta | CI95 lower | CI95 upper | n_sources | n_seeds |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in bootstrap_rows:
        if row["seed"] == "aggregate":
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_delta']} | {row['ci_lower']} | {row['ci_upper']} | {row['n_sources']} | {row['n_seeds']} |")
    lines += ["", "## 7. Per-Degradation Analysis", "", "Aggregate rows are in per_degradation_results.csv; each uses source-level bootstrap with n_sources=16.", ""]
    for comparison in ("degradation_001_vs_independent", "group_vs_independent"):
        lines.append(f"### {comparison}")
        for row in per_degradation_rows:
            if row["comparison"] == comparison and row["seed"] == "aggregate" and row["metric"] == "psnr":
                lines.append(f"- {row['degradation']}: delta={row['paired_delta_mean']}, CI95=[{row['ci_lower']}, {row['ci_upper']}]")
    lines += ["", "## 8. Data Integrity Checks", "", f"- missing artifacts: {len(integrity['missing_artifacts'])}", f"- warnings: {len(integrity['warnings'])}", "", "## 9. Backbone Decision", "", decision["summary"], "", f"- final_paper_main_model: {decision['final_paper_main_model']}", "", "## 10. Remaining Limitations", ""] + [f"- {item}" for item in report["limitations"]] + ["", "## 11. Recommended Next Step", "", "Use degradation_001 only as a provisional engineering backbone if it passes; do not call it final paper main model.", ""]
    (root / "multiseed_backbone_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "integrity_complete": integrity["complete"], "fairness_passed": fairness["passed"]}, ensure_ascii=False))
    if not integrity["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
