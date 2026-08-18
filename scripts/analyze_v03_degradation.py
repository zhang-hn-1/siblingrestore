from __future__ import annotations

"""v0.3 degradation-line analysis: fairness, paired source-level bootstrap,
per-degradation deltas, gradient/loss mechanism checks, and the main-model
decision gate for the four new methods vs the v0.2 controls.

Usage: python scripts/analyze_v03_degradation.py --root runs/v03_degradation \
       --control-root runs/ablation_v02_multiseed --seeds 13 37 73
"""

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_loss_ablation import cluster_bootstrap, finite_scalars

NEW_METHODS = ("degradation_003", "degradation_005", "degradation_001_output_001", "degradation_cond_001")
CONTROLS = ("independent", "group", "degradation_001")
ALL_METHODS = CONTROLS + NEW_METHODS
SEEDS = (13, 37, 73)
DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")
METRICS = ("psnr", "ssim")
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260815
REQUIRED = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "source_fidelity_val.json", "source_fidelity_val.csv")
EXPECTED_PARAMS = {False: 1193121, True: 1222689}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_fidelity_map(path: Path) -> dict[str, dict[str, dict]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(str(row["source_id"]), {})[str(row["degradation"])] = row
    return result


def normalized_model(config: dict) -> dict:
    """Model signature with the A4 conditioning flag stripped (expected diff)."""
    model = dict(config.get("model", {}))
    model.pop("degradation_conditioned", None)
    return model


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
        "model": normalized_model(config),
        "sibling_count": siblings,
        "degraded_images_per_step": batch if mode == "independent" else batch * siblings,
        "validation_count": validation_count,
    }


def method_directory(root: Path, control_root: Path, method: str, seed: int) -> Path:
    if method == "independent":
        return PROJECT_ROOT / "runs" / "v02_backbone_decision" / "independent" / f"seed{seed}"
    if method in CONTROLS:
        return control_root / f"seed{seed}" / method
    return root / method / f"seed{seed}"


def load_record(root: Path, control_root: Path, method: str, seed: int, integrity: dict) -> dict | None:
    directory = method_directory(root, control_root, method, seed)
    missing = [name for name in REQUIRED if not (directory / name).exists()]
    if missing:
        integrity["missing_artifacts"].append({"method": method, "seed": seed, "path": str(directory), "missing": missing})
        return None
    config = read_json(directory / "resolved_config.json")
    best = read_json(directory / "best_validation.json")
    last = read_json(directory / "last_validation.json")
    summary = read_json(directory / "summary.json")
    fidelity = read_json(directory / "source_fidelity_val.json")
    rows = list(csv.DictReader((directory / "source_fidelity_val.csv").open("r", encoding="utf-8-sig", newline="")))
    for name, payload in (("summary", summary), ("best_validation", best), ("last_validation", last), ("source_fidelity", fidelity)):
        bad = finite_scalars(payload)
        if bad:
            integrity["warnings"].append({"method": method, "seed": seed, "kind": "nonfinite", "file": name, "fields": bad})
    if len(rows) != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "view_count", "value": len(rows)})
    if len(set(row.get("source_id") for row in rows)) != 16:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "source_count"})
    if len(set((row.get("source_id"), row.get("degradation")) for row in rows)) != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "duplicate_source_degradation"})
    if fidelity.get("split") != "val" or fidelity.get("sources") != 16 or fidelity.get("restored_views") != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "validation_count_or_split"})
    if int(summary.get("steps", 0)) != 5000:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "step_count"})
    expected = EXPECTED_PARAMS[bool(config["model"].get("degradation_conditioned", False))]
    if int(summary.get("parameters", 0)) != expected:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "parameter_count", "value": summary.get("parameters"), "expected": expected})
    return {
        "method": method, "seed": seed, "directory": directory, "config": config,
        "best": best, "last": last, "summary": summary, "fidelity": fidelity,
        "fidelity_rows": rows, "fidelity_map": load_fidelity_map(directory / "source_fidelity_val.csv"),
        "train_rows": [json.loads(x) for x in (directory / "train_log.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()],
    }


def source_mean(record: dict, source_id: str, metric: str, degradation: str | None = None) -> float:
    selected = [degradation] if degradation is not None else list(DEGRADATIONS)
    return statistics.mean(float(record["fidelity_map"][source_id][name][metric]) for name in selected)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loss_trend(record: dict, key: str) -> dict:
    values = [float(row[key]) for row in record["train_rows"] if row.get(key) is not None]
    if not values:
        return {"first": None, "last": None, "min": None, "count": 0}
    return {"first": values[0], "last": values[-1], "min": min(values), "count": len(values)}


def grad_ratio(record: dict, key: str) -> float | None:
    values = [float(row[key]) for row in record["train_rows"] if row.get(key) is not None]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    root, control_root, seeds = args.root, args.control_root, tuple(args.seeds)
    root.mkdir(parents=True, exist_ok=True)
    integrity = {"missing_artifacts": [], "warnings": [], "expected_methods": list(ALL_METHODS), "expected_seeds": list(seeds), "expected_sources": 16, "expected_views_per_method_seed": 96, "test_split_used": False}
    records = {}
    for method in ALL_METHODS:
        for seed in seeds:
            record = load_record(root, control_root, method, seed, integrity)
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
    fairness = {"passed": not differences, "differences": differences, "signatures": signatures, "expected_batch_semantics": {"independent": {"batch_size": 8, "degraded_images_per_step": 8}, "grouped": {"batch_size": 4, "sibling_count": 2, "degraded_images_per_step": 8}}}
    (root / "fairness_config_diff.json").write_text(json.dumps(fairness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_seed_rows = []
    for (method, seed), record in sorted(records.items()):
        per_seed_rows.append({"method": method, "seed": seed, "psnr": record["best"]["aggregate"]["psnr"], "ssim": record["best"]["aggregate"]["ssim"], "best_step": record["best"].get("step"), "last_step": record["last"].get("step"), "elapsed_seconds": record["summary"].get("elapsed_seconds"), "parameters": record["summary"].get("parameters"), "degradation_conditioned": bool(record["config"]["model"].get("degradation_conditioned", False))})
    write_csv(root / "per_seed_results.csv", per_seed_rows)

    # Paired comparisons: every new method vs every control.
    comparisons = [(f"{method}_vs_{control}", method, control) for method in NEW_METHODS for control in CONTROLS]
    paired_rows, bootstrap_rows, per_degradation_rows = [], [], []
    cluster_report = {}
    for comparison, model_a, model_b in comparisons:
        cluster_report[comparison] = {}
        for metric in METRICS:
            per_seed_source, per_seed_pairs = {}, []
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

    # Mechanism checks for the new methods (degradation CE trend, gradient ratios).
    mechanism = {}
    for method in NEW_METHODS:
        mechanism[method] = {}
        for seed in seeds:
            if (method, seed) not in records:
                continue
            record = records[(method, seed)]
            mechanism[method][seed] = {
                "degradation_classification": loss_trend(record, "degradation_classification"),
                "output_consistency": loss_trend(record, "output_consistency"),
                "weighted_grad_ratio_degradation": grad_ratio(record, "weighted_grad_ratio_degradation_to_rec"),
                "weighted_grad_ratio_output": grad_ratio(record, "weighted_grad_ratio_output_to_rec"),
            }
    (root / "mechanism_diagnostics.json").write_text(json.dumps(mechanism, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    integrity["fairness_differences"] = differences
    integrity["fairness_passed"] = fairness["passed"]
    integrity["missing_artifacts_count"] = len(integrity["missing_artifacts"])
    integrity["warnings_count"] = len(integrity["warnings"])
    integrity["complete"] = not integrity["missing_artifacts"] and not integrity["warnings"] and fairness["passed"] and len(records) == len(ALL_METHODS) * len(seeds)
    (root / "data_integrity_report.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {}
    for method in ALL_METHODS:
        metrics[method] = {}
        for metric in METRICS:
            values = [float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records]
            metrics[method][metric] = {"mean": statistics.mean(values) if values else None, "std": statistics.stdev(values) if len(values) >= 2 else None, "per_seed": {str(seed): float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records}}

    # Decision: best new method vs the incumbent degradation_001, using the v0.2 gate
    # against independent as the reference hypothesis, and a secondary gate vs group.
    decision = {"status": "incomplete", "final_paper_main_model": False, "summary": "数据不完整或公平性检查失败，禁止判定。", "candidates": {}}
    if integrity["complete"]:
        for method in NEW_METHODS:
            comparison = f"{method}_vs_independent"
            psnr = [row["paired_delta"] for row in paired_rows if row["comparison"] == comparison and row["metric"] == "psnr"]
            ssim = [row["paired_delta"] for row in paired_rows if row["comparison"] == comparison and row["metric"] == "ssim"]
            ci = cluster_report[comparison]["psnr"]["aggregate"]
            criteria = {
                "mean_psnr_delta_gt_zero": statistics.mean(psnr) > 0,
                "mean_ssim_delta_ge_zero": statistics.mean(ssim) >= 0,
                "at_least_two_of_three_psnr_deltas_gt_zero": sum(x > 0 for x in psnr) >= 2,
                "no_seed_psnr_delta_below_minus_0_15": min(psnr) >= -0.15,
                "source_cluster_psnr_ci_lower_gt_zero": float(ci["ci_lower"]) > 0,
            }
            if all(criteria.values()):
                status = "significant_pass"
            elif statistics.mean(psnr) > 0 and float(ci["ci_lower"]) <= 0:
                status = "promising_but_inconclusive"
            else:
                status = "reject"
            decision["candidates"][method] = {"status": status, "criteria": criteria, "main_comparison": comparison, "mean_psnr_delta": statistics.mean(psnr), "ci": ci}
        passed = [name for name, candidate in decision["candidates"].items() if candidate["status"] == "significant_pass"]
        if passed:
            decision["status"] = "significant_pass"
            decision["final_paper_main_model"] = True
            decision["summary"] = f"{passed} 满足全部预注册门槛（vs independent，source-level CI 排除 0），可作为论文主模型候选。"
        else:
            inconclusive = [name for name, candidate in decision["candidates"].items() if candidate["status"] == "promising_but_inconclusive"]
            if inconclusive:
                decision["status"] = "promising_but_inconclusive"
                decision["summary"] = f"{inconclusive} 均值正向但 source-level CI95 含 0，尚不能作为主模型；本轮 loss 级到顶，建议转 B2 或接受 degradation_001 为工程主干。"
            else:
                decision["status"] = "reject"
                decision["summary"] = "所有新变体均未满足门槛；degradation 权重扫描和 FiLM 条件化在 loss 级无显著收益。"

    report = {
        "metadata": {"analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds), "methods": list(ALL_METHODS), "new_methods": list(NEW_METHODS), "bootstrap_unit": "source_id", "bootstrap_description": "Each source carries all 6 degradations and all 3 seed records.", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED, "ci_level": 0.95, "validation_sources": 16, "validation_views": 96, "test_split_used": False},
        "metrics": metrics, "paired_comparisons": paired_rows, "bootstrap_results": bootstrap_rows, "per_degradation_results": per_degradation_rows, "mechanism_diagnostics": mechanism, "training_config": {"fairness": fairness, "num_workers": 8, "amp": True, "expected_parameters": EXPECTED_PARAMS}, "data_integrity": integrity, "decision": decision,
        "limitations": ["Only fixed validation was used; test remains sealed.", "own-clean top-1 is saturated; embed_own_clean_top1 is a representation check, not a paper metric.", "Peak GPU memory was not recorded by train.py.", "A4 FiLM is zero-init at start; conditioning strength is learned, not tuned."],
    }
    (root / "v03_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# SiblingRestore v0.3 Degradation-Line Report", "", f"Decision status: {decision['status']}", "", "## 1. Experimental Setup", "", f"- Methods: {', '.join(ALL_METHODS)}.", f"- Seeds: {', '.join(str(seed) for seed in seeds)}.", "- Validation only: 16 sources × 6 degradations = 96 views; test sealed.", "- All methods share the same backbone, 5,000 steps, crop 256, batch semantics (8 degraded images/step).", "- New methods keep the degradation classification aux; A4 adds learned FiLM conditioning (zero-init).", "", "## 2. Fairness Check", "", f"- passed: {fairness['passed']}", f"- differences: {len(differences)}", f"- data integrity complete: {integrity['complete']}", "", "## 3. Three-Seed Main Metrics (best checkpoint)", "", "| method | PSNR mean | PSNR std | SSIM mean | SSIM std |", "|---|---:|---:|---:|---:|"]

    def fmt(value: float | None) -> str:
        return "None" if value is None else f"{value:.6f}"

    def fmt4(value: float | None) -> str:
        return "None" if value is None else f"{value:.4f}"

    for method in ALL_METHODS:
        lines.append(f"| {method} | {fmt(metrics[method]['psnr']['mean'])} | {fmt(metrics[method]['psnr']['std'])} | {fmt(metrics[method]['ssim']['mean'])} | {fmt(metrics[method]['ssim']['std'])} |")
    lines += ["", "## 4. Paired Source-Level Bootstrap (aggregate over 3 seeds)", "", "| comparison | metric | mean delta | CI95 lower | CI95 upper | n_sources | n_seeds |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in bootstrap_rows:
        if row["seed"] == "aggregate" and row["metric"] == "psnr":
            lines.append(f"| {row['comparison']} | {row['metric']} | {fmt(row['mean_delta'])} | {fmt(row['ci_lower'])} | {fmt(row['ci_upper'])} | {row['n_sources']} | {row['n_seeds']} |")
    lines += ["", "## 5. Per-Degradation Deltas (new methods vs independent, aggregate PSNR)", ""]
    for method in NEW_METHODS:
        comparison = f"{method}_vs_independent"
        lines.append(f"### {comparison}")
        for row in per_degradation_rows:
            if row["comparison"] == comparison and row["seed"] == "aggregate" and row["metric"] == "psnr":
                lines.append(f"- {row['degradation']}: delta={fmt(row['paired_delta_mean'])}, CI95=[{fmt(row['ci_lower'])}, {fmt(row['ci_upper'])}]")
    lines += ["", "## 6. Mechanism Checks", "", "| method | seed | deg-CE first | deg-CE last | deg-CE min | weighted grad ratio (deg) |", "|---|---:|---:|---:|---:|---:|"]
    for method in NEW_METHODS:
        for seed in seeds:
            if method not in mechanism or seed not in mechanism[method]:
                continue
            m = mechanism[method][seed]
            first, last, minimum = m["degradation_classification"]["first"], m["degradation_classification"]["last"], m["degradation_classification"]["min"]
            lines.append(f"| {method} | {seed} | {fmt4(first)} | {fmt4(last)} | {fmt4(minimum)} | {m['weighted_grad_ratio_degradation']} |")
    lines += ["", "## 7. Data Integrity", "", f"- missing artifacts: {len(integrity['missing_artifacts'])}", f"- warnings: {len(integrity['warnings'])}", "", "## 8. Decision", "", decision["summary"], "", f"- final_paper_main_model: {decision['final_paper_main_model']}", "", "## 9. Limitations", ""] + [f"- {item}" for item in report["limitations"]] + [""]
    (root / "v03_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "integrity_complete": integrity["complete"], "fairness_passed": fairness["passed"]}, ensure_ascii=False))
    if not integrity["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
