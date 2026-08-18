from __future__ import annotations

"""v0.4 identity-line analysis: fairness, paired source-level bootstrap,
identity diagnostics (embed retrieval / intra-inter ratio), and the dual-gate
main-model decision (PSNR not worse + identity clearly better).

Usage: python scripts/analyze_v04_identity.py --root runs/v04_identity \
       --incumbent-root runs/v03_degradation --control-root runs/ablation_v02_multiseed \
       --seeds 13 37 73
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

NEW_METHODS = ("identity_class_001", "identity_branch_001", "output_contrast_001")
CONTROLS = ("independent", "group", "degradation_001", "degradation_001_output_001")
ALL_METHODS = CONTROLS + NEW_METHODS
SEEDS = (13, 37, 73)
DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")
METRICS = ("psnr", "ssim")
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260816
REQUIRED = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "source_fidelity_val.json", "source_fidelity_val.csv")
PSNR_FLOOR_DB = -0.15
EMBED_TARGET = 0.6
EMBED_MIN_GAIN = 0.1


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_fidelity_map(path: Path) -> dict[str, dict[str, dict]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(str(row["source_id"]), {})[str(row["degradation"])] = row
    return result


def normalized_model(config: dict) -> dict:
    model = dict(config.get("model", {}))
    model.pop("degradation_conditioned", None)
    model.pop("identity_mode", None)
    model.pop("identity_source_count", None)
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


def method_directory(root: Path, incumbent_root: Path, control_root: Path, method: str, seed: int) -> Path:
    if method == "independent":
        return PROJECT_ROOT / "runs" / "v02_backbone_decision" / "independent" / f"seed{seed}"
    if method == "degradation_001_output_001":
        return incumbent_root / method / f"seed{seed}"
    if method in ("group", "degradation_001"):
        return control_root / f"seed{seed}" / method
    return root / method / f"seed{seed}"


def load_record(root: Path, incumbent_root: Path, control_root: Path, method: str, seed: int, integrity: dict) -> dict | None:
    directory = method_directory(root, incumbent_root, control_root, method, seed)
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
    if fidelity.get("split") != "val" or fidelity.get("sources") != 16 or fidelity.get("restored_views") != 96:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "validation_count_or_split"})
    if int(summary.get("steps", 0)) != 5000:
        integrity["warnings"].append({"method": method, "seed": seed, "kind": "step_count"})
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


def identity_metrics(record: dict) -> dict:
    aggregate = record["fidelity"].get("aggregate", {})
    return {
        "embed_own_clean_top1": aggregate.get("embed_own_clean_top1"),
        "sibling_output_l1": aggregate.get("sibling_output_l1"),
        "cross_source_output_l1": aggregate.get("cross_source_output_l1"),
        "intra_inter_output_ratio": aggregate.get("intra_inter_output_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    root, incumbent_root, control_root, seeds = args.root, args.incumbent_root, args.control_root, tuple(args.seeds)
    root.mkdir(parents=True, exist_ok=True)
    integrity = {"missing_artifacts": [], "warnings": [], "expected_methods": list(ALL_METHODS), "expected_seeds": list(seeds), "expected_sources": 16, "expected_views_per_method_seed": 96, "test_split_used": False}
    records = {}
    for method in ALL_METHODS:
        for seed in seeds:
            record = load_record(root, incumbent_root, control_root, method, seed, integrity)
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
    fairness = {"passed": not differences, "differences": differences, "signatures": signatures}
    (root / "fairness_config_diff.json").write_text(json.dumps(fairness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_seed_rows = []
    for (method, seed), record in sorted(records.items()):
        per_seed_rows.append({"method": method, "seed": seed, "psnr": record["best"]["aggregate"]["psnr"], "ssim": record["best"]["aggregate"]["ssim"], "best_step": record["best"].get("step"), "last_step": record["last"].get("step"), "elapsed_seconds": record["summary"].get("elapsed_seconds"), "parameters": record["summary"].get("parameters"), **{f"id_{key}": identity_metrics(record)[key] for key in identity_metrics(record)}})
    write_csv(root / "per_seed_results.csv", per_seed_rows)

    comparisons = [(f"{method}_vs_{control}", method, control) for method in NEW_METHODS for control in ("independent", "degradation_001_output_001")]
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

    integrity["fairness_differences"] = differences
    integrity["fairness_passed"] = fairness["passed"]
    integrity["missing_artifacts_count"] = len(integrity["missing_artifacts"])
    integrity["warnings_count"] = len(integrity["warnings"])
    integrity["complete"] = not integrity["missing_artifacts"] and not integrity["warnings"] and fairness["passed"] and len(records) == len(ALL_METHODS) * len(seeds)
    (root / "data_integrity_report.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {}
    identity_table = {}
    for method in ALL_METHODS:
        metrics[method] = {}
        identity_table[method] = {}
        for metric in METRICS:
            values = [float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records]
            metrics[method][metric] = {"mean": statistics.mean(values) if values else None, "std": statistics.stdev(values) if len(values) >= 2 else None, "per_seed": {str(seed): float(records[(method, seed)]["best"]["aggregate"][metric]) for seed in seeds if (method, seed) in records}}
        for key in ("embed_own_clean_top1", "sibling_output_l1", "cross_source_output_l1", "intra_inter_output_ratio"):
            values = [identity_metrics(records[(method, seed)])[key] for seed in seeds if (method, seed) in records and identity_metrics(records[(method, seed)])[key] is not None]
            identity_table[method][key] = {"mean": statistics.mean(values) if values else None, "per_seed": {str(seed): identity_metrics(records[(method, seed)])[key] for seed in seeds if (method, seed) in records}}
    (root / "identity_table.json").write_text(json.dumps(identity_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision = {"status": "incomplete", "final_paper_main_model": False, "summary": "数据不完整或公平性检查失败，禁止判定。", "candidates": {}}
    if integrity["complete"]:
        incumbent = "degradation_001_output_001"
        for method in NEW_METHODS:
            comparison = f"{method}_vs_{incumbent}"
            psnr = [row["paired_delta"] for row in paired_rows if row["comparison"] == comparison and row["metric"] == "psnr"]
            embed = identity_table[method]["embed_own_clean_top1"]["per_seed"]
            embed_inc = identity_table[incumbent]["embed_own_clean_top1"]["per_seed"]
            embed_deltas = [embed[str(seed)] - embed_inc[str(seed)] for seed in seeds if str(seed) in embed and str(seed) in embed_inc]
            psnr_ok = statistics.mean(psnr) >= PSNR_FLOOR_DB and all(delta >= PSNR_FLOOR_DB for delta in psnr)
            embed_ok = statistics.mean(embed_deltas) >= EMBED_MIN_GAIN and float(identity_table[method]["embed_own_clean_top1"]["mean"]) >= EMBED_TARGET
            criteria = {
                "psnr_mean_vs_incumbent_db": statistics.mean(psnr),
                "psnr_floor_db": PSNR_FLOOR_DB,
                "psnr_ok": psnr_ok,
                "embed_mean": identity_table[method]["embed_own_clean_top1"]["mean"],
                "embed_incumbent_mean": identity_table[incumbent]["embed_own_clean_top1"]["mean"],
                "embed_gain": statistics.mean(embed_deltas),
                "embed_target": EMBED_TARGET,
                "embed_ok": embed_ok,
            }
            if psnr_ok and embed_ok:
                status = "identity_main_model_pass"
            elif psnr_ok:
                status = "psnr_ok_identity_fail"
            elif embed_ok:
                status = "identity_ok_psnr_fail"
            else:
                status = "reject"
            decision["candidates"][method] = {"status": status, "criteria": criteria, "main_comparison": comparison}
        passed = [name for name, candidate in decision["candidates"].items() if candidate["status"] == "identity_main_model_pass"]
        if passed:
            decision["status"] = "identity_main_model_pass"
            decision["final_paper_main_model"] = True
            decision["summary"] = f"{passed} 同时满足双门槛：PSNR 不低于 incumbent({incumbent} − 0.15 dB) 且 embed 检索 ≥ {EMBED_TARGET} 并提升 ≥ {EMBED_MIN_GAIN}。"
        else:
            decision["status"] = "identity_not_established"
            decision["summary"] = "没有方法同时通过双门槛（PSNR 不降 + 身份检索显著提升）；详见 candidates。"

    report = {
        "metadata": {"analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds), "methods": list(ALL_METHODS), "new_methods": list(NEW_METHODS), "incumbent": "degradation_001_output_001", "bootstrap_unit": "source_id", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED, "ci_level": 0.95, "validation_sources": 16, "validation_views": 96, "test_split_used": False, "gates": {"psnr_floor_db": PSNR_FLOOR_DB, "embed_target": EMBED_TARGET, "embed_min_gain": EMBED_MIN_GAIN}},
        "metrics": metrics, "identity_table": identity_table, "paired_comparisons": paired_rows, "bootstrap_results": bootstrap_rows, "per_degradation_results": per_degradation_rows, "training_config": {"fairness": fairness, "num_workers": 8, "amp": True}, "data_integrity": integrity, "decision": decision,
        "limitations": ["Only fixed validation was used; test remains sealed.", "embed_own_clean_top1 uses the model's own identity embedding (branch mode: detached branch features).", "Identity gain is reported as effect size; statistical significance needs more sources.", "Peak GPU memory was not recorded by train.py."],
    }
    (root / "v04_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(value: float | None) -> str:
        return "None" if value is None else f"{value:.6f}"

    def fmt3(value: float | None) -> str:
        return "None" if value is None else f"{value:.3f}"

    lines = ["# SiblingRestore v0.4 Identity-Line Report", "", f"Decision status: {decision['status']}", "", "## 1. Experimental Setup", "", f"- Methods: {', '.join(ALL_METHODS)}.", f"- Seeds: {', '.join(str(seed) for seed in seeds)}; val 16 sources x 6 degradations; test sealed.", "- Base backbone = degradation_001_output_001 (v0.3 best) + identity signal.", "- V4a classify: 71-class source CE on shared latent; V4b branch: detached identity branch on restored; V4c contrast: gentle pull/push on restored features.", "", "## 2. Fairness Check", "", f"- passed: {fairness['passed']}", f"- differences: {len(differences)}", f"- data integrity complete: {integrity['complete']}", "", "## 3. Main Metrics (best checkpoint, 3 seeds)", "", "| method | PSNR mean | PSNR std | SSIM mean | embed_top1 mean |", "|---|---:|---:|---:|---:|"]
    for method in ALL_METHODS:
        embed = identity_table[method]["embed_own_clean_top1"]["mean"]
        lines.append(f"| {method} | {fmt(metrics[method]['psnr']['mean'])} | {fmt(metrics[method]['psnr']['std'])} | {fmt(metrics[method]['ssim']['mean'])} | {fmt3(embed)} |")
    lines += ["", "## 4. Identity Diagnostics (per seed)", "", "| method | seed | embed_top1 | sibling_l1 | cross_l1 | intra/inter |", "|---|---:|---:|---:|---:|---:|"]
    for method in ALL_METHODS:
        for seed in seeds:
            if (method, seed) not in records:
                continue
            ident = identity_metrics(records[(method, seed)])
            lines.append(f"| {method} | {seed} | {fmt3(ident['embed_own_clean_top1'])} | {fmt3(ident['sibling_output_l1'])} | {fmt3(ident['cross_source_output_l1'])} | {fmt3(ident['intra_inter_output_ratio'])} |")
    lines += ["", "## 5. Paired Bootstrap (aggregate, new methods vs incumbent)", "", "| comparison | metric | mean delta | CI95 lower | CI95 upper | n_sources |", "|---|---|---:|---:|---:|---:|"]
    for row in bootstrap_rows:
        if row["seed"] == "aggregate" and row["metric"] == "psnr":
            lines.append(f"| {row['comparison']} | {row['metric']} | {fmt(row['mean_delta'])} | {fmt(row['ci_lower'])} | {fmt(row['ci_upper'])} | {row['n_sources']} |")
    lines += ["", "## 6. Decision", "", decision["summary"], "", f"- final_paper_main_model: {decision['final_paper_main_model']}", "", "## 7. Candidate Criteria", ""]
    for method, candidate in decision["candidates"].items():
        lines.append(f"### {method} ({candidate['status']})")
        lines.append(f"- psnr_mean_vs_incumbent: {candidate['criteria']['psnr_mean_vs_incumbent_db']}, psnr_ok: {candidate['criteria']['psnr_ok']}")
        lines.append(f"- embed_mean: {candidate['criteria']['embed_mean']} (incumbent {candidate['criteria']['embed_incumbent_mean']}), gain: {candidate['criteria']['embed_gain']}, embed_ok: {candidate['criteria']['embed_ok']}")
    lines += ["", "## 8. Limitations", ""] + [f"- {item}" for item in report["limitations"]] + [""]
    (root / "v04_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "integrity_complete": integrity["complete"], "fairness_passed": fairness["passed"]}, ensure_ascii=False))
    if not integrity["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
