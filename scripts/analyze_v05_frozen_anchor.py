from __future__ import annotations

"""v0.5 frozen-anchor analysis: integrity, fairness, source-level paired
bootstrap over independent frozen-verifier identity metrics, and the dual-gate
main-model decision (PSNR floor + non-circular identity gain with a source-level
CI excluding zero).

Usage: python scripts/analyze_v05_frozen_anchor.py \
       --root runs/v05_frozen_verifier --seeds 13 37 73
"""

import argparse
import csv
import importlib.util
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

_EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_frozen_verifier.py"
_spec = importlib.util.spec_from_file_location("evaluate_frozen_verifier", _EVALUATOR_PATH)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
binary_auc = _evaluator_module.binary_auc
eer = _evaluator_module.eer

CANDIDATES = ("frozen_anchor_001", "frozen_anchor_pcgrad_001")
CONTROLS = ("independent", "group", "degradation_001", "degradation_001_output_001")
ALL_METHODS = CONTROLS + CANDIDATES
SEEDS = (13, 37, 73)
DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")
BOOTSTRAP_ITERATIONS = 10000
PSNR_FLOOR_DB = -0.15
BASE_REQUIRED = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def control_directory(method: str, seed: int) -> Path:
    if method == "independent":
        return PROJECT_ROOT / "runs" / "v02_backbone_decision" / "independent" / f"seed{seed}"
    if method == "degradation_001_output_001":
        return PROJECT_ROOT / "runs" / "v03_degradation" / method / f"seed{seed}"
    return PROJECT_ROOT / "runs" / "ablation_v02_multiseed" / f"seed{seed}" / method


def evaluation_directory(root: Path, method: str, seed: int) -> Path:
    if method in CANDIDATES:
        return root / method / f"seed{seed}"
    return root / "evaluations" / method / f"seed{seed}"


def method_eval_path(root: Path, method: str, seed: int, name: str) -> Path:
    return evaluation_directory(root, method, seed) / name


def per_source_metric(rows: list[dict], key: str) -> dict[str, float]:
    by_source: dict[str, list[float]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(float(row[key]))
    return {source: statistics.mean(values) for source, values in by_source.items()}


def per_source_auc_eer(pair_rows: list[dict], kind: str) -> tuple[dict[str, float], dict[str, float]]:
    auc_by_source: dict[str, float] = {}
    eer_by_source: dict[str, float] = {}
    for source in sorted({row["source_id"] for row in pair_rows}):
        selected = [row for row in pair_rows if row["source_id"] == source and row["query_type"] == kind]
        if not selected:
            continue
        scores = [float(row["score"]) for row in selected]
        labels = [int(row["label"]) for row in selected]
        auc_by_source[source] = binary_auc(scores, labels)
        eer_by_source[source] = eer(scores, labels)
    return auc_by_source, eer_by_source


def delta_per_source(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    common = sorted(set(a) & set(b))
    return {source: a[source] - b[source] for source in common}


def mean_over_seeds(values: list[dict[str, float]]) -> dict[str, float]:
    sources = sorted({source for d in values for source in d})
    return {source: statistics.mean([d[source] for d in values if source in d]) for source in sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    root, seeds = args.root, tuple(args.seeds)
    incumbent = "degradation_001_output_001"
    integrity = {"missing_artifacts": [], "warnings": [], "test_split_used": False}
    records = {}
    for method in ALL_METHODS:
        for seed in seeds:
            eval_dir = evaluation_directory(root, method, seed)
            missing = [name for name in ("frozen_verifier_val.json", "frozen_verifier_val.csv", "frozen_verifier_val_pairs.csv") if not (eval_dir / name).exists()]
            if method in CANDIDATES:
                missing += [name for name in BASE_REQUIRED if not (eval_dir / name).exists()]
            if missing:
                integrity["missing_artifacts"].append({"method": method, "seed": seed, "path": str(eval_dir), "missing": missing})
                continue
            report = read_json(eval_dir / "frozen_verifier_val.json")
            rows = read_csv(eval_dir / "frozen_verifier_val.csv")
            pairs = read_csv(eval_dir / "frozen_verifier_val_pairs.csv")
            for payload in (report,):
                for bad in finite_scalars(payload):
                    integrity["warnings"].append({"method": method, "seed": seed, "kind": "nonfinite", "fields": bad})
            if len(rows) != 96:
                integrity["warnings"].append({"method": method, "seed": seed, "kind": "view_count", "value": len(rows)})
            if len(set((row.get("source_id"), row.get("degradation")) for row in rows)) != 96:
                integrity["warnings"].append({"method": method, "seed": seed, "kind": "duplicate_source_degradation"})
            if report.get("split") != "val":
                integrity["warnings"].append({"method": method, "seed": seed, "kind": "split"})
            if report.get("test_split_used"):
                integrity["test_split_used"] = True
            base = None
            if method in CANDIDATES:
                base = {
                    "summary": read_json(eval_dir / "summary.json"),
                    "resolved_config": read_json(eval_dir / "resolved_config.json"),
                }
                if int(base["summary"].get("steps", 0)) != 5000:
                    integrity["warnings"].append({"method": method, "seed": seed, "kind": "step_count"})
            records[(method, seed)] = {"method": method, "seed": seed, "report": report, "rows": rows, "pairs": pairs, "base": base, "directory": eval_dir}

    evaluator_fingerprints = {str(records[(m, s)]["report"].get("verifier_fingerprint", {}).get("sha256")) for (m, s) in records}
    if len(evaluator_fingerprints) > 1:
        integrity["warnings"].append({"kind": "evaluator_fingerprint_mismatch", "fingerprints": sorted(evaluator_fingerprints)})
    candidate_teacher_hashes = set()
    for method in CANDIDATES:
        for seed in seeds:
            if (method, seed) in records:
                fingerprint = records[(method, seed)]["base"]["resolved_config"].get("frozen_verifier_fingerprint", {}).get("sha256")
                candidate_teacher_hashes.add(str(fingerprint))
    if len(candidate_teacher_hashes) > 1:
        integrity["warnings"].append({"kind": "teacher_fingerprint_mismatch", "fingerprints": sorted(candidate_teacher_hashes)})

    # Per-seed per-source metrics for every method.
    source_metrics: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for (method, seed), record in records.items():
        by_source = per_source_metric(record["rows"], "psnr")
        source_metrics[(method, seed)] = {
            "psnr": by_source,
            "restored_to_clean_top1": per_source_metric(record["rows"], "restored_to_clean_top1"),
            "input_to_clean_top1": per_source_metric(record["rows"], "input_to_clean_top1"),
            "restored_margin": per_source_metric(record["rows"], "restored_margin"),
            "input_margin": per_source_metric(record["rows"], "input_margin"),
            "identity_gain_margin": per_source_metric(record["rows"], "identity_gain_margin"),
        }
        auc, er = per_source_auc_eer(record["pairs"], "restored")
        source_metrics[(method, seed)]["restored_roc_auc"] = auc
        source_metrics[(method, seed)]["restored_eer"] = er

    # Comparisons: candidate vs incumbent, candidate vs independent.
    comparisons = [(f"{candidate}_vs_{control}", candidate, control) for candidate in CANDIDATES for control in ("degradation_001_output_001", "independent")]
    identity_keys = ("restored_margin", "identity_gain_margin", "restored_to_clean_top1", "restored_roc_auc")
    bootstrap_rows, paired_rows = [], []
    per_seed_rows = []
    for (method, seed), record in sorted(records.items()):
        agg = record["report"].get("aggregate", {})
        per_seed_rows.append({"method": method, "seed": seed, "psnr": agg.get("psnr"), "ssim": agg.get("ssim"), "restored_to_clean_top1": agg.get("restored_to_clean_top1"), "input_to_clean_top1": agg.get("input_to_clean_top1"), "restored_margin": agg.get("restored_margin"), "input_margin": agg.get("input_margin"), "identity_gain_margin": agg.get("identity_gain_margin"), "restored_roc_auc": agg.get("restored_roc_auc"), "restored_eer": agg.get("restored_eer")})
    for comparison, model_a, model_b in comparisons:
        for metric in ("psnr",) + identity_keys:
            per_seed_source = []
            for seed in seeds:
                if (model_a, seed) not in records or (model_b, seed) not in records:
                    continue
                deltas = delta_per_source(source_metrics[(model_a, seed)][metric], source_metrics[(model_b, seed)][metric])
                per_seed_source.append(deltas)
                ci = cluster_bootstrap(deltas, BOOTSTRAP_ITERATIONS * comparison.__len__() + metric.__len__() * 31 + seed, BOOTSTRAP_ITERATIONS)
                bootstrap_rows.append({"comparison": comparison, "metric": metric, "seed": seed, "mean_delta": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "n_sources": ci["source_count"], "bootstrap_unit": "source_id"})
                paired_rows.append({"comparison": comparison, "seed": seed, "metric": metric, "model_a": model_a, "model_b": model_b, "paired_delta": statistics.mean(deltas.values()) if deltas else None})
            if per_seed_source:
                aggregate_source = mean_over_seeds(per_seed_source)
                ci = cluster_bootstrap(aggregate_source, BOOTSTRAP_ITERATIONS * comparison.__len__() + metric.__len__() * 31 + 999, BOOTSTRAP_ITERATIONS)
                bootstrap_rows.append({"comparison": comparison, "metric": metric, "seed": "aggregate", "mean_delta": ci["mean_delta"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "n_sources": ci["source_count"], "bootstrap_unit": "source_id"})
    root.mkdir(parents=True, exist_ok=True)
    import scripts  # noqa: F401
    for name, rows in (("per_seed_results.csv", per_seed_rows), ("paired_comparisons.csv", paired_rows), ("bootstrap_results.csv", bootstrap_rows)):
        if not rows:
            continue
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    integrity["complete"] = not integrity["missing_artifacts"] and not integrity["warnings"] and len(records) == len(ALL_METHODS) * len(seeds)
    (root / "data_integrity_report.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision = {"status": "incomplete_or_integrity_fail", "final_paper_main_model": False, "summary": "integrity check failed", "candidates": {}}
    if integrity["complete"]:
        for candidate in CANDIDATES:
            comparison = f"{candidate}_vs_{incumbent}"
            psnr_deltas = [row["paired_delta"] for row in paired_rows if row["comparison"] == comparison and row["metric"] == "psnr"]
            margin_row = [row for row in bootstrap_rows if row["comparison"] == comparison and row["metric"] == "restored_margin" and row["seed"] == "aggregate"]
            gain_row = [row for row in bootstrap_rows if row["comparison"] == comparison and row["metric"] == "identity_gain_margin" and row["seed"] == "aggregate"]
            self_gain = []
            margin_vs_incumbent = []
            for seed in seeds:
                cand = source_metrics[(candidate, seed)]
                inc = source_metrics[(incumbent, seed)]
                margin_vs_incumbent.append(statistics.mean(delta_per_source(cand["restored_margin"], inc["restored_margin"]).values()))
                self_gain.append(statistics.mean(delta_per_source(cand["restored_margin"], cand["input_margin"]).values()))
            margin_ci = margin_row[0] if margin_row else {"ci_lower": None, "ci_upper": None, "mean_delta": None}
            gain_ci = gain_row[0] if gain_row else {"ci_lower": None, "ci_upper": None, "mean_delta": None}
            psnr_ok = bool(psnr_deltas) and statistics.mean(psnr_deltas) >= PSNR_FLOOR_DB and all(d >= PSNR_FLOOR_DB for d in psnr_deltas)
            self_gain_ok = bool(self_gain) and statistics.mean(self_gain) > 0.0
            incumbent_ok = bool(margin_vs_incumbent) and statistics.mean(margin_vs_incumbent) > 0.0
            ci_ok = (margin_ci["ci_lower"] is not None and margin_ci["ci_lower"] > 0.0) or (gain_ci["ci_lower"] is not None and gain_ci["ci_lower"] > 0.0)
            identity_ok = self_gain_ok and incumbent_ok and ci_ok
            criteria = {
                "psnr_mean_vs_incumbent": statistics.mean(psnr_deltas) if psnr_deltas else None,
                "psnr_floor_db": PSNR_FLOOR_DB,
                "psnr_ok": psnr_ok,
                "self_gain_mean_margin": statistics.mean(self_gain) if self_gain else None,
                "self_gain_ok": self_gain_ok,
                "margin_vs_incumbent_mean": statistics.mean(margin_vs_incumbent) if margin_vs_incumbent else None,
                "incumbent_ok": incumbent_ok,
                "margin_ci": margin_ci,
                "identity_gain_ci": gain_ci,
                "ci_ok": ci_ok,
                "identity_ok": identity_ok,
            }
            if psnr_ok and identity_ok:
                status = "validation_main_model_candidate"
            elif psnr_ok:
                status = "psnr_ok_identity_fail"
            elif identity_ok:
                status = "identity_gain_psnr_fail"
            else:
                status = "reject"
            decision["candidates"][candidate] = {"status": status, "criteria": criteria, "main_comparison": comparison}
        passed = [name for name, candidate in decision["candidates"].items() if candidate["status"] == "validation_main_model_candidate"]
        decision["status"] = "validation_main_model_candidate" if passed else "no_validation_main_model"
        decision["summary"] = f"{passed} pass the dual gate (PSNR floor {PSNR_FLOOR_DB} dB + non-circular identity gain with source-level CI excluding zero); final paper status requires sealed test." if passed else "no candidate passes the dual gate."

    report = {
        "metadata": {"analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds), "methods": list(ALL_METHODS), "candidates": list(CANDIDATES), "incumbent": "degradation_001_output_001", "bootstrap_unit": "source_id", "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "ci_level": 0.95, "validation_sources": 16, "validation_views": 96, "test_split_used": False, "gates": {"psnr_floor_db": PSNR_FLOOR_DB}},
        "data_integrity": integrity,
        "decision": decision,
        "paired_comparisons": paired_rows,
        "bootstrap_results": bootstrap_rows,
        "per_seed_results": per_seed_rows,
        "limitations": ["Validation-only; test remains sealed.", "Headline identity metrics come from the independent frozen evaluator.", "Candidate identity evidence requires a source-level CI excluding zero."],
    }
    (root / "v05_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# SiblingRestore v0.5 Frozen-Anchor Report", "", f"Decision status: {decision['status']}", "", f"- seeds: {', '.join(str(seed) for seed in seeds)}; val 16 sources x 6 degradations; test sealed.", f"- candidates: {', '.join(CANDIDATES)}; incumbent: {incumbent}.", "", "## 1. Integrity", "", f"- missing artifacts: {len(integrity['missing_artifacts'])}", f"- warnings: {len(integrity['warnings'])}", f"- complete: {integrity['complete']}", "", "## 2. Per-Seed Summary", "", "| method | seed | psnr | restored_top1 | restored_margin | identity_gain_margin | restored_roc_auc | restored_eer |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_seed_rows:
        lines.append("| {method} | {seed} | {psnr:.4f} | {restored_to_clean_top1:.3f} | {restored_margin:.4f} | {identity_gain_margin:.5f} | {restored_roc_auc:.4f} | {restored_eer:.4f} |".format(**{k: (v if v is not None else float("nan")) for k, v in row.items()}))
    lines += ["", "## 3. Aggregate Bootstrap vs Incumbent (psnr + headline identity)", "", "| comparison | metric | mean delta | CI95 lower | CI95 upper |", "|---|---|---:|---:|---:|"]
    for row in bootstrap_rows:
        if row["seed"] == "aggregate" and row["metric"] in ("psnr", "restored_margin", "identity_gain_margin", "restored_roc_auc"):
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:.6f} | {row['ci_lower']:.6f} | {row['ci_upper']:.6f} |")
    lines += ["", "## 4. Decision", "", decision["summary"], "", f"- final_paper_main_model: {decision['final_paper_main_model']}", ""]
    for candidate, entry in decision["candidates"].items():
        lines.append(f"### {candidate} ({entry['status']})")
        lines.append(f"- psnr_mean_vs_incumbent: {entry['criteria']['psnr_mean_vs_incumbent']}, psnr_ok: {entry['criteria']['psnr_ok']}")
        lines.append(f"- self_gain_mean_margin: {entry['criteria']['self_gain_mean_margin']}, margin_vs_incumbent_mean: {entry['criteria']['margin_vs_incumbent_mean']}")
        lines.append(f"- margin_ci: {entry['criteria']['margin_ci']}, identity_gain_ci: {entry['criteria']['identity_gain_ci']}")
        lines.append(f"- identity_ok: {entry['criteria']['identity_ok']}")
    lines += ["", "## 5. Limitations", ""] + [f"- {item}" for item in report["limitations"]] + [""]
    (root / "v05_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "integrity_complete": integrity["complete"], "candidates": {k: v["status"] for k, v in decision["candidates"].items()}}, ensure_ascii=False))
    if not integrity["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
