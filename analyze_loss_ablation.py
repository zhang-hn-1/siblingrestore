from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS = (
    "group",
    "source_0001",
    "source_0003",
    "source_001",
    "output_001",
    "degradation_001",
)
MULTISEED_EXPERIMENTS = ("group", "source_0003", "degradation_001")
AUXILIARIES = ("source", "output", "degradation")
IGNORED_CONFIG_KEYS = {"output_dir", "mode", "loss_weights", "seed"}
BOOTSTRAP_METHOD = "paired source-level cluster bootstrap"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_scalars(value: object, prefix: str = "") -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[dict] = []
        for key, item in value.items():
            result.extend(finite_scalars(item, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(finite_scalars(item, f"{prefix}[{index}]"))
        return result
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        return [{"field": prefix, "value": value}]
    return []


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def percentile(values: list[float], fraction: float) -> float | None:
    values = sorted(_finite(values))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summary_stats(values: list[float]) -> dict:
    values = _finite(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def sample_mean_std(values: list[float]) -> dict:
    values = _finite(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) >= 2 else None,
    }


def bootstrap_ci(values: list[float], seed: int = 13, rounds: int = 10000) -> list[float | None]:
    values = _finite(values)
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return [means[int(0.025 * (rounds - 1))], means[int(0.975 * (rounds - 1))]]


def cluster_bootstrap(
    deltas: dict[str, float], seed: int, rounds: int = 10000
) -> dict:
    clean = {str(key): float(value) for key, value in deltas.items() if math.isfinite(float(value))}
    source_ids = sorted(clean)
    values = [clean[source_id] for source_id in source_ids]
    if not values:
        return {
            "mean_delta": None,
            "ci_lower": None,
            "ci_upper": None,
            "source_count": 0,
            "bootstrap_method": BOOTSTRAP_METHOD,
            "bootstrap_iterations": rounds,
        }
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sampled = [values[index] for index in (rng.randrange(len(values)) for _ in values)]
        means.append(statistics.mean(sampled))
    means.sort()
    return {
        "mean_delta": statistics.mean(values),
        "ci_lower": means[int(0.025 * (rounds - 1))],
        "ci_upper": means[int(0.975 * (rounds - 1))],
        "source_count": len(values),
        "bootstrap_method": BOOTSTRAP_METHOD,
        "bootstrap_iterations": rounds,
    }


def config_signature(config: dict, validation: dict) -> dict:
    signature = {
        "seed": config.get("seed"),
        "crop_size": config.get("crop_size"),
        "max_steps": config.get("max_steps"),
        "learning_rate": config.get("learning_rate"),
        "scheduler": config.get("scheduler", "cosine"),
        "optimizer": config.get("optimizer", "adamw"),
        "model": config.get("model"),
        "batch_size": config.get("batch_size"),
        "sibling_count": config.get("sibling_count"),
        "degraded_images_per_step": config.get("batch_size", 0) * (
            config.get("sibling_count", 1) if config.get("mode") != "independent" else 1
        ),
        "validation_split": config.get("validation_split", "val"),
        "validation_count": validation.get("count"),
        "validation_interval_steps": config.get("validation_interval_steps"),
    }
    return signature


def fairness_check(
    configs: dict[str, dict], validations: dict[str, dict], experiment_names: tuple[str, ...]
) -> dict:
    signatures = {name: config_signature(config, validations[name]) for name, config in configs.items()}
    reference = signatures[experiment_names[0]]
    differences = []
    for name in experiment_names[1:]:
        for key, expected in reference.items():
            actual = signatures[name].get(key)
            if actual != expected:
                differences.append({"experiment": name, "field": key, "reference": expected, "actual": actual})
    return {"passed": not differences, "differences": differences, "signatures": signatures}


def diagnostics(logs: list[dict]) -> dict:
    result = {}
    for auxiliary in AUXILIARIES:
        cosine_key = f"grad_cos_rec_{auxiliary}"
        aux_norm_key = f"grad_norm_{auxiliary}"
        raw_ratio_key = f"grad_ratio_{auxiliary}_to_rec"
        weighted_norm_key = f"weighted_grad_norm_{auxiliary}"
        weighted_ratio_key = f"weighted_grad_ratio_{auxiliary}_to_rec"
        attempted = [
            row for row in logs
            if row.get(aux_norm_key) is not None or row.get(cosine_key) is not None
        ]
        rec_norms = [
            float(row["grad_norm_rec"])
            for row in attempted
            if row.get("grad_norm_rec") is not None and math.isfinite(float(row["grad_norm_rec"]))
        ]
        near_zero = sum(value < 1e-12 for value in rec_norms)
        valid_norm_rows = [
            row for row in attempted
            if row.get("grad_norm_rec") is not None
            and row.get(aux_norm_key) is not None
            and math.isfinite(float(row["grad_norm_rec"]))
            and math.isfinite(float(row[aux_norm_key]))
            and float(row["grad_norm_rec"]) >= 1e-12
        ]
        raw_ratio = [
            float(row[raw_ratio_key]) for row in valid_norm_rows
            if row.get(raw_ratio_key) is not None and math.isfinite(float(row[raw_ratio_key]))
        ]
        weighted_ratio = [
            float(row[weighted_ratio_key]) for row in valid_norm_rows
            if row.get(weighted_ratio_key) is not None and math.isfinite(float(row[weighted_ratio_key]))
        ]
        raw_norm_pairs = [
            (float(row[aux_norm_key]), float(row["grad_norm_rec"])) for row in valid_norm_rows
        ]
        weighted_norm_pairs = [
            (float(row[weighted_norm_key]), float(row["grad_norm_rec"]))
            for row in valid_norm_rows
            if row.get(weighted_norm_key) is not None and math.isfinite(float(row[weighted_norm_key]))
        ]
        cosines = [
            float(row[cosine_key]) for row in attempted
            if row.get(cosine_key) is not None and math.isfinite(float(row[cosine_key]))
        ]
        result[auxiliary] = {
            "diagnostic_points": len(attempted),
            "diagnostic_point_count": len(attempted),
            "gradient_cosine_mean": statistics.mean(cosines) if cosines else None,
            "gradient_cosine_median": statistics.median(cosines) if cosines else None,
            "cosine_mean": statistics.mean(cosines) if cosines else None,
            "cosine_median": statistics.median(cosines) if cosines else None,
            "negative_fraction": sum(value < 0 for value in cosines) / len(cosines) if cosines else None,
            "cosine_negative_fraction": sum(value < 0 for value in cosines) / len(cosines) if cosines else None,
            "raw_gradient_norm_ratio_mean": statistics.mean(raw_ratio) if raw_ratio else None,
            "weighted_gradient_norm_mean": statistics.mean(
                [float(row[weighted_norm_key]) for row in valid_norm_rows if row.get(weighted_norm_key) is not None]
            ) if valid_norm_rows and any(row.get(weighted_norm_key) is not None for row in valid_norm_rows) else None,
            "weighted_gradient_norm_ratio_mean": statistics.mean(weighted_ratio) if weighted_ratio else None,
            "raw_ratio": summary_stats(raw_ratio),
            "weighted_ratio": summary_stats(weighted_ratio),
            "ratio_of_mean_norms": (
                statistics.mean([pair[0] for pair in raw_norm_pairs]) /
                statistics.mean([pair[1] for pair in raw_norm_pairs])
                if raw_norm_pairs and statistics.mean([pair[1] for pair in raw_norm_pairs]) >= 1e-12 else None
            ),
            "weighted_ratio_of_mean_norms": (
                statistics.mean([pair[0] for pair in weighted_norm_pairs]) /
                statistics.mean([pair[1] for pair in weighted_norm_pairs])
                if weighted_norm_pairs and statistics.mean([pair[1] for pair in weighted_norm_pairs]) >= 1e-12 else None
            ),
            "reconstruction_gradient_norm": {
                "min": min(rec_norms) if rec_norms else None,
                "median": statistics.median(rec_norms) if rec_norms else None,
            },
            "reconstruction_gradient_norm_min": min(rec_norms) if rec_norms else None,
            "reconstruction_gradient_norm_median": statistics.median(rec_norms) if rec_norms else None,
            "near_zero_denominator_count": near_zero,
        }
    return result


def analyze(root: Path, experiment_names: tuple[str, ...] = EXPERIMENTS) -> tuple[dict, bool]:
    if "group" not in experiment_names:
        raise ValueError("experiments must include group as the comparison baseline")
    configs = {name: read_json(root / name / "resolved_config.json") for name in experiment_names}
    validations = {name: read_json(root / name / "best_validation.json") for name in experiment_names}
    fairness = fairness_check(configs, validations, experiment_names)
    experiments = {}
    nonfinite = []
    final_steps = {}
    for name in experiment_names:
        directory = root / name
        logs = read_jsonl(directory / "train_log.jsonl")
        history = read_jsonl(directory / "validation_history.jsonl")
        fidelity_path = directory / "source_fidelity_val.json"
        fidelity = read_json(fidelity_path) if fidelity_path.exists() else {}
        final_step = int(logs[-1]["step"]) if logs else 0
        final_steps[name] = final_step
        for row in logs + history:
            for bad in finite_scalars(row):
                nonfinite.append({"experiment": name, **bad})
        experiments[name] = {
            "config": configs[name],
            "best_validation": validations[name],
            "last_validation": read_json(directory / "last_validation.json"),
            "validation_history": history,
            "source_fidelity": fidelity,
            "diagnostics": diagnostics(logs),
            "convergence": {
                "first_total_loss": logs[0].get("total_loss") if logs else None,
                "last_total_loss": logs[-1].get("total_loss") if logs else None,
                "final_step": final_step,
            },
        }

    comparison_invalid = len(set(final_steps.values())) != 1
    group_validation = validations["group"]
    group_rows = {
        (row.get("source_id"), row.get("degradation")): float(row["psnr"])
        for row in read_csv(root / "group" / "source_fidelity_val.csv")
    }
    comparison = {}
    bootstrap = {}
    for name in experiment_names:
        validation = validations[name]
        comparison[name] = {
            "psnr_delta": validation["aggregate"]["psnr"] - group_validation["aggregate"]["psnr"],
            "ssim_delta": validation["aggregate"]["ssim"] - group_validation["aggregate"]["ssim"],
            "by_degradation": {
                degradation: {
                    "psnr_delta": metrics["psnr"] - group_validation["by_degradation"][degradation]["psnr"],
                    "ssim_delta": metrics["ssim"] - group_validation["by_degradation"][degradation]["ssim"],
                }
                for degradation, metrics in validation["by_degradation"].items()
            },
        }
        deltas = []
        rows = read_csv(root / name / "source_fidelity_val.csv")
        for row in rows:
            key = (row.get("source_id"), row.get("degradation"))
            if key in group_rows:
                deltas.append(float(row["psnr"]) - group_rows[key])
        sources = len({row.get("source_id") for row in rows})
        views = len(rows)
        bootstrap[name] = {
            "status": "ok" if sources >= 16 and views >= 96 else "diagnostic_only_insufficient_sample",
            "sources": sources,
            "views": views,
            "n": len(deltas),
            "mean": statistics.mean(deltas) if deltas else None,
            "ci95": bootstrap_ci(deltas),
        }

    smoke_only = any(step < 500 for step in final_steps.values())
    formal_criteria_check = {
        "steps_5000": all(step == 5000 for step in final_steps.values()),
        "gradient_diagnostic_points": all(
            all(
                experiments[name]["diagnostics"][auxiliary]["diagnostic_points"] >= 10
                for auxiliary in AUXILIARIES
                if any(row.get(f"grad_cos_rec_{auxiliary}") is not None for row in read_jsonl(root / name / "train_log.jsonl"))
            )
            for name in experiment_names
        ),
        "full_validation_16_sources_96_views": all(
            item["source_fidelity"].get("sources") == 16 and item["source_fidelity"].get("restored_views") == 96
            for item in experiments.values()
        ),
        "bootstrap_96_pairs": all(item["n"] == 96 for item in bootstrap.values()),
        "best_and_last_step_recorded": all(
            item["best_validation"].get("step") is not None and item["last_validation"].get("step") is not None
            for item in experiments.values()
        ),
        "raw_and_weighted_ratios_recorded": all(
            all(
                any(row.get(f"grad_ratio_{auxiliary}_to_rec") is not None for row in read_jsonl(root / name / "train_log.jsonl"))
                and any(row.get(f"weighted_grad_ratio_{auxiliary}_to_rec") is not None for row in read_jsonl(root / name / "train_log.jsonl"))
                for auxiliary in AUXILIARIES
                if any(row.get(f"grad_cos_rec_{auxiliary}") is not None for row in read_jsonl(root / name / "train_log.jsonl"))
            )
            for name in experiment_names
        ),
    }
    formal_criteria_check["passed"] = all(formal_criteria_check.values())
    if comparison_invalid:
        status = "comparison_invalid"
        conclusion = "comparison_invalid: final step counts differ; no effect conclusion is emitted."
    elif smoke_only:
        status = "smoke_only_no_effect_claim"
        conclusion = "代码路径与指标生成正常，当前步数不足以判断损失效果。"
    elif not fairness["passed"] or (not smoke_only and not formal_criteria_check["passed"]):
        status = "comparison_invalid"
        conclusion = "comparison_invalid: fairness check failed; no effect conclusion is emitted."
    else:
        status = "formal_analysis_ready"
        conclusion = "formal result interpretation is permitted only after the stated validation criteria pass."
    report = {
        "status": status,
        "comparison_invalid": comparison_invalid or not fairness["passed"],
        "final_steps": final_steps,
        "conclusion": conclusion,
        "fairness_check": fairness,
        "experiments": experiments,
        "comparison_to_group": comparison,
        "bootstrap_psnr_delta_vs_group": bootstrap,
        "data_quality": {"nonfinite_fields": nonfinite},
        "formal_criteria": {
            "required_steps": 5000,
            "required_gradient_diagnostic_points": 10,
            "required_validation_sources": 16,
            "required_validation_views": 96,
            "required_bootstrap_pairs": 96,
        },
        "formal_criteria_check": formal_criteria_check,
    }
    return report, comparison_invalid or not fairness["passed"]


def _load_fidelity(path: Path) -> dict[str, dict[str, dict]]:
    rows = read_csv(path)
    result: dict[str, dict[str, dict]] = {}
    for row in rows:
        source = str(row.get("source_id", ""))
        degradation = str(row.get("degradation", ""))
        if not source or not degradation:
            continue
        if source in result and degradation in result[source]:
            raise ValueError(f"duplicate source/degradation row in {path}: {source}/{degradation}")
        result.setdefault(source, {})[degradation] = row
    return result


def _source_pair_rows(
    model_a: dict[str, dict[str, dict]],
    model_b: dict[str, dict[str, dict]],
    metric: str,
    seed_label: str,
    comparison: str,
    warnings: list[str],
) -> list[dict]:
    rows = []
    common_sources = sorted(set(model_a) & set(model_b))
    for source_id in common_sources:
        a_degradations = set(model_a[source_id])
        b_degradations = set(model_b[source_id])
        if len(a_degradations) != 6 or len(b_degradations) != 6 or a_degradations != b_degradations:
            warnings.append(
                f"incomplete source excluded: seed={seed_label}, comparison={comparison}, source_id={source_id}, "
                f"model_a_degradations={sorted(a_degradations)}, model_b_degradations={sorted(b_degradations)}"
            )
            continue
        a_values = [float(model_a[source_id][degradation][metric]) for degradation in sorted(a_degradations)]
        b_values = [float(model_b[source_id][degradation][metric]) for degradation in sorted(b_degradations)]
        if not all(math.isfinite(value) for value in a_values + b_values):
            warnings.append(f"non-finite source excluded: seed={seed_label}, comparison={comparison}, source_id={source_id}")
            continue
        a_value = statistics.mean(a_values)
        b_value = statistics.mean(b_values)
        rows.append({
            "seed": seed_label,
            "comparison": comparison,
            "source_id": source_id,
            "metric": metric,
            "model_a_value": a_value,
            "model_b_value": b_value,
            "paired_delta": a_value - b_value,
            "n_degradations": 6,
        })
    missing_a = sorted(set(model_b) - set(model_a))
    missing_b = sorted(set(model_a) - set(model_b))
    if missing_a or missing_b:
        warnings.append(f"source coverage mismatch: seed={seed_label}, comparison={comparison}, only_a={missing_a}, only_b={missing_b}")
    return rows


def _per_degradation_rows(
    model_a: dict[str, dict[str, dict]],
    model_b: dict[str, dict[str, dict]],
    metric: str,
    seed_label: str,
    comparison: str,
    rounds: int,
    warnings: list[str],
) -> list[dict]:
    source_rows = []
    common_sources = sorted(set(model_a) & set(model_b))
    degradations = sorted(set().union(*(set(model_a[source]) for source in common_sources), *(set(model_b[source]) for source in common_sources)))
    for degradation in degradations:
        deltas = {}
        a_values = []
        b_values = []
        for source_id in common_sources:
            if degradation not in model_a.get(source_id, {}) or degradation not in model_b.get(source_id, {}):
                warnings.append(f"missing degradation excluded: seed={seed_label}, comparison={comparison}, degradation={degradation}, source_id={source_id}")
                continue
            a_value = float(model_a[source_id][degradation][metric])
            b_value = float(model_b[source_id][degradation][metric])
            if math.isfinite(a_value) and math.isfinite(b_value):
                a_values.append(a_value)
                b_values.append(b_value)
                deltas[source_id] = a_value - b_value
        ci = cluster_bootstrap(deltas, 1000 + rounds, rounds)
        source_rows.append({
            "seed": seed_label,
            "comparison": comparison,
            "degradation": degradation,
            "metric": metric,
            "n_sources": len(deltas),
            "model_a_mean": statistics.mean(a_values) if a_values else None,
            "model_b_mean": statistics.mean(b_values) if b_values else None,
            "paired_delta_mean": statistics.mean(deltas.values()) if deltas else None,
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
        })
    return source_rows


def _mean_std_metric(values: list[float]) -> dict:
    return sample_mean_std(values)


def _multiseed_decision(report: dict) -> dict:
    if report["status"] != "complete":
        return {
            "status": "incomplete",
            "recommendation": "do_not_upgrade",
            "summary": "seed37/seed73结果尚未全部完成，不能生成三seed效果结论。",
        }
    metrics = report["metrics"]
    psnr = {name: metrics[name]["validation_best"]["psnr"] for name in MULTISEED_EXPERIMENTS}
    source_values = psnr["source_0003"]["mean"]
    group_values = psnr["group"]["mean"]
    source_seed_deltas = report["comparisons"]["source_0003_vs_group"]["validation_best"]["psnr"]["per_seed"]
    at_least_two = sum(delta["delta"] > 0 for delta in source_seed_deltas) >= 2
    cluster = report["cluster_bootstrap"]["source_0003_vs_group"]["psnr"]["aggregate"]
    clearly_negative = bool(cluster["ci_upper"] is not None and cluster["ci_upper"] < 0 and cluster["mean_delta"] < -0.1)
    source_vs_deg = report["cluster_bootstrap"]["source_0003_vs_degradation_001"]["psnr"]["aggregate"]
    no_independent_source_evidence = (
        abs(float(source_vs_deg["mean_delta"] or 0.0)) < 0.1
        and source_vs_deg["ci_lower"] is not None
        and source_vs_deg["ci_lower"] <= 0 <= source_vs_deg["ci_upper"]
        and all(float(item.get("own_clean_top1", 0.0)) >= 0.99 for item in report["source_fidelity"]["per_seed"].get("source_0003", []))
        and all(float(item.get("own_clean_top1", 0.0)) >= 0.99 for item in report["source_fidelity"]["per_seed"].get("degradation_001", []))
    )
    per_deg = report["cluster_bootstrap"]["source_0003_vs_group"]["psnr"]["per_degradation_aggregate"]
    stable_large_drop = any(item["ci_upper"] is not None and item["ci_upper"] < -0.3 for item in per_deg.values())
    if no_independent_source_evidence:
        return {
            "status": "insufficient_evidence_source_mechanism",
            "recommendation": "add_unsaturated_structure_or_inspection_metrics",
            "summary": "source_0003与degradation_001无明显差异且own-clean top-1饱和，当前证据不足以证明SiblingRestore的source sibling机制具有独立收益。",
        }
    passed = source_values >= group_values and at_least_two and not clearly_negative and source_values >= psnr["degradation_001"]["mean"] and not stable_large_drop
    return {
        "status": "upgrade_candidate" if passed else "not_yet_upgrade",
        "recommendation": "upgrade_source_0003_to_main_model" if passed else "keep_group_as_primary_pending_more_evidence",
        "summary": "source_0003满足三seed主模型判定规则。" if passed else "source_0003尚未同时满足三seed主模型判定规则。",
        "criteria": {
            "mean_psnr_not_below_group": source_values >= group_values,
            "at_least_two_of_three_seeds_above_group": at_least_two,
            "cluster_bootstrap_not_clearly_negative": not clearly_negative,
            "source_0003_not_below_degradation_001": source_values >= psnr["degradation_001"]["mean"],
            "no_stable_large_per_degradation_drop": not stable_large_drop,
        },
    }


def analyze_multiseed(root: Path, seeds: tuple[int, ...], rounds: int = 10000) -> tuple[dict, bool]:
    warnings: list[str] = []
    missing: list[str] = []
    records: dict[tuple[int, str], dict] = {}
    for seed in seeds:
        for experiment in MULTISEED_EXPERIMENTS:
            directory = root / f"seed{seed}" / experiment
            required = ("resolved_config.json", "best_validation.json", "last_validation.json", "train_log.jsonl", "validation_history.jsonl", "source_fidelity_val.json", "source_fidelity_val.csv")
            absent = [name for name in required if not (directory / name).exists()]
            if absent:
                missing.append(f"seed{seed}/{experiment}: missing {absent}")
                continue
            config = read_json(directory / "resolved_config.json")
            best = read_json(directory / "best_validation.json")
            last = read_json(directory / "last_validation.json")
            logs = read_jsonl(directory / "train_log.jsonl")
            history = read_jsonl(directory / "validation_history.jsonl")
            fidelity = read_json(directory / "source_fidelity_val.json")
            records[(seed, experiment)] = {
                "config": config,
                "best": best,
                "last": last,
                "logs": logs,
                "history": history,
                "fidelity": fidelity,
                "fidelity_rows": _load_fidelity(directory / "source_fidelity_val.csv"),
            }
            for row in logs + history + [best, last, fidelity]:
                for bad in finite_scalars(row):
                    warnings.append(f"non-finite seed{seed}/{experiment}: {bad}")

    def seed_record(seed: int, experiment: str) -> dict:
        return records[(seed, experiment)]

    config_fairness = {}
    for seed in seeds:
        available = [experiment for experiment in MULTISEED_EXPERIMENTS if (seed, experiment) in records]
        if available:
            configs = {experiment: records[(seed, experiment)]["config"] for experiment in available}
            validations = {experiment: records[(seed, experiment)]["best"] for experiment in available}
            config_fairness[f"seed{seed}"] = fairness_check(configs, validations, tuple(available)) if "group" in available else {"passed": False, "differences": [{"reason": "group missing"}]}
    if "seed13" in config_fairness:
        reference_seed_signature = config_fairness["seed13"]["signatures"].get("group", {})
        for seed_label, item in config_fairness.items():
            for experiment, signature in item.get("signatures", {}).items():
                for key, expected in reference_seed_signature.items():
                    actual = signature.get(key)
                    if key == "seed":
                        continue
                    if actual != expected:
                        item.setdefault("differences", []).append({"experiment": f"{seed_label}/{experiment}", "field": key, "reference": expected, "actual": actual})
            item["passed"] = not item.get("differences")

    metric_names = ("psnr", "ssim")
    metrics = {}
    for experiment in MULTISEED_EXPERIMENTS:
        validation_best = {}
        validation_by_degradation = {}
        fidelity_metrics = {}
        for metric in metric_names:
            values = [float(records[(seed, experiment)]["best"]["aggregate"][metric]) for seed in seeds if (seed, experiment) in records]
            validation_best[metric] = _mean_std_metric(values)
            validation_best[metric]["per_seed"] = {str(seed): float(records[(seed, experiment)]["best"]["aggregate"][metric]) for seed in seeds if (seed, experiment) in records}
        degradations = sorted(set().union(*(set(records[(seed, experiment)]["best"].get("by_degradation", {})) for seed in seeds if (seed, experiment) in records)))
        for degradation in degradations:
            validation_by_degradation[degradation] = {}
            for metric in metric_names:
                values = [float(records[(seed, experiment)]["best"]["by_degradation"][degradation][metric]) for seed in seeds if (seed, experiment) in records]
                validation_by_degradation[degradation][metric] = _mean_std_metric(values)
        for metric in ("psnr", "ssim", "own_clean_top1", "own_clean_same_class_top1", "sibling_output_l1"):
            values = [float(records[(seed, experiment)]["fidelity"]["aggregate"][metric]) for seed in seeds if (seed, experiment) in records]
            fidelity_metrics[metric] = _mean_std_metric(values)
            fidelity_metrics[metric]["per_seed"] = {str(seed): float(records[(seed, experiment)]["fidelity"]["aggregate"][metric]) for seed in seeds if (seed, experiment) in records}
        metrics[experiment] = {"validation_best": validation_best, "validation_by_degradation": validation_by_degradation, "source_fidelity": fidelity_metrics}

    comparison_pairs = (
        ("source_0003_vs_group", "source_0003", "group"),
        ("degradation_001_vs_group", "degradation_001", "group"),
        ("source_0003_vs_degradation_001", "source_0003", "degradation_001"),
    )
    per_source_rows: list[dict] = []
    per_degradation_rows: list[dict] = []
    cluster_report: dict = {}
    comparisons: dict = {}
    for comparison, model_a_name, model_b_name in comparison_pairs:
        comparisons[comparison] = {"validation_best": {}}
        cluster_report[comparison] = {}
        for metric in metric_names:
            per_seed_delta = []
            per_seed_cluster = {}
            aggregate_source_deltas: dict[str, list[float]] = {}
            aggregate_deg_values: dict[str, dict[str, dict[str, list[float]]]] = {}
            for seed in seeds:
                if (seed, model_a_name) not in records or (seed, model_b_name) not in records:
                    continue
                a = records[(seed, model_a_name)]["fidelity_rows"]
                b = records[(seed, model_b_name)]["fidelity_rows"]
                source_rows = _source_pair_rows(a, b, metric, str(seed), comparison, warnings)
                per_source_rows.extend(source_rows)
                for row in source_rows:
                    aggregate_source_deltas.setdefault(row["source_id"], []).append(float(row["paired_delta"]))
                per_seed_cluster[str(seed)] = cluster_bootstrap({row["source_id"]: row["paired_delta"] for row in source_rows}, seed * 1000 + len(metric), rounds)
                for row in _per_degradation_rows(a, b, metric, str(seed), comparison, rounds, warnings):
                    per_degradation_rows.append(row)
                for source_id in sorted(set(a) & set(b)):
                    for degradation in sorted(set(a[source_id]) & set(b[source_id])):
                        a_value = float(a[source_id][degradation][metric])
                        b_value = float(b[source_id][degradation][metric])
                        if math.isfinite(a_value) and math.isfinite(b_value):
                            aggregate_deg_values.setdefault(degradation, {}).setdefault(source_id, {"a": [], "b": []})["a"].append(a_value)
                            aggregate_deg_values[degradation][source_id]["b"].append(b_value)
                a_val = float(records[(seed, model_a_name)]["best"]["aggregate"][metric])
                b_val = float(records[(seed, model_b_name)]["best"]["aggregate"][metric])
                per_seed_delta.append({"seed": seed, "model_a": model_a_name, "model_b": model_b_name, "delta": a_val - b_val})
            aggregate_deltas = {source_id: statistics.mean(values) for source_id, values in aggregate_source_deltas.items() if values}
            aggregate_ci = cluster_bootstrap(aggregate_deltas, 900000 + len(comparison) + len(metric), rounds)
            for source_id, delta in aggregate_deltas.items():
                matching = [row for row in per_source_rows if row["comparison"] == comparison and row["metric"] == metric and row["seed"] != "aggregate" and row["source_id"] == source_id]
                a_values = [float(row["model_a_value"]) for row in matching]
                b_values = [float(row["model_b_value"]) for row in matching]
                per_source_rows.append({"seed": "aggregate", "comparison": comparison, "source_id": source_id, "metric": metric, "model_a_value": statistics.mean(a_values), "model_b_value": statistics.mean(b_values), "paired_delta": delta, "n_degradations": 6})
            aggregate_deg = {}
            for degradation, by_source in aggregate_deg_values.items():
                source_a = {source_id: statistics.mean(values["a"]) for source_id, values in by_source.items() if values["a"] and values["b"]}
                source_b = {source_id: statistics.mean(values["b"]) for source_id, values in by_source.items() if values["a"] and values["b"]}
                deltas = {source_id: source_a[source_id] - source_b[source_id] for source_id in source_a}
                aggregate_deg[degradation] = cluster_bootstrap(deltas, 910000 + len(degradation), rounds)
                per_degradation_rows.append({"seed": "aggregate", "comparison": comparison, "degradation": degradation, "metric": metric, "n_sources": len(deltas), "model_a_mean": statistics.mean(source_a.values()) if source_a else None, "model_b_mean": statistics.mean(source_b.values()) if source_b else None, "paired_delta_mean": aggregate_deg[degradation]["mean_delta"], "ci_lower": aggregate_deg[degradation]["ci_lower"], "ci_upper": aggregate_deg[degradation]["ci_upper"]})
            cluster_report[comparison][metric] = {"per_seed": per_seed_cluster, "aggregate": aggregate_ci, "per_degradation_aggregate": aggregate_deg}
            comparisons[comparison]["validation_best"][metric] = {"per_seed": per_seed_delta, "mean_delta": statistics.mean([item["delta"] for item in per_seed_delta]) if per_seed_delta else None, "std_delta": statistics.stdev([item["delta"] for item in per_seed_delta]) if len(per_seed_delta) >= 2 else None}

    gradient_report = {}
    for experiment in MULTISEED_EXPERIMENTS:
        per_seed = {}
        combined = []
        for seed in seeds:
            if (seed, experiment) in records:
                logs = records[(seed, experiment)]["logs"]
                per_seed[str(seed)] = diagnostics(logs)
                combined.extend(logs)
        gradient_report[experiment] = {"per_seed": per_seed, "aggregate": diagnostics(combined)}

    source_fidelity_report = {"primary_metrics_note": "own-clean top1 is appendix diagnostic only; it is not primary success evidence.", "per_seed": {experiment: [dict(records[(seed, experiment)]["fidelity"]["aggregate"], seed=seed) for seed in seeds if (seed, experiment) in records] for experiment in MULTISEED_EXPERIMENTS}}
    all_complete = not missing and all((seed, experiment) in records for seed in seeds for experiment in MULTISEED_EXPERIMENTS)
    fairness_passed = all(item.get("passed", False) for item in config_fairness.values()) and len(config_fairness) == len(seeds)
    data_integrity = {
        "complete": all_complete,
        "missing_artifacts": missing,
        "warnings": warnings,
        "expected_seeds": list(seeds),
        "expected_experiments": list(MULTISEED_EXPERIMENTS),
        "expected_source_count": 16,
        "expected_degradation_count": 6,
        "expected_view_count": 96,
        "fairness_passed": fairness_passed,
        "nonfinite_count": sum(1 for warning in warnings if "non-finite" in warning),
    }
    status = "complete" if all_complete and fairness_passed and not data_integrity["nonfinite_count"] else "incomplete"
    report = {
        "metadata": {
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit_hash": None,
            "git_status": "repository is not a git worktree",
            "seeds": list(seeds),
            "experiments": list(MULTISEED_EXPERIMENTS),
            "bootstrap_method": BOOTSTRAP_METHOD,
            "bootstrap_iterations": rounds,
            "ci_level": 0.95,
            "source_count": 16,
            "degradation_count": 6,
        },
        "status": status,
        "training_config": {"fairness_by_seed": config_fairness},
        "seeds": list(seeds),
        "experiments": list(MULTISEED_EXPERIMENTS),
        "metrics": metrics,
        "comparisons": comparisons,
        "cluster_bootstrap": cluster_report,
        "per_degradation": {"csv": "per_degradation_results.csv", "bootstrap_unit": "source_id", "n_sources": 16},
        "gradient_diagnostics": gradient_report,
        "source_fidelity": source_fidelity_report,
        "decision": {"status": "incomplete", "summary": "训练结果不完整，暂不输出效果结论。"},
        "data_integrity": data_integrity,
    }
    if status == "complete":
        report["decision"] = _multiseed_decision(report)
    return report, status != "complete"


def to_markdown(report: dict) -> str:
    lines = [
        "# SiblingRestore v0.2 Loss Ablation Report",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Conclusion: {report['conclusion']}",
        "",
        "## Fairness check",
        "",
        f"- passed: `{report['fairness_check']['passed']}`",
        f"- differences: `{len(report['fairness_check']['differences'])}`",
        "",
        "## Metrics",
        "",
        "| experiment | PSNR | SSIM | ΔPSNR vs group | raw source ratio | weighted source ratio | raw output ratio | weighted output ratio | raw degradation ratio | weighted degradation ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["experiments"].items():
        overall = item["best_validation"]["aggregate"]
        diag = item["diagnostics"]
        comparison = report["comparison_to_group"][name]
        lines.append(
            f"| {name} | {overall['psnr']:.6f} | {overall['ssim']:.6f} | {comparison['psnr_delta']:.6f} | "
            f"{diag['source']['raw_gradient_norm_ratio_mean']} | {diag['source']['weighted_gradient_norm_ratio_mean']} | "
            f"{diag['output']['raw_gradient_norm_ratio_mean']} | {diag['output']['weighted_gradient_norm_ratio_mean']} | "
            f"{diag['degradation']['raw_gradient_norm_ratio_mean']} | {diag['degradation']['weighted_gradient_norm_ratio_mean']} |"
        )
    lines += ["", "## Bootstrap", ""]
    for name, item in report["bootstrap_psnr_delta_vs_group"].items():
        lines.append(f"- {name}: `{item['status']}`, n={item['n']}, CI95={item['ci95']}")
    lines += ["", "## Data quality", "", f"Non-finite fields: {len(report['data_quality']['nonfinite_fields'])}", ""]
    return "\n".join(lines) + "\n"


def multiseed_markdown(report: dict) -> str:
    lines = [
        "# SiblingRestore v0.2 Multiseed Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## 1. Experimental Setup",
        "",
        "- seeds: " + ", ".join(str(seed) for seed in report["seeds"]),
        "- experiments: " + ", ".join(report["experiments"]),
        "- bootstrap unit = `source_id`; n_sources = 16; each source's 6 degradations are resampled as one cluster.",
        f"- bootstrap method: `{report['metadata']['bootstrap_method']}`, iterations={report['metadata']['bootstrap_iterations']}, CI=95%.",
        "- own-clean top-1 is an appendix diagnostic, not primary model evidence.",
        "",
        "## 2. Three-Seed Main Metrics",
        "",
        "| experiment | PSNR mean ± sample std | SSIM mean ± sample std |",
        "|---|---:|---:|",
    ]
    for experiment in report["experiments"]:
        p = report["metrics"][experiment]["validation_best"]["psnr"]
        s = report["metrics"][experiment]["validation_best"]["ssim"]
        lines.append(f"| {experiment} | {p['mean']} ± {p['std']} | {s['mean']} ± {s['std']} |")
    lines += ["", "## 3. Per-Seed Paired Deltas", ""]
    for comparison, item in report["comparisons"].items():
        lines.append(f"### {comparison}")
        for metric, data in item["validation_best"].items():
            values = ", ".join(f"seed{entry['seed']}={entry['delta']:.6f}" for entry in data["per_seed"])
            lines.append(f"- {metric}: {values}; mean={data['mean_delta']}, std={data['std_delta']}")
    lines += ["", "## 4. Source-Level Cluster Bootstrap", ""]
    for comparison, metrics in report["cluster_bootstrap"].items():
        for metric, item in metrics.items():
            aggregate = item["aggregate"]
            lines.append(f"- {comparison} / {metric}: mean delta={aggregate['mean_delta']}, CI95=[{aggregate['ci_lower']}, {aggregate['ci_upper']}], n_sources={aggregate['source_count']}")
    lines += ["", "## 5. Per-Degradation Analysis", "", "See `per_degradation_results.csv`; each row records n_sources and a source-level paired CI.", "", "## 6. Source Fidelity Diagnostics", "", report["source_fidelity"]["primary_metrics_note"], "", "## 7. Gradient Diagnostics", "", "Raw and weighted ratio distributions include mean, median, p10, p25, p75, p90, and ratio_of_mean_norms. Near-zero reconstruction denominators are excluded from ratios and retained as null/diagnostic counts.", "", "## 8. Data Integrity Checks", "", f"- complete: `{report['data_integrity']['complete']}`", f"- fairness_passed: `{report['data_integrity']['fairness_passed']}`", f"- missing artifacts: `{len(report['data_integrity']['missing_artifacts'])}`", f"- warnings: `{len(report['data_integrity']['warnings'])}`", "", "## 9. Main-Model Decision", "", report["decision"].get("summary", "")]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--fairness-output", type=Path)
    parser.add_argument("--multiseed-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73])
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    args = parser.parse_args()
    if args.multiseed_root is not None:
        output_root = args.multiseed_root
        report, invalid = analyze_multiseed(output_root, tuple(args.seeds), args.bootstrap_rounds)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "multiseed_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_root / "multiseed_report.md").write_text(multiseed_markdown(report), encoding="utf-8")
        per_source_rows = []
        per_degradation_rows = []
        # Reconstruct CSV rows from the same persisted records without changing report semantics.
        for comparison, model_a_name, model_b_name in (("source_0003_vs_group", "source_0003", "group"), ("degradation_001_vs_group", "degradation_001", "group"), ("source_0003_vs_degradation_001", "source_0003", "degradation_001")):
            for seed in tuple(args.seeds):
                a_path = output_root / f"seed{seed}" / model_a_name / "source_fidelity_val.csv"
                b_path = output_root / f"seed{seed}" / model_b_name / "source_fidelity_val.csv"
                if not a_path.exists() or not b_path.exists():
                    continue
                a = _load_fidelity(a_path)
                b = _load_fidelity(b_path)
                warnings: list[str] = []
                for metric in ("psnr", "ssim"):
                    per_source_rows.extend(_source_pair_rows(a, b, metric, str(seed), comparison, warnings))
                    per_degradation_rows.extend(_per_degradation_rows(a, b, metric, str(seed), comparison, args.bootstrap_rounds, warnings))
        source_groups = {}
        for row in per_source_rows:
            source_groups.setdefault((row["comparison"], row["source_id"], row["metric"]), []).append(row)
        for (comparison, source_id, metric), rows in source_groups.items():
            per_source_rows.append({
                "seed": "aggregate",
                "comparison": comparison,
                "source_id": source_id,
                "metric": metric,
                "model_a_value": sum(float(row["model_a_value"]) for row in rows) / len(rows),
                "model_b_value": sum(float(row["model_b_value"]) for row in rows) / len(rows),
                "paired_delta": sum(float(row["paired_delta"]) for row in rows) / len(rows),
                "n_degradations": 6,
            })
        degradation_groups = {}
        for row in per_degradation_rows:
            degradation_groups.setdefault((row["comparison"], row["degradation"], row["metric"]), []).append(row)
        for (comparison, degradation, metric), rows in degradation_groups.items():
            if not rows:
                continue
            aggregate_ci = report["cluster_bootstrap"].get(comparison, {}).get(metric, {}).get("per_degradation_aggregate", {}).get(degradation, {})
            per_degradation_rows.append({
                "seed": "aggregate",
                "comparison": comparison,
                "degradation": degradation,
                "metric": metric,
                "n_sources": aggregate_ci.get("source_count", min(int(row["n_sources"]) for row in rows)),
                "model_a_mean": sum(float(row["model_a_mean"]) for row in rows) / len(rows),
                "model_b_mean": sum(float(row["model_b_mean"]) for row in rows) / len(rows),
                "paired_delta_mean": aggregate_ci.get("mean_delta"),
                "ci_lower": aggregate_ci.get("ci_lower"),
                "ci_upper": aggregate_ci.get("ci_upper"),
            })
        write_csv(output_root / "per_source_paired_results.csv", per_source_rows, ["seed", "comparison", "source_id", "metric", "model_a_value", "model_b_value", "paired_delta", "n_degradations"])
        write_csv(output_root / "per_degradation_results.csv", per_degradation_rows, ["seed", "comparison", "degradation", "metric", "n_sources", "model_a_mean", "model_b_mean", "paired_delta_mean", "ci_lower", "ci_upper"])
        print(json.dumps({"status": report["status"], "missing": len(report["data_integrity"]["missing_artifacts"]), "warnings": len(report["data_integrity"]["warnings"])}, ensure_ascii=False))
        if invalid:
            raise SystemExit(2)
        return
    if args.root is None:
        parser.error("--root is required unless --multiseed-root is provided")
    report, invalid = analyze(args.root, tuple(args.experiments))
    json_output = args.json_output or args.root / "ablation_report.json"
    markdown_output = args.markdown_output or args.root / "ablation_report.md"
    fairness_output = args.fairness_output or args.root / "fairness_check.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(to_markdown(report), encoding="utf-8")
    fairness_output.write_text(json.dumps(report["fairness_check"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "fairness_passed": report["fairness_check"]["passed"], "nonfinite": len(report["data_quality"]["nonfinite_fields"])}, ensure_ascii=False))
    if invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
