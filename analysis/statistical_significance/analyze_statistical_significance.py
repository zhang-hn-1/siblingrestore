"""Generate Table 7 from source-level paired bootstrap on official test outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.identity_preservation.analyze_identity_preservation import (  # noqa: E402
    DEGRADATIONS,
    EXPECTED_VERIFIER_SHA,
    OFFICIAL_VERIFIER,
    load_groups,
    load_method_scores,
    read_json,
    result_path,
    sha256_file,
)
from analysis.statistical_significance.core import (  # noqa: E402
    HIGHER_IS_BETTER,
    benjamini_hochberg,
    bootstrap_mean_distribution,
    centered_bootstrap_pvalue,
    gallery_bootstrap_distribution,
    gallery_metric,
    improvement,
    source_level_mean,
)

METHODS = ("Ours", "Restormer", "DehazeFormer", "PromptIR")
BASELINES = ("Restormer", "DehazeFormer", "PromptIR")
METRICS = ("PSNR", "SSIM", "LPIPS", "Top1", "AUC", "EER", "Cosine", "Margin")
NATURAL_METRICS = ("PSNR", "SSIM", "LPIPS", "Top1", "Cosine", "Margin")
MAIN_TABLE_METRICS = ("PSNR", "LPIPS", "Top1", "Margin")
REPETITIONS = 10000
SEED = 13
CONFIDENCE = 0.95
TOLERANCES = {"PSNR": 0.005, "SSIM": 0.0005, "LPIPS": 0.0005,
              "Top1": 0.002, "AUC": 0.002, "EER": 0.002}
QUALITY_KEYS = ("psnr", "ssim", "lpips", "restored_to_clean_top1")


def _method_prefix(method: str) -> str:
    if method == "Ours":
        return "Ours"
    return method


def load_quality(method: str, source_ids: Sequence[str], missing: List[dict]) -> Dict[Tuple[str, str], dict]:
    path = result_path(_method_prefix(method), ".test.csv")
    rows: Dict[Tuple[str, str], dict] = {}
    if not path.exists():
        for sid in source_ids:
            for degradation in DEGRADATIONS:
                missing.append({"method": method, "source_id": sid, "degradation": degradation,
                                "metric": "PSNR/SSIM/LPIPS/Top1", "reason": "missing official test CSV"})
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = (record.get("source_id", ""), record.get("degradation", ""))
            if key in rows:
                missing.append({"method": method, "source_id": key[0], "degradation": key[1],
                                "metric": "all quality metrics", "reason": "duplicate source/degradation row"})
            rows[key] = record
    expected = {(sid, d) for sid in source_ids for d in DEGRADATIONS}
    for sid, degradation in sorted(expected - set(rows)):
        missing.append({"method": method, "source_id": sid, "degradation": degradation,
                        "metric": "PSNR/SSIM/LPIPS/Top1", "reason": "missing official test row"})
    for key, record in rows.items():
        if key not in expected:
            missing.append({"method": method, "source_id": key[0], "degradation": key[1],
                            "metric": "all quality metrics", "reason": "row not in official test gallery"})
            continue
        for field in QUALITY_KEYS:
            try:
                value = float(record[field])
            except (KeyError, TypeError, ValueError):
                missing.append({"method": method, "source_id": key[0], "degradation": key[1],
                                "metric": field, "reason": "missing or non-numeric value"})
                continue
            if not math.isfinite(value):
                missing.append({"method": method, "source_id": key[0], "degradation": key[1],
                                "metric": field, "reason": "non-finite value"})
    return rows


def make_source_metric_values(method: str, quality: Dict[Tuple[str, str], dict],
                              score_records: Dict[Tuple[str, str], dict],
                              source_ids: Sequence[str], missing: List[dict]) -> Dict[str, np.ndarray]:
    values: Dict[str, List[float]] = {metric: [] for metric in NATURAL_METRICS}
    index_of = {sid: index for index, sid in enumerate(source_ids)}
    for sid in source_ids:
        per_view = {metric: [] for metric in NATURAL_METRICS}
        own = index_of[sid]
        for degradation in DEGRADATIONS:
            key = (sid, degradation)
            if key not in quality or key not in score_records:
                continue
            record = quality[key]
            scores = score_records[key]["restored_scores"]
            wrong = np.delete(scores, own)
            per_view["PSNR"].append(float(record["psnr"]))
            per_view["SSIM"].append(float(record["ssim"]))
            per_view["LPIPS"].append(float(record["lpips"]))
            per_view["Top1"].append(float(int(np.argmax(scores) == own)))
            per_view["Cosine"].append(float(scores[own]))
            per_view["Margin"].append(float(scores[own] - np.max(wrong)))
        for metric in NATURAL_METRICS:
            if len(per_view[metric]) != len(DEGRADATIONS):
                missing.append({"method": method, "source_id": sid, "degradation": "<source>",
                                "metric": metric, "reason": "source does not have all seven views"})
            else:
                values[metric].append(float(np.mean(per_view[metric])))
    return {metric: np.asarray(array, dtype=np.float64) for metric, array in values.items()}


def make_per_view_metric_values(quality: Dict[Tuple[str, str], dict],
                                score_records: Dict[Tuple[str, str], dict],
                                source_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Return the official natural metric values in source/degradation order."""
    values: Dict[str, List[float]] = {metric: [] for metric in NATURAL_METRICS}
    index_of = {sid: index for index, sid in enumerate(source_ids)}
    for sid in source_ids:
        own = index_of[sid]
        for degradation in DEGRADATIONS:
            key = (sid, degradation)
            if key not in quality or key not in score_records:
                continue
            record = quality[key]
            scores = score_records[key]["restored_scores"]
            values["PSNR"].append(float(record["psnr"]))
            values["SSIM"].append(float(record["ssim"]))
            values["LPIPS"].append(float(record["lpips"]))
            values["Top1"].append(float(int(np.argmax(scores) == own)))
            values["Cosine"].append(float(scores[own]))
            values["Margin"].append(float(scores[own] - np.max(np.delete(scores, own))))
    return {metric: np.asarray(array, dtype=np.float64) for metric, array in values.items()}


def score_blocks(score_records: Dict[Tuple[str, str], dict], source_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([
        [score_records[(sid, degradation)]["restored_scores"] for degradation in DEGRADATIONS]
        for sid in source_ids
    ], dtype=np.float64)


def aggregate_sanity(method: str, quality: Dict[Tuple[str, str], dict], blocks: np.ndarray,
                     source_ids: Sequence[str], metadata: dict) -> dict:
    means = {"PSNR": float(np.mean([float(row["psnr"]) for row in quality.values()])),
             "SSIM": float(np.mean([float(row["ssim"]) for row in quality.values()])),
             "LPIPS": float(np.mean([float(row["lpips"]) for row in quality.values()])),
             "Top1": float(np.mean([int(np.argmax(blocks[i, j]) == i)
                                    for i in range(len(source_ids)) for j in range(len(DEGRADATIONS))]))}
    means["AUC"] = gallery_metric(blocks, list(range(len(source_ids))), "AUC")
    means["EER"] = gallery_metric(blocks, list(range(len(source_ids))), "EER")
    aggregate = metadata.get("aggregate", {})
    official = {"PSNR": aggregate.get("psnr"), "SSIM": aggregate.get("ssim"),
                "LPIPS": aggregate.get("lpips"), "Top1": aggregate.get("restored_to_clean_top1"),
                "AUC": aggregate.get("restored_roc_auc"), "EER": aggregate.get("restored_eer")}
    differences = {metric: abs(means[metric] - official[metric]) for metric in means
                   if official[metric] is not None}
    passed = all(differences[metric] <= TOLERANCES[metric] for metric in differences)
    return {"method": method, "recomputed": means, "official": official,
            "absolute_difference": differences, "passed": passed, "tolerances": TOLERANCES}


def protocol_issues(metadata_by_method: Dict[str, dict], audit: dict, source_ids: Sequence[str]) -> List[str]:
    issues = []
    if not OFFICIAL_VERIFIER.exists():
        issues.append("official verifier checkpoint missing")
    elif sha256_file(OFFICIAL_VERIFIER) != EXPECTED_VERIFIER_SHA:
        issues.append("official verifier checkpoint sha256 differs")
    expected_groups = audit.get("groups_sha256")
    expected_index = audit.get("index_sha256")
    for method, metadata in metadata_by_method.items():
        fingerprint = metadata.get("verifier_fingerprint", {})
        package = metadata.get("data_pkg_fingerprint", {})
        if fingerprint.get("sha256") != EXPECTED_VERIFIER_SHA:
            issues.append("%s uses a different verifier" % method)
        if fingerprint.get("embedding_dim") != 128:
            issues.append("%s embedding dimension differs" % method)
        if metadata.get("split") != "test" or metadata.get("sources") != len(source_ids) or metadata.get("restored_views") != len(source_ids) * len(DEGRADATIONS):
            issues.append("%s split/source/view counts differ" % method)
        if package.get("audit_groups_sha256") != expected_groups or package.get("audit_index_sha256") != expected_index:
            issues.append("%s data package fingerprint differs" % method)
    return issues


def ci(values: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def add_stats_row(rows: List[dict], comparison: str, metric: str, ours_point: float,
                  baseline_point: float, distribution: np.ndarray, scope: str,
                  degradation: str = "aggregate") -> None:
    point = improvement(ours_point, baseline_point, metric)
    low, high = ci(distribution)
    rows.append({
        "comparison": comparison, "degradation": degradation, "metric": metric,
        "ours": ours_point, "baseline": baseline_point, "delta_improvement": point,
        "bootstrap_mean": float(np.mean(distribution)), "ci_low": low, "ci_high": high,
        "significant": bool(low > 0.0), "raw_p": centered_bootstrap_pvalue(point, distribution),
        "scope": scope, "sample_count": int(distribution.size if scope == "source" else 0),
    })


def natural_bootstrap_rows(source_values: Dict[str, Dict[str, np.ndarray]],
                           sample_indices: np.ndarray) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    replicate_rows: List[dict] = []
    for baseline in BASELINES:
        comparison = "Ours_vs_%s" % baseline
        for metric in NATURAL_METRICS:
            ours = source_values["Ours"][metric]
            base = source_values[baseline][metric]
            distribution = bootstrap_mean_distribution(ours, base, sample_indices, metric)
            add_stats_row(rows, comparison, metric, float(ours.mean()), float(base.mean()), distribution, "source")
            for index, delta in enumerate(distribution):
                replicate_rows.append({"comparison": comparison, "metric": metric,
                                       "bootstrap_index": index, "delta_improvement": float(delta),
                                       "unit": "source"})
    return rows, replicate_rows


def gallery_bootstrap_rows(score_blocks_by_method: Dict[str, np.ndarray],
                           sample_indices: np.ndarray, source_values: Dict[str, Dict[str, np.ndarray]]) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    replicate_rows: List[dict] = []
    for baseline in BASELINES:
        comparison = "Ours_vs_%s" % baseline
        for metric in ("AUC", "EER"):
            ours_dist = gallery_bootstrap_distribution(score_blocks_by_method["Ours"], sample_indices, metric)
            base_dist = gallery_bootstrap_distribution(score_blocks_by_method[baseline], sample_indices, metric)
            distribution = ours_dist - base_dist if metric == "AUC" else base_dist - ours_dist
            ours_point = gallery_metric(score_blocks_by_method["Ours"], list(range(sample_indices.shape[1])), metric)
            base_point = gallery_metric(score_blocks_by_method[baseline], list(range(sample_indices.shape[1])), metric)
            add_stats_row(rows, comparison, metric, ours_point, base_point, distribution, "source")
            for index, delta in enumerate(distribution):
                replicate_rows.append({"comparison": comparison, "metric": metric,
                                       "bootstrap_index": index, "delta_improvement": float(delta),
                                       "unit": "source"})
    return rows, replicate_rows


def per_degradation_rows(source_values_by_deg: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                         sample_indices: np.ndarray) -> List[dict]:
    rows = []
    for baseline in BASELINES:
        comparison = "Ours_vs_%s" % baseline
        for degradation in DEGRADATIONS:
            for metric in NATURAL_METRICS:
                ours = source_values_by_deg["Ours"][degradation][metric]
                base = source_values_by_deg[baseline][degradation][metric]
                distribution = bootstrap_mean_distribution(ours, base, sample_indices, metric)
                add_stats_row(rows, comparison, metric, float(ours.mean()), float(base.mean()), distribution, "source", degradation)
    return rows


def add_fdr(rows: List[dict]) -> None:
    p_values = np.asarray([row["raw_p"] for row in rows], dtype=np.float64)
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values):
        row["fdr_q"] = float(q_value)
        row["significant_raw"] = bool(row["raw_p"] < 0.05)
        row["significant_fdr"] = bool(q_value < 0.05)


def view_diagnostic(source_values: Dict[str, Dict[str, np.ndarray]],
                    per_view_values: Dict[str, Dict[str, np.ndarray]], seed: int) -> List[dict]:
    rng = np.random.default_rng(seed + 100003)
    source_sample_indices = np.random.default_rng(seed).integers(
        0, len(source_values["Ours"]["PSNR"]), size=(REPETITIONS, len(source_values["Ours"]["PSNR"]))
    )
    rows = []
    for baseline in BASELINES:
        comparison = "Ours_vs_%s" % baseline
        for metric in NATURAL_METRICS:
            ours_source = source_values["Ours"][metric]
            base_source = source_values[baseline][metric]
            source_dist = bootstrap_mean_distribution(ours_source, base_source, source_sample_indices, metric)
            ours_views = per_view_values["Ours"][metric]
            base_views = per_view_values[baseline][metric]
            view_indices = rng.integers(0, len(ours_views), size=(REPETITIONS, len(ours_views)))
            view_dist = bootstrap_mean_distribution(ours_views, base_views, view_indices, metric)
            source_width = ci(source_dist)[1] - ci(source_dist)[0]
            view_width = ci(view_dist)[1] - ci(view_dist)[0]
            rows.append({"comparison": comparison, "metric": metric,
                         "source_level_ci_width": source_width,
                         "view_level_ci_width": view_width,
                         "ratio": view_width / source_width if source_width else float("nan"),
                         "view_level_status": "diagnostic_only"})
    return rows


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return "%.*f" % (digits, value)


def write_artifacts(full_rows: Sequence[dict], table_rows: Sequence[dict], out_dir: Path) -> None:
    fields = ["comparison", "metric", "delta_improvement", "ci_low", "ci_high", "significant"]
    write_csv(out_dir / "table7_statistics.csv", table_rows, fields)
    md = [
        "# Table 7. Source-level Statistical Significance", "",
        "Positive Δ always means Ours is better. Confidence intervals are obtained by paired source-level bootstrap over 256 independent clean sources with 10,000 resamples. All degradation views belonging to the same source are resampled jointly.", "",
        "| Comparison | Metric | Δ Improvement | 95% CI | Significant |",
        "|---|---|---:|---:|:---:|",
    ]
    for row in table_rows:
        unit = " dB" if row["metric"] == "PSNR" else ""
        delta_text = ("+" if row["delta_improvement"] >= 0 else "") + fmt(row["delta_improvement"]) + unit
        md.append("| %s | %s | %s | [%s, %s] | %s |" % (row["comparison"].replace("_", " "), row["metric"], delta_text, fmt(row["ci_low"]), fmt(row["ci_high"]), str(row["significant"]).lower()))
    (out_dir / "table7_source_level_significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex_row_end = "\\\\"
    tex = [
        "% Auto-generated from source-level paired bootstrap",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Source-level statistical significance. Positive $\\Delta$ always indicates an improvement by Ours. Confidence intervals are obtained by paired source-level bootstrap over 256 independent clean sources with 10,000 resamples. All degradation views belonging to the same source are resampled jointly.}",
        "\\begin{tabular}{llrrc}", "\\toprule",
        "Comparison & Metric & $\\Delta$ Improvement & 95\\% CI & Significant " + tex_row_end, "\\midrule",
    ]
    for row in table_rows:
        unit = " dB" if row["metric"] == "PSNR" else ""
        delta_text = ("+" if row["delta_improvement"] >= 0 else "") + fmt(row["delta_improvement"]) + unit
        tex.append("%s & %s & %s & [%s, %s] & %s \\\\" % (row["comparison"].replace("_", " "), row["metric"], delta_text, fmt(row["ci_low"]), fmt(row["ci_high"]), "Yes" if row["significant"] else "No"))
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (out_dir / "table7_source_level_significance.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    supp = [
        "% Full aggregate statistical results; q-values use BH-FDR over aggregate and per-degradation tests.",
        "\\begin{longtable}{llrrrrrr}", "\\caption{Full source-level statistical results.} " + tex_row_end, "\\toprule",
        "Comparison & Metric & Ours & Baseline & $\\Delta$ & CI low & CI high & FDR $q$ " + tex_row_end, "\\midrule",
    ]
    for row in full_rows:
        supp.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (row["comparison"].replace("_", " "), row["metric"], fmt(row["ours"]), fmt(row["baseline"]), fmt(row["delta_improvement"]), fmt(row["ci_low"]), fmt(row["ci_high"]), fmt(row.get("fdr_q"))))
    supp.extend(["\\bottomrule", "\\end{longtable}"])
    (out_dir / "supplementary_statistics.tex").write_text("\n".join(supp) + "\n", encoding="utf-8")


def write_report(path: Path, protocol: dict, full_rows: Sequence[dict], per_deg: Sequence[dict], diagnostic: Sequence[dict], issues: Sequence[str], missing: Sequence[dict]) -> None:
    lookup = {(row["comparison"], row["metric"]): row for row in full_rows}
    lines = [
        "# Statistical Significance Report", "",
        "## 1. Statistical unit and protocol", "",
        "- Independent unit: source; 256 clean sources, 7 jointly resampled views per source, 1792 total views.",
        "- Paired bootstrap: 10,000 repetitions, with replacement, seed 13, percentile 95% CI.",
        "- Positive Δ means Ours is better for every metric; LPIPS/EER use baseline minus Ours.", "",
        "## 2. Data integrity audit", "",
        "- Required methods: Ours, Restormer, DehazeFormer, PromptIR; valid sources: %d." % protocol["independent_source_count"],
        "- Missing statistical samples: %d; protocol issues: %d." % (len(missing), len(issues)),
        "- Same source IDs, seven degradations, test-only split, and official frozen verifier were checked.", "",
        "## 3. Aggregate sanity check", "",
    ]
    for method, sanity in protocol["aggregate_sanity"].items():
        lines.append("- %s: %s." % (method, "PASS" if sanity["passed"] else "MISMATCH"))
        for metric in ("PSNR", "SSIM", "LPIPS", "Top1", "AUC", "EER"):
            lines.append("  - %s recomputed %.8f vs official %.8f (difference %.8f)." % (metric, sanity["recomputed"][metric], sanity["official"][metric], sanity["absolute_difference"][metric]))
    if issues or missing or not full_rows:
        lines += ["", "## Reliability gate", "", "Formal Table 7 was not generated because required inputs or protocol checks failed."]
        if issues:
            lines.append("- Issues: %s." % "; ".join(issues))
        if missing:
            lines.append("- Missing sample records: %d; see `missing_statistical_samples.csv`." % len(missing))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines += ["", "## 4. Table 7 headline results", ""]
    for row in full_rows:
        if row["metric"] in MAIN_TABLE_METRICS:
            lines.append("- %s %s: Δ=%+.6f, 95%% CI [%.6f, %.6f], significant=%s." % (row["comparison"], row["metric"], row["delta_improvement"], row["ci_low"], row["ci_high"], row["significant"]))
    crossing = [row for row in full_rows if row["ci_low"] <= 0.0 <= row["ci_high"]]
    baseline_favored = [row for row in full_rows if row["ci_high"] < 0.0]
    positive_significant = [row for row in full_rows if row["ci_low"] > 0.0]
    lines.append("- Aggregate metrics whose 95%% CI crosses zero: %s." % (", ".join("%s/%s" % (r["comparison"], r["metric"]) for r in crossing) or "none"))
    lines.append("- Aggregate metrics with a CI entirely above zero: %s." % (", ".join("%s/%s" % (r["comparison"], r["metric"]) for r in positive_significant) or "none"))
    lines.append("- Aggregate metrics with a CI entirely below zero (baseline favored under the Δ convention): %s." % (", ".join("%s/%s" % (r["comparison"], r["metric"]) for r in baseline_favored) or "none"))
    lines += ["", "## 5. Full aggregate results", "", "All eight requested metrics are retained in `full_statistical_results.csv`; no non-significant metric was hidden from the supplementary analysis.", ""]
    lines += ["## 6. Per-degradation analysis", "", "Per-degradation source-level CIs cover PSNR, SSIM, LPIPS, Top1, Cosine, and Margin. AUC/EER remain aggregate gallery-level metrics and are recomputed at aggregate bootstrap level, not split into pseudo-source values.", ""]
    unstable = [row for row in per_deg if not row["significant"]]
    per_crossing = [row for row in per_deg if row["ci_low"] <= 0.0 <= row["ci_high"]]
    per_baseline_favored = [row for row in per_deg if row["ci_high"] < 0.0]
    lines.append("- Non-significant per-degradation natural-metric comparisons: %d/%d." % (len(unstable), len(per_deg)))
    lines.append("- Per-degradation CIs crossing zero: %s." % (", ".join("%s/%s/%s" % (r["comparison"], r["degradation"], r["metric"]) for r in per_crossing) or "none"))
    lines.append("- Per-degradation CIs entirely below zero (baseline favored): %s." % (", ".join("%s/%s/%s" % (r["comparison"], r["degradation"], r["metric"]) for r in per_baseline_favored) or "none"))
    lines += ["", "## 7. Multiple comparisons", "", "Supplementary raw p-values use a centered-bootstrap two-sided tail calculation: the bootstrap distribution is centered by subtracting the observed point estimate, and both tails at least as extreme as the observed statistic are doubled conservatively. Benjamini-Hochberg q-values are computed over aggregate and per-degradation tests.", ""]
    lines += ["## 8. Source bootstrap versus naive view bootstrap", "", "The source-level result is the formal analysis. The naive view-level comparison is diagnostic only and resamples 1792 views independently. If its CI is narrower, that is evidence that treating correlated degradation views as independent underestimates uncertainty."]
    narrower = sum(row["view_level_ci_width"] < row["source_level_ci_width"] for row in diagnostic)
    wider = sum(row["view_level_ci_width"] > row["source_level_ci_width"] for row in diagnostic)
    lines.append("- Across %d natural-metric diagnostics, naive view-level CIs are narrower in %d and wider in %d; the effect is not uniform, but independent-view resampling does not preserve source clustering and is not used for the paper result." % (len(diagnostic), narrower, wider))
    for row in diagnostic:
        lines.append("- %s/%s: source CI width %.6f; view CI width %.6f; view/source ratio %.3f." % (row["comparison"], row["metric"], row["source_level_ci_width"], row["view_level_ci_width"], row["ratio"]))
    lines += ["- AUC/EER are omitted from the naive diagnostic because the formal gallery-level bootstrap is the required evidence and pseudo-source AUC/EER are invalid.", "", "## 9. Paper-safe conclusions", ""]
    significant = [row for row in full_rows if row["significant"]]
    lines.append("1. Ours improves aggregate PSNR over Restormer by %.4f dB (95%% CI [%.4f, %.4f])." % (lookup[("Ours_vs_Restormer", "PSNR")]["delta_improvement"], lookup[("Ours_vs_Restormer", "PSNR")]["ci_low"], lookup[("Ours_vs_Restormer", "PSNR")]["ci_high"]))
    lines.append("2. LPIPS, Top1, and Margin should be described as statistically significant only for comparisons whose source-level 95% CI is entirely above zero; DehazeFormer has a significantly lower LPIPS than Ours in this frozen test result.")
    lines.append("3. The source-level bootstrap preserves the seven correlated degradation views within each source, so its intervals are the appropriate uncertainty statement for this test set.")
    if not significant:
        lines.append("4. No requested aggregate metric has a CI entirely above zero.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args(argv)
    repetitions = int(args.repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    out_results = ROOT / "results/statistical_significance"
    out_artifacts = ROOT / "artifacts/statistical_significance"
    out_results.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)

    groups, source_ids, audit = load_groups()
    missing: List[dict] = []
    metadata_by_method: Dict[str, dict] = {}
    quality_by_method: Dict[str, Dict[Tuple[str, str], dict]] = {}
    scores_by_method: Dict[str, Dict[Tuple[str, str], dict]] = {}
    source_values: Dict[str, Dict[str, np.ndarray]] = {}
    per_view_values: Dict[str, Dict[str, np.ndarray]] = {}
    source_values_by_deg: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for method in METHODS:
        quality = load_quality(method, source_ids, missing)
        scores, metadata = load_method_scores(method, source_ids, missing)
        quality_by_method[method] = quality
        scores_by_method[method] = scores
        metadata_by_method[method] = metadata
        source_values[method] = make_source_metric_values(method, quality, scores, source_ids, missing)
        per_view_values[method] = make_per_view_metric_values(quality, scores, source_ids)
        source_values_by_deg[method] = {degradation: {metric: np.asarray([
            (float(quality[(sid, degradation)][{"PSNR": "psnr", "SSIM": "ssim", "LPIPS": "lpips"}[metric]])
             if metric in ("PSNR", "SSIM", "LPIPS") else
             (float(int(np.argmax(scores[(sid, degradation)]["restored_scores"]) == source_ids.index(sid))) if metric == "Top1" else
              float(scores[(sid, degradation)]["restored_scores"][source_ids.index(sid)]) if metric == "Cosine" else
              float(scores[(sid, degradation)]["restored_scores"][source_ids.index(sid)] - np.max(np.delete(scores[(sid, degradation)]["restored_scores"], source_ids.index(sid))))))
            for sid in source_ids if (sid, degradation) in quality and (sid, degradation) in scores], dtype=np.float64) for metric in NATURAL_METRICS} for degradation in DEGRADATIONS}

    required_keys = {(sid, degradation) for sid in source_ids for degradation in DEGRADATIONS}
    for method in METHODS:
        if set(quality_by_method[method]) != required_keys:
            missing.append({"method": method, "source_id": "<set>", "degradation": "<set>", "metric": "all", "reason": "quality source/degradation set mismatch"})
        if set(scores_by_method[method]) != required_keys:
            missing.append({"method": method, "source_id": "<set>", "degradation": "<set>", "metric": "identity", "reason": "score source/degradation set mismatch"})

    issues = protocol_issues(metadata_by_method, audit, source_ids)
    sanity: Dict[str, dict] = {}
    if not missing and not issues:
        for method in METHODS:
            sanity[method] = aggregate_sanity(method, quality_by_method[method], score_blocks(scores_by_method[method], source_ids), source_ids, metadata_by_method[method])
            if not sanity[method]["passed"]:
                issues.append("%s aggregate sanity check exceeded tolerance" % method)
    protocol = {
        "bootstrap_unit": "source",
        "independent_source_count": len(source_ids), "views_per_source": len(DEGRADATIONS),
        "total_views": len(source_ids) * len(DEGRADATIONS), "bootstrap_sample_size": len(source_ids),
        "bootstrap_repetitions": repetitions, "bootstrap_sampling": "with_replacement", "paired": True,
        "seed": SEED, "ci_method": "percentile", "confidence_level": CONFIDENCE,
        "recomputed_per_bootstrap": {"AUC": True, "EER": True},
        "verifier_checkpoint": str(OFFICIAL_VERIFIER), "verifier_sha256": sha256_file(OFFICIAL_VERIFIER) if OFFICIAL_VERIFIER.exists() else None,
        "embedding_dim": metadata_by_method["Ours"].get("verifier_fingerprint", {}).get("embedding_dim"),
        "test_split": "test", "degradations": list(DEGRADATIONS), "methods": list(METHODS),
        "aggregate_sanity": sanity, "protocol_issues": issues,
        "source_id_order_sha256": __import__("hashlib").sha256("|".join(source_ids).encode()).hexdigest(),
    }
    write_csv(out_results / "missing_statistical_samples.csv", missing, ["method", "source_id", "degradation", "metric", "reason"])
    (out_results / "bootstrap_metadata.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    if missing or issues:
        (out_artifacts / "STATISTICAL_PROTOCOL_MISMATCH.md").write_text("# STATISTICAL_PROTOCOL_MISMATCH\n\n" + "\n".join(["- " + x for x in issues]) + "\n", encoding="utf-8")
        write_report(out_artifacts / "STATISTICAL_SIGNIFICANCE_REPORT.md", protocol, [], [], [], issues, missing)
        return 2

    rng = np.random.default_rng(SEED)
    sample_indices = rng.integers(0, len(source_ids), size=(repetitions, len(source_ids)))
    full_rows, bootstrap_rows = natural_bootstrap_rows(source_values, sample_indices)
    blocks = {method: score_blocks(scores_by_method[method], source_ids) for method in METHODS}
    gallery_rows, gallery_bootstrap_replicates = gallery_bootstrap_rows(blocks, sample_indices, source_values)
    full_rows.extend(gallery_rows)
    bootstrap_rows.extend(gallery_bootstrap_replicates)
    # Correct the complete aggregate + per-degradation family together.
    source_values_by_deg = {method: {d: {metric: values for metric, values in source_values_by_deg[method][d].items()} for d in DEGRADATIONS} for method in METHODS}
    per_deg = per_degradation_rows(source_values_by_deg, sample_indices)
    add_fdr(full_rows + per_deg)
    table_rows = [row for row in full_rows if row["metric"] in MAIN_TABLE_METRICS]
    write_csv(out_results / "source_level_metrics.csv", [
        {"source_id": sid, "method": method, "metric": metric, "value": float(value)}
        for method in METHODS for metric in NATURAL_METRICS for sid, value in zip(source_ids, source_values[method][metric])
    ], ["source_id", "method", "metric", "value"])
    write_csv(out_results / "source_level_bootstrap.csv", bootstrap_rows,
              ["comparison", "metric", "bootstrap_index", "delta_improvement", "unit"])
    write_csv(out_results / "full_statistical_results.csv", full_rows,
              ["comparison", "degradation", "metric", "ours", "baseline", "delta_improvement", "bootstrap_mean", "ci_low", "ci_high", "significant", "raw_p", "fdr_q", "significant_raw", "significant_fdr", "scope", "sample_count"])
    write_csv(out_results / "table7_statistics.csv", table_rows,
              ["comparison", "metric", "delta_improvement", "ci_low", "ci_high", "significant"])
    write_csv(out_results / "per_degradation_significance.csv", per_deg,
              ["comparison", "degradation", "metric", "ours", "baseline", "delta_improvement", "bootstrap_mean", "ci_low", "ci_high", "significant", "raw_p", "fdr_q", "significant_raw", "significant_fdr", "scope", "sample_count"])
    diagnostic = view_diagnostic(source_values, per_view_values, SEED)
    write_csv(out_results / "bootstrap_unit_diagnostic.csv", diagnostic,
              ["comparison", "metric", "source_level_ci_width", "view_level_ci_width", "ratio", "view_level_status"])
    write_artifacts(full_rows, table_rows, out_artifacts)
    write_report(out_artifacts / "STATISTICAL_SIGNIFICANCE_REPORT.md", protocol, full_rows, per_deg, diagnostic, issues, missing)
    print("generated source-level statistical significance")
    print("sources=%d views=%d repetitions=%d" % (len(source_ids), len(source_ids) * len(DEGRADATIONS), repetitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
