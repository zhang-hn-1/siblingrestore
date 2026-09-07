"""Generate Table 3 identity-preservation evidence from official v2 outputs.

This module never runs a restoration model or verifier. It consumes the score
vectors emitted by ``scripts/evaluate_frozen_verifier_v2.py`` so that all
methods share the verifier, gallery, preprocessing, and test split already
used by Table 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.identity_preservation.core import (  # noqa: E402
    bootstrap_mean_ci,
    paired_bootstrap_ci,
    recovery_metrics,
)

DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")
DISPLAY_DEGRADATIONS = {
    "blur": "Blur", "haze": "Haze", "inpainting": "Inpainting",
    "lowlight": "Low-light", "noise": "Noise", "rain": "Rain", "snow": "Snow",
}
MAIN_METHODS = ("Degraded", "Restormer", "DehazeFormer", "PromptIR", "Ours")
RESTORATION_METHODS = ("Restormer", "DehazeFormer", "PromptIR", "Ours")
METHOD_PATHS = {
    "Degraded": "results/ablation_core/A5_no_sibling.13",
    "Ours": "results/ablation_core/A5_no_sibling.13",
    "Restormer": "results/campaigns/c005_official_group2/restormer.13",
    "DehazeFormer": "results/campaigns/c005_official_group1/dehazeformer.13",
    "PromptIR": "results/campaigns/c005_official_group2/promptir.13",
    "DFPIR": "results/campaigns/c005_official_group1/dfpir.13",
    "FFANet": "results/campaigns/c005_official_group1/ffanet.13",
    "Uformer": "results/campaigns/c005_official_group1/uformer.13",
}
OFFICIAL_VERIFIER = ROOT / "runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt"
EXPECTED_VERIFIER_SHA = "d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6"
BOOTSTRAP_SEED = 13
BOOTSTRAP_SAMPLES = 2000
SANITY_TOLERANCE = 0.005


def result_path(method: str, suffix: str) -> Path:
    prefix = ROOT / METHOD_PATHS[method]
    return prefix.with_suffix(prefix.suffix + suffix)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_groups() -> Tuple[List[dict], List[str], dict]:
    data_root = ROOT / "data/plamd_sfr_v1"
    groups = read_json(data_root / "metadata/source_groups.json")
    test = [g for g in groups if g.get("split") == "test"]
    test_ids = [g["source_id"] for g in test]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("test source_id is not unique")
    train_ids = {g["source_id"] for g in groups if g.get("split") == "train"}
    if train_ids.intersection(test_ids):
        raise ValueError("train source entered test gallery")
    if len(test) != 256:
        raise ValueError("official Table2 test split must contain 256 sources")
    if any(tuple(sorted(g.get("degraded", {}).keys())) != tuple(sorted(DEGRADATIONS)) for g in test):
        raise ValueError("test source does not contain exactly seven degradations")
    audit = read_json(data_root / "metadata/audit.json")
    return test, test_ids, audit


def _missing_row(missing: List[dict], method: str, source_id: str,
                 degradation: str, expected_path: Path, reason: str) -> None:
    missing.append({
        "method": method, "source_id": source_id, "degradation": degradation,
        "expected_path": str(expected_path), "reason": reason,
    })


def load_method_scores(
    method: str,
    source_ids: Sequence[str],
    missing: List[dict],
) -> Tuple[Dict[Tuple[str, str], dict], dict]:
    """Load the last official record per source, matching v2 resume semantics."""
    source_path = result_path(method, ".test.json.sources.jsonl")
    json_path = result_path(method, ".test.json")
    csv_path = result_path(method, ".test.csv")
    records: Dict[Tuple[str, str], dict] = {}
    metadata: dict = {}
    if json_path.exists():
        metadata = read_json(json_path)
    else:
        for source_id in source_ids:
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, json_path, "missing result json")
    if not csv_path.exists():
        for source_id in source_ids:
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, csv_path, "missing result csv")
    if not source_path.exists():
        for source_id in source_ids:
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, source_path, "missing official score log")
        return records, metadata

    last_by_source: Dict[str, dict] = {}
    try:
        for line_no, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                for degradation in DEGRADATIONS:
                    _missing_row(missing, method, "<unknown>", degradation, source_path,
                                 "invalid JSON at line %d: %s" % (line_no, exc.msg))
                continue
            source_id = record.get("source_id")
            if source_id in source_ids:
                last_by_source[source_id] = record
    except OSError as exc:
        for source_id in source_ids:
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, source_path, str(exc))
        return records, metadata

    source_set = set(source_ids)
    for source_id in source_ids:
        record = last_by_source.get(source_id)
        if record is None:
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, source_path,
                             "source missing from official score log")
            continue
        views = {view.get("degradation"): view for view in record.get("views", [])}
        if len(views) != len(record.get("views", [])):
            for degradation in DEGRADATIONS:
                _missing_row(missing, method, source_id, degradation, source_path,
                             "duplicate degradation in source record")
        for degradation in DEGRADATIONS:
            view = views.get(degradation)
            if view is None:
                _missing_row(missing, method, source_id, degradation, source_path,
                             "degradation missing from source record")
                continue
            for query_type in ("input_scores", "restored_scores"):
                scores = view.get(query_type)
                if not isinstance(scores, list) or len(scores) != len(source_ids):
                    _missing_row(missing, method, source_id, degradation, source_path,
                                 "%s length is not gallery size %d" % (query_type, len(source_ids)))
                    break
                if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
                    _missing_row(missing, method, source_id, degradation, source_path,
                                 "%s contains non-finite values" % query_type)
                    break
            else:
                records[(source_id, degradation)] = {
                    "input_scores": np.asarray(view["input_scores"], dtype=np.float64),
                    "restored_scores": np.asarray(view["restored_scores"], dtype=np.float64),
                }
    unknown = sorted(set(last_by_source).difference(source_set))
    if unknown:
        for source_id in unknown:
            _missing_row(missing, method, source_id, "<unknown>", source_path,
                         "source is not in official test gallery")
    return records, metadata


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    return (sum(rank for rank, label in zip(ranks, labels) if label)
            - positives * (positives + 1) / 2) / (positives * negatives)


def eer(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int8)
    positives = int(labels_np.sum())
    negatives = int(labels_np.size - positives)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(-scores_np, kind="stable")
    sorted_scores = scores_np[order]
    sorted_labels = labels_np[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    fpr = cumulative_fp[group_ends] / negatives
    fnr = (positives - cumulative_tp[group_ends]) / positives
    return float(np.min((fpr + fnr) / 2.0))


def official_sanity(records: Dict[Tuple[str, str], dict], metadata: dict,
                    source_ids: Sequence[str]) -> dict:
    restored_scores: List[float] = []
    labels: List[int] = []
    top1_hits = 0
    total = 0
    index_of = {source_id: index for index, source_id in enumerate(source_ids)}
    for source_id in source_ids:
        own = index_of[source_id]
        for degradation in DEGRADATIONS:
            scores = records[(source_id, degradation)]["restored_scores"]
            restored_scores.extend(scores.tolist())
            labels.extend(int(i == own) for i in range(len(scores)))
            top1_hits += int(int(np.argmax(scores)) == own)
            total += 1
    recomputed = {
        "top1": top1_hits / total,
        "auc": binary_auc(restored_scores, labels),
        "eer": eer(restored_scores, labels),
    }
    official = metadata.get("aggregate", {})
    official_values = {
        "top1": official.get("restored_to_clean_top1"),
        "auc": official.get("restored_roc_auc"),
        "eer": official.get("restored_eer"),
    }
    differences = {
        key: (abs(recomputed[key] - official_values[key])
              if official_values[key] is not None else float("inf"))
        for key in recomputed
    }
    return {
        "recomputed": recomputed,
        "official": official_values,
        "absolute_difference": differences,
        "passed": all(value <= SANITY_TOLERANCE for value in differences.values()),
        "tolerance": SANITY_TOLERANCE,
    }


def finite_summary(values: np.ndarray) -> dict:
    summary = bootstrap_mean_ci(values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    return {key: float(value) if isinstance(value, (np.floating, float)) else value
            for key, value in summary.items()}


def method_protocol(metadata: dict) -> dict:
    fingerprint = metadata.get("verifier_fingerprint", {})
    return {
        "verifier_sha256": fingerprint.get("sha256"),
        "verifier_path": fingerprint.get("path"),
        "embedding_dim": fingerprint.get("embedding_dim"),
        "eval_version": metadata.get("eval_version"),
        "split": metadata.get("split"),
        "tile_size": metadata.get("config", {}).get("eval_tile_size", 512),
        "tile_overlap": metadata.get("config", {}).get("eval_tile_overlap", 32),
        "data_pkg_fingerprint": metadata.get("data_pkg_fingerprint", {}),
    }


def validate_protocol(metadata_by_method: Dict[str, dict], audit: dict,
                      missing: List[dict]) -> List[str]:
    issues: List[str] = []
    if not OFFICIAL_VERIFIER.exists():
        issues.append("official verifier checkpoint is missing: %s" % OFFICIAL_VERIFIER)
    elif sha256_file(OFFICIAL_VERIFIER) != EXPECTED_VERIFIER_SHA:
        issues.append("official verifier checkpoint hash differs from Table2 protocol")
    for method, metadata in metadata_by_method.items():
        protocol = method_protocol(metadata)
        if protocol["verifier_sha256"] != EXPECTED_VERIFIER_SHA:
            issues.append("%s verifier fingerprint differs" % method)
        if protocol["split"] != "test":
            issues.append("%s is not from test split" % method)
        if protocol["embedding_dim"] != 128:
            issues.append("%s embedding_dim is not 128" % method)
        pkg = protocol["data_pkg_fingerprint"]
        if pkg.get("audit_index_sha256") and pkg["audit_index_sha256"] != audit.get("index_sha256"):
            issues.append("%s data index fingerprint differs" % method)
        if pkg.get("audit_groups_sha256") and pkg["audit_groups_sha256"] != audit.get("groups_sha256"):
            issues.append("%s data group fingerprint differs" % method)
    return issues


def build_metric_rows(
    scores_by_method: Dict[str, Dict[Tuple[str, str], dict]],
    source_ids: Sequence[str],
) -> Tuple[List[dict], Dict[str, dict], Dict[str, Dict[str, np.ndarray]]]:
    index_of = {source_id: i for i, source_id in enumerate(source_ids)}
    long_rows: List[dict] = []
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for method, records in scores_by_method.items():
        arrays[method] = {"cosine": [], "margin": [], "source_ids": []}
        for source_id in source_ids:
            for degradation in DEGRADATIONS:
                entry = records[(source_id, degradation)]
                scores = entry["input_scores"] if method == "Degraded" else entry["restored_scores"]
                own = index_of[source_id]
                wrong = np.delete(scores, own)
                cosine = float(scores[own])
                hardest = float(np.max(wrong))
                margin = cosine - hardest
                long_rows.append({
                    "method": method, "source_id": source_id, "degradation": degradation,
                    "cosine": cosine, "hardest_negative": hardest, "margin": margin,
                })
                arrays[method]["cosine"].append(cosine)
                arrays[method]["margin"].append(margin)
                arrays[method]["source_ids"].append(source_id)
        arrays[method]["cosine"] = np.asarray(arrays[method]["cosine"], dtype=np.float64)
        arrays[method]["margin"] = np.asarray(arrays[method]["margin"], dtype=np.float64)
        arrays[method]["source_ids"] = np.asarray(arrays[method]["source_ids"])

    row_lookup = {(r["method"], r["source_id"], r["degradation"]): r for r in long_rows}
    summary_rows: List[dict] = []
    for method in scores_by_method:
        for degradation in list(DEGRADATIONS) + ["average"]:
            if degradation == "average":
                per_source_cos = np.asarray([
                    np.mean([row_lookup[(method, source_id, d)]["cosine"] for d in DEGRADATIONS])
                    for source_id in source_ids])
                per_source_margin = np.asarray([
                    np.mean([row_lookup[(method, source_id, d)]["margin"] for d in DEGRADATIONS])
                    for source_id in source_ids])
                n_views = len(source_ids) * len(DEGRADATIONS)
            else:
                per_source_cos = np.asarray([
                    row_lookup[(method, source_id, degradation)]["cosine"] for source_id in source_ids])
                per_source_margin = np.asarray([
                    row_lookup[(method, source_id, degradation)]["margin"] for source_id in source_ids])
                n_views = len(source_ids)
            cos_summary = finite_summary(per_source_cos)
            margin_summary = finite_summary(per_source_margin)
            summary_rows.append({
                "method": method, "degradation": degradation,
                "sample_count": len(source_ids), "view_count": n_views,
                "cosine_mean": cos_summary["mean"], "cosine_std": cos_summary["std"],
                "cosine_ci_low": cos_summary["ci_low"], "cosine_ci_high": cos_summary["ci_high"],
                "margin_mean": margin_summary["mean"], "margin_std": margin_summary["std"],
                "margin_ci_low": margin_summary["ci_low"], "margin_ci_high": margin_summary["ci_high"],
            })
    by_key = {(r["method"], r["degradation"]): r for r in summary_rows}
    degraded_cos = {(d): np.asarray([
        row_lookup[("Degraded", sid, d)]["cosine"] for sid in source_ids]) for d in DEGRADATIONS}
    for method in scores_by_method:
        if method == "Degraded":
            continue
        restored_cos = {d: np.asarray([
            row_lookup[(method, sid, d)]["cosine"] for sid in source_ids]) for d in DEGRADATIONS}
        restored_all = np.asarray([
            np.mean([restored_cos[d][i] for d in DEGRADATIONS]) for i in range(len(source_ids))])
        degraded_all = np.asarray([
            np.mean([degraded_cos[d][i] for d in DEGRADATIONS]) for i in range(len(source_ids))])
        for degradation in list(DEGRADATIONS) + ["average"]:
            d_cos = degraded_all if degradation == "average" else degraded_cos[degradation]
            r_cos = restored_all if degradation == "average" else restored_cos[degradation]
            recovery = recovery_metrics(d_cos, r_cos)
            delta_summary = finite_summary(recovery["delta_cosine"])
            ratio_values = recovery["recovery_ratio"]
            ratio_summary = finite_summary(ratio_values[np.isfinite(ratio_values)])
            row = by_key[(method, degradation)]
            row.update({
                "delta_cosine_mean": delta_summary["mean"],
                "delta_cosine_std": delta_summary["std"],
                "delta_cosine_ci_low": delta_summary["ci_low"],
                "delta_cosine_ci_high": delta_summary["ci_high"],
                "recovery_ratio_mean": ratio_summary["mean"],
                "recovery_ratio_std": ratio_summary["std"],
                "recovery_ratio_ci_low": ratio_summary["ci_low"],
                "recovery_ratio_ci_high": ratio_summary["ci_high"],
            })
    for row in summary_rows:
        for key in ("delta_cosine_mean", "delta_cosine_std", "delta_cosine_ci_low",
                    "delta_cosine_ci_high", "recovery_ratio_mean", "recovery_ratio_std",
                    "recovery_ratio_ci_low", "recovery_ratio_ci_high"):
            row.setdefault(key, None)
    return long_rows, by_key, arrays


def paired_rows(scores_by_method: Dict[str, Dict[Tuple[str, str], dict]],
                source_ids: Sequence[str], summary_lookup: Dict[Tuple[str, str], dict]) -> List[dict]:
    rows: List[dict] = []
    for baseline in ("Restormer", "DehazeFormer", "PromptIR"):
        for degradation in list(DEGRADATIONS) + ["average"]:
            ours_delta = []
            baseline_delta = []
            for source_id in source_ids:
                degs = DEGRADATIONS if degradation == "average" else (degradation,)
                ours_values, baseline_values, degraded_values = [], [], []
                own = source_ids.index(source_id)
                for d in degs:
                    degraded = scores_by_method["Degraded"][(source_id, d)]["input_scores"][own]
                    ours = scores_by_method["Ours"][(source_id, d)]["restored_scores"][own]
                    base = scores_by_method[baseline][(source_id, d)]["restored_scores"][own]
                    degraded_values.append(float(degraded))
                    ours_values.append(float(ours))
                    baseline_values.append(float(base))
                ours_delta.append(float(np.mean(ours_values) - np.mean(degraded_values)))
                baseline_delta.append(float(np.mean(baseline_values) - np.mean(degraded_values)))
            result = paired_bootstrap_ci(np.asarray(ours_delta), np.asarray(baseline_delta),
                                         BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            rows.append({
                "comparison": "Ours_vs_%s" % baseline,
                "degradation": degradation,
                "metric": "delta_cosine",
                "mean_difference": result["mean"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "sample_count": result["count"],
                "significant": result["significant"],
            })
    return rows


def make_protocol(metadata_by_method: Dict[str, dict], source_ids: Sequence[str], audit: dict,
                  sanity: dict, methods: Sequence[str], issues: Sequence[str]) -> dict:
    ours_protocol = method_protocol(metadata_by_method["Ours"])
    return {
        "verifier_checkpoint": str(OFFICIAL_VERIFIER),
        "checkpoint_sha256": sha256_file(OFFICIAL_VERIFIER) if OFFICIAL_VERIFIER.exists() else None,
        "embedding_dim": ours_protocol["embedding_dim"],
        "test_split": "test",
        "test_source_count": len(source_ids),
        "gallery_source_count": len(source_ids),
        "degradation_count": len(DEGRADATIONS),
        "degradations": list(DEGRADATIONS),
        "preprocessing": {
            "input_range": "RGB float32 [0,1]",
            "verifier_input": "official evaluate_frozen_verifier_v2 verifier_embed_tiled",
            "tile_size": ours_protocol["tile_size"],
            "tile_overlap": ours_protocol["tile_overlap"],
            "noise": "clean Lanczos alignment to noise native size per Method B protocol",
        },
        "normalization": "verifier.embed output is L2-normalized; score is dot product cosine",
        "distance_metric": "cosine similarity / dot product of L2-normalized embeddings",
        "negative_mining_strategy": "hardest negative = max similarity over all wrong clean gallery sources",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "methods": list(methods),
        "verifier_fingerprints_by_method": {
            method: method_protocol(metadata_by_method[method]) for method in methods
        },
        "data_audit": {
            "index_sha256": audit.get("index_sha256"),
            "groups_sha256": audit.get("groups_sha256"),
            "source_audit_status": audit.get("source_audit_status"),
        },
        "aggregate_sanity_check": sanity,
        "protocol_issues": list(issues),
    }


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return ("%.*f" % (digits, value))


def rank_format(values: Sequence[Optional[float]], index: int, higher: bool = True) -> str:
    valid = [(i, value) for i, value in enumerate(values) if value is not None]
    ordered = sorted(valid, key=lambda pair: pair[1], reverse=higher)
    ranks = {item[0]: rank for rank, item in enumerate(ordered)}
    value = fmt(values[index])
    if ranks.get(index) == 0:
        return "\\textbf{%s}" % value
    if ranks.get(index) == 1:
        return "\\underline{%s}" % value
    return value


def write_tables(summary_lookup: Dict[Tuple[str, str], dict], out_dir: Path) -> None:
    table_rows = []
    for degradation in list(DEGRADATIONS) + ["average"]:
        row = {"Degradation": DISPLAY_DEGRADATIONS.get(degradation, "Average")}
        for method in MAIN_METHODS:
            row[method] = summary_lookup[(method, degradation)]["cosine_mean"]
        row["Ours ΔCosine"] = summary_lookup[("Ours", degradation)]["delta_cosine_mean"]
        table_rows.append(row)
    fields = ["Degradation"] + list(MAIN_METHODS) + ["Ours ΔCosine"]
    write_csv(out_dir / "table3_identity_recovery.csv", table_rows, fields)
    md_lines = [
        "# Table 3. Per-Degradation Identity Recovery", "",
        "Values are means over the 256-source test split. Bold/underline indicate the best/second restoration method per row; Degraded is a reference and is not ranked.", "",
        "| Degradation | Degraded Cosine↑ | Restormer Cosine↑ | DehazeFormer Cosine↑ | PromptIR Cosine↑ | Ours Cosine↑ | Ours ΔCosine↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "% Auto-generated from results/identity_preservation/per_degradation_identity.csv",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Per-degradation identity recovery on the PLAMD test split. Cosine is the similarity to the paired clean source in the shared 256-source gallery. Bold and underline mark the best and second restoration methods in each row; Degraded is reference-only.}",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Degradation & Degraded Cosine$\\uparrow$ & Restormer Cosine$\\uparrow$ & DehazeFormer Cosine$\\uparrow$ & PromptIR Cosine$\\uparrow$ & Ours Cosine$\\uparrow$ & Ours $\\Delta$Cosine$\\uparrow$ " + "\\\\",
        "\\midrule",
    ]
    for row in table_rows:
        values = [row[m] for m in MAIN_METHODS]
        restoration_values = values[1:]
        md_values = [fmt(values[0])] + [fmt(v) for v in restoration_values] + [fmt(row["Ours ΔCosine"])]
        ranked = sorted(restoration_values, reverse=True)
        md_values[1:] = [(("**%s**" % fmt(v)) if v == ranked[0] else ("_%s_" % fmt(v) if len(ranked) > 1 and v == ranked[1] else fmt(v))) for v in restoration_values]
        md_lines.append("| %s | %s |" % (row["Degradation"], " | ".join(md_values)))
        tex_values = [fmt(values[0])] + [rank_format(restoration_values, i) for i in range(4)] + [fmt(row["Ours ΔCosine"])]
        tex_lines.append("%s & %s \\\\" % (row["Degradation"], " & ".join(tex_values)))
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "table3_identity_recovery.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "table3_identity_recovery.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    margin_rows = []
    margin_fields = ["Degradation"] + [method + " Margin" for method in MAIN_METHODS]
    for degradation in list(DEGRADATIONS) + ["average"]:
        row = {"Degradation": DISPLAY_DEGRADATIONS.get(degradation, "Average")}
        for method in MAIN_METHODS:
            row[method + " Margin"] = summary_lookup[(method, degradation)]["margin_mean"]
        margin_rows.append(row)
    write_csv(out_dir / "table3_identity_margin_supp.csv", margin_rows, margin_fields)
    margin_tex = [
        "% Auto-generated supplementary identity-margin table",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Hardest-negative identity margin. The negative gallery contains all wrong clean sources and excludes the query's own clean source.}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Degradation & Degraded Margin$\\uparrow$ & Restormer Margin$\\uparrow$ & DehazeFormer Margin$\\uparrow$ & PromptIR Margin$\\uparrow$ & Ours Margin$\\uparrow$ " + "\\\\",
        "\\midrule",
    ]
    for row in margin_rows:
        values = [row[method + " Margin"] for method in MAIN_METHODS]
        margin_tex.append("%s & %s \\\\" % (row["Degradation"], " & ".join([fmt(values[0])] + [rank_format(values[1:], i) for i in range(4)])))
    margin_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "table3_identity_margin_supp.tex").write_text("\n".join(margin_tex) + "\n", encoding="utf-8")


def plot_figure(rows: Sequence[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(DEGRADATIONS))
    colors = {"Restormer": "#333333", "DehazeFormer": "#666666", "PromptIR": "#999999", "Ours": "#000000"}
    markers = {"Restormer": "o", "DehazeFormer": "s", "PromptIR": "^", "Ours": "D"}
    for method in RESTORATION_METHODS:
        values = [next(r["delta_cosine"] for r in rows if r["method"] == method and r["degradation"] == d) for d in DEGRADATIONS]
        ax.plot(x, values, label=method, color=colors[method], marker=markers[method], linewidth=1.5, markersize=4)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(x, [DISPLAY_DEGRADATIONS[d] for d in DEGRADATIONS], rotation=20, ha="right")
    ax.set_ylabel("Identity recovery gain ($\\Delta$Cosine)")
    ax.set_xlabel("Degradation")
    ax.set_title("Per-degradation identity recovery")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_dir / "figure6d_identity_recovery.pdf")
    fig.savefig(out_dir / "figure6d_identity_recovery.png", dpi=300)
    plt.close(fig)


def write_report(path: Path, protocol: dict, summary_lookup: Dict[Tuple[str, str], dict],
                 bootstrap_rows: Sequence[dict], optional_methods: Sequence[str],
                 missing: Sequence[dict], issues: Sequence[str]) -> None:
    if not summary_lookup:
        lines = [
            "# Identity Preservation Report", "",
            "The analysis did not produce paper tables because the reliability gate failed.", "",
            "- Missing sample records: %d" % len(missing),
            "- Protocol issues: %d" % len(issues),
        ]
        if issues:
            lines += ["", "Issues:"] + ["- " + issue for issue in issues]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    degraded = {d: summary_lookup[("Degraded", d)]["cosine_mean"] for d in DEGRADATIONS}
    ours_gain = {d: summary_lookup[("Ours", d)]["delta_cosine_mean"] for d in DEGRADATIONS}
    hardest = min(degraded, key=degraded.get)
    easiest = max(degraded, key=degraded.get)
    max_gain = max(ours_gain, key=ours_gain.get)
    min_gain = min(ours_gain, key=ours_gain.get)
    ours = summary_lookup[("Ours", "average")]
    lines = [
        "# Identity Preservation Report", "",
        "## 1. Protocol", "",
        "- Test split: `%s`, %d sources × %d degradations = %d source-degradation queries." % (protocol["test_split"], protocol["test_source_count"], protocol["degradation_count"], protocol["test_source_count"] * protocol["degradation_count"]),
        "- Metrics are computed from the official frozen-verifier score vectors; no verifier or restoration model was trained in this analysis.",
        "- Cosine uses the paired clean source; margin uses the hardest wrong clean source from the full gallery.", "",
        "## 2. Verifier information", "",
        "- Checkpoint: `%s`" % protocol["verifier_checkpoint"],
        "- SHA-256: `%s`" % protocol["checkpoint_sha256"],
        "- Embedding dimension: `%s`; normalization: `%s`." % (protocol["embedding_dim"], protocol["normalization"]),
        "- Preprocessing: `%s`; distance: `%s`." % (json.dumps(protocol["preprocessing"], ensure_ascii=False), protocol["distance_metric"]), "",
        "## 3. Dataset integrity check", "",
        "- Required methods: %s." % ", ".join(MAIN_METHODS),
        "- Optional complete methods included in machine-readable results: %s." % (", ".join(optional_methods) if optional_methods else "none"),
        "- Missing required samples: %d; protocol issues: %d." % (len(missing), len(issues)),
        "- Gallery: %d unique test source IDs; train/test overlap: none; every query has one clean-source index." % protocol["gallery_source_count"], "",
        "## 4. Aggregate sanity check", "",
        "- A5 recomputed vs official: Top1 %.6f vs %.6f; AUC %.6f vs %.6f; EER %.6f vs %.6f." % (protocol["aggregate_sanity_check"]["recomputed"]["top1"], protocol["aggregate_sanity_check"]["official"]["top1"], protocol["aggregate_sanity_check"]["recomputed"]["auc"], protocol["aggregate_sanity_check"]["official"]["auc"], protocol["aggregate_sanity_check"]["recomputed"]["eer"], protocol["aggregate_sanity_check"]["official"]["eer"]),
        "- Protocol sanity result: `%s` (tolerance %.3f)." % ("PASS" if protocol["aggregate_sanity_check"]["passed"] else "MISMATCH", protocol["aggregate_sanity_check"]["tolerance"]), "",
        "## 5. Per-degradation Cosine", "",
        "See `table3_identity_recovery.md` and `results/identity_preservation/per_degradation_identity.csv`. Ours aggregate mean is %.6f (95%% CI [%.6f, %.6f])." % (ours["cosine_mean"], ours["cosine_ci_low"], ours["cosine_ci_high"]), "",
        "## 6. Per-degradation Margin", "",
        "Margin is positive when the paired clean source is more similar than every wrong clean source. The supplementary LaTeX table reports mean margins.", "",
        "## 7. Identity Recovery Gain", "",
        "- Ours highest gain: %s (%.6f)." % (DISPLAY_DEGRADATIONS[max_gain], ours_gain[max_gain]),
        "- Ours lowest gain: %s (%.6f)." % (DISPLAY_DEGRADATIONS[min_gain], ours_gain[min_gain]), "",
        "## 8. Relative Recovery Ratio", "",
        "The ratio is `(restored cosine - degraded cosine) / (1 - degraded cosine)` and is summarized per degradation in the JSON/CSV outputs.", "",
        "## 9. Ours vs baselines", "",
    ]
    for row in bootstrap_rows:
        if row["degradation"] == "average":
            lines.append("- %s: mean paired ΔCosine difference %.6f, 95%% CI [%.6f, %.6f], significant=`%s`." % (row["comparison"], row["mean_difference"], row["ci_low"], row["ci_high"], row["significant"]))
    lines += [
        "", "## 10. Bootstrap confidence intervals", "",
        "Source-level bootstrap: %d resamples, seed %d. Each source contributes a seven-degradation block for the Average row; no individual views are treated as independent in paired comparisons." % (protocol["bootstrap_samples"], protocol["bootstrap_seed"]), "",
        "## 11. Hardest degradation analysis", "",
        "%s has the lowest degraded own-source cosine (%.6f), so it is the hardest by the pre-restoration cosine criterion." % (DISPLAY_DEGRADATIONS[hardest], degraded[hardest]), "",
        "## 12. Easiest degradation analysis", "",
        "%s has the highest degraded own-source cosine (%.6f), so it is the easiest by the pre-restoration cosine criterion." % (DISPLAY_DEGRADATIONS[easiest], degraded[easiest]), "",
        "## 13. Paper-ready conclusions", "",
        "1. Across the seven PLAMD degradations, restoration changes source identity similarity by a measurable amount; the largest Ours gain occurs on %s (%.4f)." % (DISPLAY_DEGRADATIONS[max_gain], ours_gain[max_gain]),
        "2. The hardest and easiest degradations are data-driven: %s is lowest before restoration, while %s is highest." % (DISPLAY_DEGRADATIONS[hardest], DISPLAY_DEGRADATIONS[easiest]),
        "3. Ours is significantly better than a baseline only where the corresponding source-level paired bootstrap CI is entirely above zero; see `identity_bootstrap_ci.csv` for the exact comparisons.",
    ]
    if len(issues):
        lines += ["", "## Reliability gate", "", "Paper tables/figures were not generated because required inputs or protocol checks failed."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mismatch(path: Path, protocol: dict, issues: Sequence[str]) -> None:
    sanity = protocol.get("aggregate_sanity_check", {})
    lines = [
        "# IDENTITY_PROTOCOL_MISMATCH", "",
        "The recomputed A5 aggregate identity metrics differ from the official Table2 aggregate beyond the allowed tolerance.", "",
        "| Metric | Recomputed | Official | Absolute difference |", "|---|---:|---:|---:|",
    ]
    for metric in ("top1", "auc", "eer"):
        lines.append("| %s | %.8f | %.8f | %.8f |" % (metric, sanity["recomputed"][metric], sanity["official"][metric], sanity["absolute_difference"][metric]))
    lines += ["", "Possible causes to investigate:", "", "- gallery differs", "- split differs", "- verifier checkpoint differs", "- image preprocessing differs", "- embedding normalization differs", "- restored outputs version differs", "", "Protocol issues:"]
    lines.extend("- " + issue for issue in issues)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-main-only", action="store_true", help="do not include optional complete methods")
    args = parser.parse_args(argv)
    results_dir = ROOT / "results/identity_preservation"
    artifacts_dir = ROOT / "artifacts/identity_preservation"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        groups, source_ids, audit = load_groups()
    except Exception as exc:
        (results_dir / "IDENTITY_PROTOCOL_MISMATCH.md").write_text("# IDENTITY_PROTOCOL_MISMATCH\n\n- %s\n" % exc, encoding="utf-8")
        return 2

    missing: List[dict] = []
    metadata_by_method: Dict[str, dict] = {}
    loaded: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for method in MAIN_METHODS[1:]:
        records, metadata = load_method_scores(method, source_ids, missing)
        loaded[method] = records
        metadata_by_method[method] = metadata
    loaded["Degraded"], metadata_by_method["Degraded"] = load_method_scores("Degraded", source_ids, missing)
    metadata_by_method["Ours"] = metadata_by_method["Degraded"]
    loaded["Ours"] = loaded["Degraded"]

    # Input scores are method-independent by protocol. Compare every baseline
    # against the Degraded/A5 input score vector before using it.
    degraded_records = loaded["Degraded"]
    for method in MAIN_METHODS[1:]:
        records = loaded[method]
        for key in set(degraded_records).intersection(records):
            if not np.allclose(degraded_records[key]["input_scores"], records[key]["input_scores"], atol=1e-12, rtol=1e-12):
                missing.append({"method": method, "source_id": key[0], "degradation": key[1], "expected_path": str(result_path(method, ".test.json.sources.jsonl")), "reason": "input score vector differs from shared Degraded input"})

    issues = validate_protocol(metadata_by_method, audit, missing)
    required_keys = {(source_id, degradation) for source_id in source_ids for degradation in DEGRADATIONS}
    for method in MAIN_METHODS[1:]:
        missing_keys = required_keys.difference(loaded[method])
        if missing_keys:
            issues.append("%s is missing %d required source-degradation scores" % (method, len(missing_keys)))
    if len(degraded_records) != len(required_keys):
        issues.append("Degraded input score coverage is %d/%d" % (len(degraded_records), len(required_keys)))

    missing_fields = ["method", "source_id", "degradation", "expected_path", "reason"]
    write_csv(results_dir / "missing_identity_samples.csv", missing, missing_fields)
    required_missing_count = len(missing)
    methods_for_protocol = list(MAIN_METHODS)
    optional_methods: List[str] = []
    if not args.force_main_only:
        for method in ("DFPIR", "FFANet", "Uformer"):
            optional_missing: List[dict] = []
            optional_records, optional_metadata = load_method_scores(method, source_ids, optional_missing)
            missing.extend(optional_missing)
            if len(optional_records) == len(required_keys) and not optional_missing:
                loaded[method] = optional_records
                metadata_by_method[method] = optional_metadata
                optional_methods.append(method)
                methods_for_protocol.append(method)
    write_csv(results_dir / "missing_identity_samples.csv", missing, missing_fields)

    sanity = None
    if not issues and len(loaded["Ours"]) == len(required_keys):
        sanity = official_sanity(loaded["Ours"], metadata_by_method["Ours"], source_ids)
        if not sanity["passed"]:
            issues.append("A5 aggregate sanity check exceeded tolerance")
    else:
        sanity = {"passed": False, "tolerance": SANITY_TOLERANCE, "recomputed": {}, "official": {}, "absolute_difference": {}}

    protocol = make_protocol(metadata_by_method, source_ids, audit, sanity, methods_for_protocol, issues)
    (results_dir / "verifier_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    required_missing = missing[:required_missing_count]
    if issues or required_missing:
        write_mismatch(artifacts_dir / "IDENTITY_PROTOCOL_MISMATCH.md", protocol, issues + (["missing_identity_samples.csv contains %d rows" % len(missing)] if missing else []))
        write_report(artifacts_dir / "IDENTITY_PRESERVATION_REPORT.md", protocol, {}, [], optional_methods, missing, issues)
        print("IDENTITY_PROTOCOL_MISMATCH")
        print("issues=%d missing=%d" % (len(issues), len(missing)))
        return 2

    long_rows, summary_lookup, _arrays = build_metric_rows(loaded, source_ids)
    per_deg_fields = ["method", "degradation", "sample_count", "view_count", "cosine_mean", "cosine_std", "cosine_ci_low", "cosine_ci_high", "margin_mean", "margin_std", "margin_ci_low", "margin_ci_high", "delta_cosine_mean", "delta_cosine_std", "delta_cosine_ci_low", "delta_cosine_ci_high", "recovery_ratio_mean", "recovery_ratio_std", "recovery_ratio_ci_low", "recovery_ratio_ci_high"]
    per_deg_rows = [summary_lookup[(method, degradation)] for method in loaded for degradation in list(DEGRADATIONS) + ["average"]]
    write_csv(results_dir / "per_degradation_identity.csv", per_deg_rows, per_deg_fields)
    json_payload = {
        "protocol": protocol,
        "methods": methods_for_protocol,
        "rows": per_deg_rows,
        "per_view_metric_rows": long_rows,
        "notes": ["Cosine and margin CIs bootstrap source blocks with seed 13 and 2000 samples.", "No Top1/AUC/EER is used as a Table3 headline metric."],
    }
    (results_dir / "per_degradation_identity.json").write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    figure_rows = []
    for method in RESTORATION_METHODS:
        for degradation in DEGRADATIONS:
            row = summary_lookup[(method, degradation)]
            figure_rows.append({
                "degradation": degradation, "method": method,
                "degraded_cosine": summary_lookup[("Degraded", degradation)]["cosine_mean"],
                "restored_cosine": row["cosine_mean"], "delta_cosine": row["delta_cosine_mean"],
                "recovery_ratio": row["recovery_ratio_mean"],
            })
    write_csv(results_dir / "figure6d_identity_recovery.csv", figure_rows,
              ["degradation", "method", "degraded_cosine", "restored_cosine", "delta_cosine", "recovery_ratio"])

    bootstrap_rows = paired_rows(loaded, source_ids, summary_lookup)
    write_csv(results_dir / "identity_bootstrap_ci.csv", bootstrap_rows,
              ["comparison", "degradation", "metric", "mean_difference", "ci_low", "ci_high", "sample_count", "significant"])
    write_tables(summary_lookup, artifacts_dir)
    plot_figure(figure_rows, artifacts_dir)
    write_report(artifacts_dir / "IDENTITY_PRESERVATION_REPORT.md", protocol, summary_lookup,
                 bootstrap_rows, optional_methods, missing, issues)
    print("generated identity preservation analysis")
    print("methods=%s" % ",".join(methods_for_protocol))
    print("sources=%d degradations=%d" % (len(source_ids), len(DEGRADATIONS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
