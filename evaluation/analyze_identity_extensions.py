#!/usr/bin/env python3
"""Offline identity cosine and source-level bootstrap analysis.

This script only reads existing frozen-verifier evaluation artifacts. It never
loads or modifies a restoration checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")
METHODS = {
    "A5": Path("results/ablation_core/A5_no_sibling.13"),
    "Restormer": Path("results/campaigns/c005_official_group2/restormer.13"),
    "DehazeFormer": Path("results/campaigns/c005_official_group1/dehazeformer.13"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0, len(values))


def bootstrap(values_by_source: dict[str, float], seed: int, rounds: int = 10000):
    items = [float(v) for v in values_by_source.values() if math.isfinite(float(v))]
    if not items:
        return {"mean": None, "lower95": None, "upper95": None, "num_sources": 0}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(items, k=len(items))) for _ in range(rounds))
    lo = samples[int((rounds - 1) * 0.025)]
    hi = samples[int((rounds - 1) * 0.975)]
    return {"mean": statistics.mean(items), "lower95": lo, "upper95": hi, "num_sources": len(items)}


def source_view_means(rows, key):
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(float(row[key]))
    return {source: statistics.mean(values) for source, values in by_source.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/identity_evaluation"))
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    args = parser.parse_args()
    root = args.artifact_root
    cosine_dir = root / "identity_cosine"
    bootstrap_dir = root / "bootstrap"
    figures_dir = root / "paper_figures"
    for directory in (cosine_dir, bootstrap_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    method_rows = {}
    method_sources = {}
    for method, stem in METHODS.items():
        csv_path = Path(str(stem) + ".test.csv")
        source_path = Path(str(stem) + ".test.json.sources.jsonl")
        method_rows[method] = read_csv(csv_path)
        method_sources[method] = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]

    cosine_rows = []
    gain_rows = []
    for method, rows in method_rows.items():
        for degradation in DEGRADATIONS:
            selected = [r for r in rows if r["degradation"] == degradation]
            add = lambda key, _name: mean_std([float(r[key]) for r in selected])
            degraded_mean, degraded_std, count = add("input_own_anchor_cos", "degraded")
            restored_mean, restored_std, _ = add("restored_own_anchor_cos", "restored")
            gain = restored_mean - degraded_mean
            cosine_rows.append({"method": method, "degradation": degradation, "degraded_cosine": degraded_mean, "degraded_std": degraded_std, "restored_cosine": restored_mean, "restored_std": restored_std, "gain": gain, "num_samples": count})
            gain_rows.append({"method": method, "degradation": degradation, "degraded_cosine": degraded_mean, "restored_cosine": restored_mean, "gain": gain, "num_samples": count})

    for filename, rows in (("identity_cosine_per_degradation.csv", cosine_rows), ("identity_recovery_gain.csv", gain_rows)):
        with (cosine_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    # Source-level paired bootstrap for all core metrics.
    metrics = {
        "psnr": "psnr", "ssim": "ssim", "lpips": "lpips",
        "restored_to_clean_top1": "restored_to_clean_top1",
        "restored_margin": "restored_margin", "identity_gain_margin": "identity_gain_margin",
    }
    bootstrap_rows = []
    for comparison, left, right in (("A5_vs_Restormer", "A5", "Restormer"), ("A5_vs_DehazeFormer", "A5", "DehazeFormer")):
        for metric, key in metrics.items():
            left_by_source = source_view_means(method_rows[left], key)
            right_by_source = source_view_means(method_rows[right], key)
            deltas = {source: left_by_source[source] - right_by_source[source] for source in left_by_source.keys() & right_by_source.keys()}
            result = bootstrap(deltas, seed=13 + len(bootstrap_rows), rounds=args.bootstrap_rounds)
            result.update({"comparison": comparison, "metric": metric, "bootstrap_unit": "source_id", "bootstrap_iterations": args.bootstrap_rounds})
            bootstrap_rows.append(result)
    with (bootstrap_dir / "bootstrap_significance_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0])); writer.writeheader(); writer.writerows(bootstrap_rows)

    report = ["# Identity Evaluation Extensions", "", "- Evaluation split: SFR v1 test (256 sources, 1792 views)", "- Bootstrap unit: `source_id`", f"- Bootstrap iterations: `{args.bootstrap_rounds}`", "- Restoration inference rerun: **no**", "", "## Per-Degradation Identity Cosine", "", "| Method | Degradation | Degraded Cosine | Restored Cosine | Gain | N |", "|---|---|---:|---:|---:|---:|"]
    for row in gain_rows:
        report.append(f"| {row['method']} | {row['degradation']} | {row['degraded_cosine']:.6f} | {row['restored_cosine']:.6f} | {row['gain']:.6f} | {row['num_samples']} |")
    report += ["", "## Source-Level Bootstrap", "", "| Comparison | Metric | Mean Delta | Lower 95% | Upper 95% | Sources |", "|---|---|---:|---:|---:|---:|"]
    for row in bootstrap_rows:
        report.append(f"| {row['comparison']} | {row['metric']} | {row['mean']:.6f} | {row['lower95']:.6f} | {row['upper95']:.6f} | {row['num_sources']} |")
    (root / "IDENTITY_COSINE_BOOTSTRAP_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"artifact_root": str(root), "methods": list(METHODS), "cosine_rows": len(cosine_rows), "bootstrap_rows": len(bootstrap_rows)}, indent=2))


if __name__ == "__main__":
    main()
