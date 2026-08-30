#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Campaign 结果自动汇总（对应 LARGE_SCALE_EXPERIMENT_DESIGN.md 第 6.4 节）。

从 results/campaigns/<name>/<model>.<seed>.<split>.json 生成：
  - 主对比表（method × PSNR/SSIM/LPIPS/top1/margin/AUC/EER/增益/intra-inter）
  - 多 seed mean±std（同 method 同 split 多 seed 时）
  - 按退化表（PSNR，val/test 各一份）
  - 输出 results/campaigns/<name>/COMPARISON_SUMMARY.md + metrics.csv

用法：
  python scripts/summary_campaign.py results/campaigns/c001_sfr_v1
  （目录内所有 *.json 评估报告都会纳入；报告必须含 verifier_fingerprint 与 aggregate）
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["psnr", "ssim", "lpips", "restored_to_clean_top1", "restored_margin",
        "restored_roc_auc", "restored_eer", "roc_auc_gain", "eer_gain",
        "identity_gain_margin", "intra_inter_output_ratio"]
HEADERS = ["method", "split", "sources", "verifier", "psnr", "ssim", "lpips", "top1",
           "margin", "AUC", "EER", "AUC_gain", "EER_gain", "margin_gain", "intra/inter"]
DEGS = ["blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow"]


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def mean_std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--min-verifier", type=str, default="", help="只汇总该 verifier sha 前缀的报告")
    args = parser.parse_args()

    reports = []
    for path in sorted(args.campaign_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "aggregate" not in report or "verifier_fingerprint" not in report:
            continue
        verifier_sha = report["verifier_fingerprint"]["sha256"][:12]
        if args.min_verifier and not verifier_sha.startswith(args.min_verifier):
            continue
        method, seed, split = path.stem.rsplit(".", 2)
        reports.append({"method": method, "seed": int(seed), "split": split,
                        "report": report, "verifier_sha": verifier_sha})

    rows = []
    grouped = defaultdict(list)
    for r in reports:
        grouped[(r["method"], r["split"])].append(r)
    for (method, split), items in sorted(grouped.items()):
        aggs = [it["report"]["aggregate"] for it in items]
        first = items[0]
        row = {"method": method, "split": split,
               "sources": first["report"]["sources"], "verifier": first["verifier_sha"]}
        multi = len(aggs) > 1
        for key in KEYS:
            values = [a.get(key) for a in aggs]
            m, s = mean_std(values)
            row[key] = f"{fmt(m)}±{fmt(s)}" if (multi and m is not None) else fmt(m)
        rows.append(row)

    rows.sort(key=lambda r: (r["split"], -float(r["psnr"].split("±")[0])))
    with (args.campaign_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADERS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in HEADERS})

    lines = [f"# Campaign 汇总：{args.campaign_dir.name}", ""]
    lines.append("自动生成自评估报告（口径：同一 verifier sha 内汇总，多 seed 显示 mean±std）")
    lines.append("")
    lines.append("## 主表")
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("|" + "---|" * len(HEADERS))
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in HEADERS) + " |")
    lines.append("")
    lines.append("## 按退化 PSNR（val / test）")
    for split in ("val", "test"):
        lines.append(f"### {split}")
        lines.append("| method | " + " | ".join(DEGS) + " |")
        lines.append("|" + "---|" * (len(DEGS) + 1))
        for (method, sp), items in sorted(grouped.items()):
            if sp != split:
                continue
            deg_psnr = []
            for it in items:
                pd = it["report"].get("per_degradation", {})
                deg_psnr.append({d: pd.get(d, {}).get("psnr") for d in DEGS})
            cells = []
            for d in DEGS:
                m, s = mean_std([p[d] for p in deg_psnr])
                cells.append(f"{fmt(m, 2)}±{fmt(s, 2)}" if (len(deg_psnr) > 1 and m is not None) else fmt(m, 2))
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        lines.append("")
    out = args.campaign_dir / "COMPARISON_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {out}（{len(rows)} 行主表）")


if __name__ == "__main__":
    main()
