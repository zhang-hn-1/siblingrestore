"""01_assets_manifest.py —— 资产清单与可用性报告。

对每个方法记录：结果 JSON/CSV/sources.jsonl 是否齐全、checkpoint 存在性、
checkpoint 大小/修改时间/sha256、config 指纹、verifier 指纹、训练元信息、
恢复图是否落盘、聚合指标（来自官方 .test.json）。
输出: outputs/paper_figures_v1/asset_manifest.csv / asset_manifest.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ANALYSIS, METHODS, OUT, VERIFIER_SHA, csv_path, json_path,  # noqa: E402
                    means_dir, sources_path)

CHECK_SHA_CACHE = OUT / "checkpoint_shas.json"


def sha256_file(path: Path) -> str:
    if CHECK_SHA_CACHE.exists():
        cache = json.loads(CHECK_SHA_CACHE.read_text())
        if str(path) in cache:
            return cache[str(path)]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CHECK_SHA_CACHE.exists():
        cache = json.loads(CHECK_SHA_CACHE.read_text())
    rows = []
    for m in METHODS:
        ckpt = Path(m["ckpt"])
        if not (ANALYSIS.parent.parent / ckpt).exists():
            ckpt_rel = None
        else:
            ckpt_rel = ckpt
        if ckpt_rel is not None:
            cp = ANALYSIS.parent.parent / ckpt_rel
            stat = cp.stat()
            sha = sha256_file(cp)
            cache[str(cp)] = sha
            ckpt_bytes = stat.st_size
            ckpt_mtime = stat.st_mtime
        else:
            sha = ckpt_bytes = ckpt_mtime = None
        res = json.loads(json_path(m).read_text()) if json_path(m).exists() else {}
        json_ok = json_path(m).exists()
        csv_ok = csv_path(m).exists()
        src_ok = sources_path(m).exists()
        means_ok = means_dir(m).exists() and any(means_dir(m).iterdir())
        agg = res.get("aggregate", {})
        cfg_hash = res.get("config_fingerprint", {}).get("sha256")
        vf = res.get("verifier_fingerprint", {}).get("sha256")
        rows.append({
            "method": m["id"], "cohort": m["cohort"], "model_family": m["model_family"],
            "config_note": m["config_note"], "group": m["group"] or "",
            "checkpoint_path": str(ckpt_rel) if ckpt_rel else "",
            "checkpoint_exists": bool(ckpt_rel), "checkpoint_bytes": ckpt_bytes,
            "checkpoint_sha256": sha or "",
            "config_fingerprint": cfg_hash or "",
            "verifier_fingerprint": vf or "",
            "result_json": json_ok, "per_image_csv": csv_ok,
            "sources_scores_jsonl": src_ok, "anchor_means": means_ok,
            "restored_images_saved": False,
            "aggregate_psnr": agg.get("psnr"), "aggregate_ssim": agg.get("ssim"),
            "aggregate_lpips": agg.get("lpips"),
            "aggregate_restored_top1": agg.get("restored_to_clean_top1"),
            "aggregate_eer": agg.get("restored_eer"),
            "aggregate_auc": agg.get("restored_roc_auc"),
        })
        if ckpt_rel is not None and not json_ok:
            print("note: json missing", m["id"])
    with (OUT / "asset_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    CHECK_SHA_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"wrote asset_manifest.csv with {len(rows)} methods")

    # markdown report
    lines = ["# 资产清单与可用性报告（paper_figures_v1）", "",
             f"生成时间见文件 mtime；协议：test 256 sources × 7 退化 = 1792 views；",
             f"verifier sha256 `{VERIFIER_SHA}`；LPIPS alex_256x256；",
             "恢复图实物（restored images）全项目均未落盘 → `restored_images_saved=False`，",
             "图 2/3/5 需要恢复图的环节将用已有 checkpoint 补跑少量推理。", "",
             "| method | cohort | 训练元信息 | ckpt | 汇总json | 逐图csv | sources分 | ckpt_sha256(前12) | PSNR | Top1 | EER |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["cohort"], x["method"])):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | `{}` | {:.4f} | {:.4f} | {:.4f} |".format(
            r["method"], r["cohort"], (r["config_note"] or "").replace("|", "\\|"),
            "Y" if r["checkpoint_exists"] else "-",
            "Y" if r["result_json"] else "-", "Y" if r["per_image_csv"] else "-",
            "Y" if r["sources_scores_jsonl"] else "-",
            r["checkpoint_sha256"][:12] if r["checkpoint_sha256"] else "",
            r["aggregate_psnr"] or float("nan"), r["aggregate_restored_top1"] or float("nan"),
            r["aggregate_eer"] or float("nan")))
    (OUT / "asset_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote asset_manifest.md")


if __name__ == "__main__":
    main()
