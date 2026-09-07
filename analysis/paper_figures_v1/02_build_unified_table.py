"""02_build_unified_table.py —— 建立统一逐图结果表。

数据源：各方法官方 v2 评测产物（scripts/evaluate_frozen_verifier_v2.py）
  - *.test.csv            : 逐退化图 PSNR/SSIM/LPIPS 与检索余弦/命中（1792 行）
  - *.test.json.sources.jsonl : 每 view 对 256 anchor 的全 score 向量
  - data/cache/anchors/*_test_*.pt : anchor 库顺序（== test source_groups 顺序）
  - data/plamd_sfr_v1/metadata/source_groups.json : 路径与 subcategory

对每个方法×每张测试退化图，补算官方文件没有直接保存的：
  correct_source_rank / top5 / retrieved_source_id / 同类别(same subcategory) top1 与 rank
（输入侧同样补算 input_*，供表 2 的退化输入对照）。
并列处理：与官方 argmax 一致 —— 相等分数按更小 index 优先。

输出: outputs/paper_figures_v1/unified_per_image.csv（所有方法全量行）
      outputs/paper_figures_v1/unified_schema.md（字段与协议说明）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEGRADATIONS, METHODS, OUT, ROOT, VERIFIER_SHA,  # noqa: E402
                    csv_path, json_path, sources_path)

FIELDS = [
    "method", "cohort", "seed", "source_id", "sample_id", "degradation",
    "component_category", "input_path", "clean_path", "restored_path",
    "checkpoint_fingerprint", "verifier_fingerprint", "evaluation_protocol",
    # 恢复侧
    "psnr", "ssim", "lpips",
    "own_clean_similarity", "nearest_wrong_similarity", "correct_source_rank",
    "top1_correct", "top5_correct", "samecat_top1_correct", "samecat_correct_source_rank",
    "retrieved_source_id", "restored_margin",
    # 退化输入侧
    "input_own_clean_similarity", "input_nearest_wrong_similarity", "input_correct_source_rank",
    "input_top1_correct", "input_top5_correct", "input_samecat_top1_correct",
    "input_retrieved_source_id", "input_margin",
]

PROTOCOL = (
    "eval_v2 test(256src/1792views) verifier=%s lpips=alex_256x256 "
    "psnr/ssim=masked_official noise=clean_lanczos_to_noise "
    "retrieval=cos_over_256_test_clean_anchors incl_self tie=first_index t512_o32" % VERIFIER_SHA)


def load_groups() -> tuple[dict, dict]:
    groups = json.loads((ROOT / "data/plamd_sfr_v1/metadata/source_groups.json").read_text())
    test = [g for g in groups if g["split"] == "test"]
    meta = {}
    for g in test:
        meta[g["source_id"]] = g
    return meta, test


def resolve(p: str) -> str:
    """metadata 中路径相对 data/plamd_sfr_v1，解析为相对项目根的绝对路径字符串。"""
    full = (ROOT / "data/plamd_sfr_v1" / p).resolve()
    return str(full)


def main() -> None:
    meta, test = load_groups()
    bank_ids = [g["source_id"] for g in test]  # == anchor 库顺序
    own_idx = {sid: i for i, sid in enumerate(bank_ids)}
    cat_of = {g["source_id"]: g["subcategory"] for g in test}
    cat_members: dict[str, list[int]] = {}
    for i, sid in enumerate(bank_ids):
        cat_members.setdefault(cat_of[sid], []).append(i)
    for c, inds in cat_members.items():
        assert len(inds) >= 5, f"类别 {c} 候选不足 5，无法报告常规 Top5/排名：{len(inds)}"

    out_rows = []
    nrows_total = 0
    for m in METHODS:
        # checkpoint fingerprint from manifest cache
        sha_map_path = OUT / "checkpoint_shas.json"
        sha_map = json.loads(sha_map_path.read_text())
        ckpt_abs = str((ROOT / m["ckpt"]).resolve())
        ckpt_sha = sha_map.get(ckpt_abs, "")
        # per-image 行（官方 csv）
        with open(csv_path(m), newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows_by_key = {(r["source_id"], r["degradation"]): r for r in reader}
        # 汇总 json 指纹
        res = json.loads(json_path(m).read_text())
        vf = res.get("verifier_fingerprint", {}).get("sha256", "")
        assert vf == VERIFIER_SHA, f"{m['id']} verifier {vf} 与统一口径不一致"
        cfg_fp = res.get("config_fingerprint", {}).get("sha256", "")
        sample_id_of = {d: i for i, d in enumerate(DEGRADATIONS)}

        # 逐 source 补算 rank/top5/检索来源
        # 说明：个别方法（如 M3_alcrb）因断点续跑在 .sources.jsonl 里对少数 source
        # 写了重复行；官方 consume() 按 source_id 取最后一次覆盖，这里保持一致。
        with open(sources_path(m), encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        last_index = {}
        for i, ln in enumerate(lines):
            last_index[json.loads(ln)["source_id"]] = i
        for i, ln in enumerate(lines):
            rec = json.loads(ln)
            if i != last_index[rec["source_id"]]:
                continue
            sid = rec["source_id"]
            g = meta[sid]
            oi = own_idx[sid]
            my_cat = cat_of[sid]
            cat_inds = cat_members[my_cat]
            cat_pos = {idx: j for j, idx in enumerate(cat_inds)}
            for view in rec["views"]:
                deg = view["degradation"]
                row = rows_by_key[(sid, deg)]
                s_rest = np.asarray(view["restored_scores"], dtype=np.float64)
                s_inp = np.asarray(view["input_scores"], dtype=np.float64)

                def der(scores, own):
                    rk = int(1 + np.sum(scores > scores[own]) +
                              np.sum((scores == scores[own]) & (np.arange(len(scores)) < own)))
                    argmax = int(np.argmax(scores))
                    top5 = rk <= 5
                    # 同类别内（含自身）排名与 top1
                    sub = scores[cat_inds]
                    rk_cat = int(1 + np.sum(sub > sub[cat_pos[own]]) +
                                  np.sum((sub == sub[cat_pos[own]]) &
                                         (np.arange(len(sub)) < cat_pos[own])))
                    return {
                        "rank": rk, "top1": rk == 1, "top5": top5,
                        "rank_cat": rk_cat, "top1_cat": rk_cat == 1,
                        "own_sim": float(scores[own]),
                        "nearest_wrong": float(np.max(np.delete(scores, own))),
                        "retrieved": bank_ids[argmax],
                        "margin": float(scores[own] - np.max(np.delete(scores, own))),
                    }

                r = der(s_rest, oi)
                ri = der(s_inp, oi)
                out_rows.append({
                    "method": m["id"], "cohort": m["cohort"],
                    "seed": row.get("seed", "13"), "source_id": sid,
                    "sample_id": sample_id_of[deg], "degradation": deg,
                    "component_category": g["subcategory"],
                    "input_path": resolve(g["degraded"][deg]),
                    "clean_path": resolve(g["clean_path"]),
                    "restored_path": "",  # 项目未落盘恢复图
                    "checkpoint_fingerprint": ckpt_sha or cfg_fp,
                    "verifier_fingerprint": vf, "evaluation_protocol": PROTOCOL,
                    "psnr": float(row["psnr"]), "ssim": float(row["ssim"]),
                    "lpips": float(row["lpips"]),
                    "own_clean_similarity": r["own_sim"],
                    "nearest_wrong_similarity": r["nearest_wrong"],
                    "correct_source_rank": r["rank"],
                    "top1_correct": int(r["top1"]), "top5_correct": int(r["top5"]),
                    "samecat_top1_correct": int(r["top1_cat"]),
                    "samecat_correct_source_rank": r["rank_cat"],
                    "retrieved_source_id": r["retrieved"],
                    "restored_margin": r["margin"],
                    "input_own_clean_similarity": ri["own_sim"],
                    "input_nearest_wrong_similarity": ri["nearest_wrong"],
                    "input_correct_source_rank": ri["rank"],
                    "input_top1_correct": int(ri["top1"]),
                    "input_top5_correct": int(ri["top5"]),
                    "input_samecat_top1_correct": int(ri["top1_cat"]),
                    "input_retrieved_source_id": ri["retrieved"],
                    "input_margin": ri["margin"],
                })
        nrows_method = len(last_index)
        print(f"{m['id']:24s} rows={nrows_method} done", flush=True)

    # write unified CSV
    out_csv = OUT / "unified_per_image.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWROTE {out_csv} rows={len(out_rows)}")

    # schema / protocol markdown
    (OUT / "unified_schema.md").write_text(_schema_md(), encoding="utf-8")


def _schema_md() -> str:
    return "\n".join([
        "# 统一逐图结果表字段说明", "",
        "## evaluation_protocol（所有行相同）", "", "```", PROTOCOL, "```", "",
        "## 字段", "",
        "| 字段 | 含义 | 来源 |",
        "|---|---|---|",
        "| method | 方法 ID（见表 catalog） | |",
        "| cohort | 训练批次（ablation_core=20k sibling 采样；c005=12k independent 官方对比；psnr_modules_12k/psnr_safe_refine=12k 主线） | |",
        "| source_id | 测试来源 ID（vari-grip/<class>/<stem>） | source_groups.json |",
        "| sample_id | 0..6 对应 blur,haze,inpainting,lowlight,noise,rain,snow | |",
        "| degradation | 退化类型 | |",
        "| component_category | 数据包 subcategory：good/rust/bird-nest（设备部件状态类，非观测到的缺陷标注） | |",
        "| input_path / clean_path | 退化图/对应 clean 图绝对路径 | metadata 解析 |",
        "| restored_path | 恢复图路径：项目未落盘恢复图，恒为空（缺失项） | |",
        "| checkpoint_fingerprint | best.pt sha256（全 64 位）或 config 指纹兜底 | 本清单 sha |",
        "| verifier_fingerprint | 评测 frozen verifier sha256 | .test.json |",
        "| psnr/ssim/lpips | 官方 v2 masked PSNR / masked SSIM / LPIPS alex_256x256 | .test.csv |",
        "| own_clean_similarity | 恢复图 embedding 与自身 clean anchor 余弦 | sources scores |",
        "| nearest_wrong_similarity | 恢复图与最近错误 anchor（排除自身）余弦 | sources scores |",
        "| correct_source_rank | 正确来源在 256 全库中的检索排名（并列按更小 index 优先，与官方 argmax 一致） | 由全 score 向量重算 |",
        "| top1_correct / top5_correct | 正确来源命中 Top1 / Top5（0/1） | 重算 |",
        "| samecat_top1_correct / samecat_correct_source_rank | 同 subcategory 候选库内的 Top1/排名（候选>=5） | 重算 |",
        "| retrieved_source_id | 检索返回的最高相似来源 | 重算 |",
        "| restored_margin | own_sim − nearest_wrong | 重算（等于官方 margin） |",
        "| input_* | 退化输入 embedding 的对应指标（表 2 退化输入对照行） | 重算 |",
    ])


if __name__ == "__main__":
    main()
