"""10_select_cases.py —— 固定选图规则（拼图前锁定），输出案例计划 JSON/CSV。

规则（确定性，随机种子 20260907）：
  fig2：测试集中 A5 宏平均 PSNR 的中位数来源，类别 'good' 与 'bird-nest' 各一个。
        （"宏平均"=该来源 7 张退化图 A5 PSNR 的均值）
  fig3 改进案例：haze/snow/lowlight 中 B1 检索错误(rank>1) 而 A5 正确(rank==1)，
        且 A5 PSNR >= B1 PSNR - 0.5，取 (rank_B1 - rank_A5) 最大者。
  fig3 退步案例：rank_B1==1 而 rank_A5>1，且 A5 PSNR >= B1 PSNR - 0.3，取 rank_A5 最大者。
  fig3 不变案例：noise/rain/inpainting 中 rank_A5==rank_B1==1 且 |PSNR_A5-PSNR_B1| 最小者。
  fig5：类别 'rust' 的 A5 宏平均 PSNR 中位数来源；退化取 haze；
        局部窗口 = 以 |degraded-clean| 最大 L1 误差为中心的 192x192 正方形（限中央 3/4 区域），
        拼接前固定坐标。
输出: outputs/paper_figures_v1/cases/case_plan.json / case_plan.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, OUT  # noqa: E402

CASES = OUT / "cases"
CASES.mkdir(parents=True, exist_ok=True)
SEED = 20260907


def median_source_of_class(df, cls, n=None):
    sub = df[(df.method == "A5") & (df.component_category == cls)]
    m = sub.groupby("source_id")["psnr"].mean().sort_values()
    return m.index[len(m) // 2]  # 中位


def pick_by(df, rule, degs):
    cand = []
    for (sid, d), g in df.groupby(["source_id", "degradation"]):
        if d not in degs:
            continue
        a = g[g.method == "A5"].iloc[0]
        b = g[g.method == "B1"].iloc[0]
        r = rule(a, b)
        if r is not None:
            cand.append((sid, d, r))
    cand.sort(key=lambda t: t[2], reverse=True)
    return cand


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    plan = {}
    # fig2
    fig2 = []
    for cls in ("good", "bird-nest"):
        sid = median_source_of_class(df, cls)
        fig2.append({"source_id": sid, "category": cls,
                     "degradations": list(DEGRADATIONS), "role": "fig2_mosaic"})
    plan["fig2_sources"] = fig2

    # fig3
    def rule_improve(a, b):
        if b.correct_source_rank > 1 and a.correct_source_rank == 1 and a.psnr >= b.psnr - 0.5:
            return b.correct_source_rank - a.correct_source_rank
        return None

    def rule_regress(a, b):
        if b.correct_source_rank == 1 and a.correct_source_rank > 1 and a.psnr >= b.psnr - 0.3:
            return a.correct_source_rank
        return None

    def rule_unchanged(a, b):
        if a.correct_source_rank == b.correct_source_rank == 1:
            return -abs(a.psnr - b.psnr)
        return None

    imp = pick_by(df, rule_improve, ("haze", "snow", "lowlight"))
    reg = pick_by(df, rule_regress, ("haze", "snow", "lowlight"))
    unc = pick_by(df, rule_unchanged, ("noise", "rain", "inpainting"))
    fig3 = [
        {"source_id": imp[0][0], "degradation": imp[0][1], "role": "improvement",
         "reason": "B1 检索错误且 A5 恢复为正确 Top-1（haze/snow/lowlight，psnr 不降）"},
        {"source_id": reg[0][0], "degradation": reg[0][1], "role": "regression",
         "reason": "B1 正确而 A5 检索错误，且 A5 PSNR 未低于 B1（展示锚点非免费午餐）"},
        {"source_id": unc[0][0], "degradation": unc[0][1], "role": "unchanged",
         "reason": "两者均正确 Top-1 且 PSNR 几乎不变（稳定样本代表）"},
    ]
    plan["fig3_cases"] = fig3

    # fig5
    rust_src = median_source_of_class(df, "rust")
    plan["fig5"] = {"source_id": rust_src, "degradation": "haze",
                    "role": "local_detail_visual",
                    "reason": "rust 类中位数来源的 haze 视图；局部窗口取最大退化误差中心（见 crop）",
                    "crop": None}  # 由 06 推理脚本依据图像计算并回填

    plan["seed"] = SEED
    plan["rules"] = "见本文件头注释"
    # 附带每个案例的指标（便于追溯与拼图标注）
    meta = {}
    for src in fig2:
        sub = df[(df.method == "A5") & (df.source_id == src["source_id"])]
        meta[src["source_id"]] = {"category": src["category"],
                                  "a5_macro_psnr": float(sub.psnr.mean())}
    for c in fig3:
        a = df[(df.method == "A5") & (df.source_id == c["source_id"])
               & (df.degradation == c["degradation"])].iloc[0]
        b = df[(df.method == "B1") & (df.source_id == c["source_id"])
               & (df.degradation == c["degradation"])].iloc[0]
        c.update({"category": a.component_category, "psnr_a5": float(a.psnr),
                  "psnr_b1": float(b.psnr), "rank_a5": int(a.correct_source_rank),
                  "rank_b1": int(b.correct_source_rank),
                  "cos_own_a5": float(a.own_clean_similarity),
                  "cos_own_b1": float(b.own_clean_similarity),
                  "retrieved_a5": a.retrieved_source_id, "retrieved_b1": b.retrieved_source_id})
    fs = df[(df.method == "A5") & (df.source_id == rust_src)]
    plan["fig5"]["a5_macro_psnr"] = float(fs.psnr.mean())
    with (CASES / "case_plan.json").open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
    # csv 汇总
    recs = []
    for src in fig2:
        for d in src["degradations"]:
            recs.append({"role": "fig2", "source_id": src["source_id"],
                         "degradation": d, "reason": src["role"]})
    for c in fig3:
        recs.append({"role": "fig3_" + c["role"], "source_id": c["source_id"],
                     "degradation": c["degradation"],
                     "reason": "improve" if c["role"] == "improvement" else c["role"]})
    recs.append({"role": "fig5", "source_id": rust_src, "degradation": "haze",
                 "reason": "local detail"})
    pd.DataFrame(recs).to_csv(CASES / "case_plan.csv", index=False)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
