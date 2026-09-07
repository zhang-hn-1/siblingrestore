"""08_consistency_check.py —— 交付前一致性核对。

核对项：
 1) 逐图行数/来源数/退化覆盖/每来源视图数；
 2) 输入侧指标应与方法无关（同一批退化输入）；
 3) 统一表均值 vs 官方 .test.json aggregate 与 per_degradation；
 4) 表 1 宏平均与官方 aggregate 的一致性；
 5) rank/top1/top5 逻辑（top1<=top5，top1 == rank==1，同类别候选>=5）；
 6) 图 1 计数总和 == 视图数/来源数；图 4 统计完整性；
 7) case_plan 中每个案例都能在统一表找到且样本图存在；
 8) 缺少值统计（restored_path 恒空是已知缺失，其余列应无缺失）。
输出: outputs/paper_figures_v1/consistency_report.md（控制台显示结论）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, METHODS, OUT, json_path  # noqa: E402

issues: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str):
    (notes if cond else issues).append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    df["restored_path"] = df["restored_path"].fillna("")  # 已知缺失列：空串标记
    # 1) coverage
    per = df.groupby("method").size()
    check((per == 1792).all(), "每个方法 1792 行")
    nsrc = df.groupby("method").source_id.nunique()
    check((nsrc == 256).all(), "每个方法 256 个来源")
    degset = df.groupby("method").degradation.apply(lambda s: set(s))
    check(all(s == set(DEGRADATIONS) for s in degset), "每个方法覆盖 7 类退化")
    vc = df.groupby(["method", "source_id"]).size()
    check((vc == 7).all(), "每来源 7 视图")
    # 2) input-side method-independent
    spread = df.groupby("method")["input_top1_correct"].mean()
    check(float(spread.max() - spread.min()) < 1e-12, "退化输入检索与方法无关")
    # 3) vs official
    worst = 0.0
    for m in METHODS:
        res = json.loads(json_path(m).read_text())
        a = res["aggregate"]
        sub = df[df.method == m["id"]]
        for col, key in [("psnr", "psnr"), ("ssim", "ssim"), ("lpips", "lpips"),
                         ("top1_correct", "restored_to_clean_top1"),
                         ("own_clean_similarity", "restored_own_anchor_cos"),
                         ("nearest_wrong_similarity", "restored_nearest_impostor_cos"),
                         ("restored_margin", "restored_margin")]:
            worst = max(worst, abs(float(sub[col].mean()) - float(a[key])))
        for d in DEGRADATIONS:
            pdg = res["per_degradation"][d]
            for col, key in [("psnr", "psnr"), ("ssim", "ssim"), ("lpips", "lpips")]:
                worst = max(worst, abs(float(sub[sub.degradation == d][col].mean()) - float(pdg[key])))
            worst = max(worst, abs(float(sub[sub.degradation == d].top1_correct.mean())
                                   - float(pdg["restored_to_clean_top1"])))
    check(worst < 1e-6, f"统一表均值与官方 json 最大偏差 {worst:.2e} < 1e-6")
    # 4) table1 macro consistency vs aggregate
    mm = df.groupby(["method", "degradation"])["psnr"].mean().groupby("method").mean()
    agg_psnr = {m["id"]: json.loads(json_path(m).read_text())["aggregate"]["psnr"] for m in METHODS}
    dmax = max(abs(mm[mid] - agg_psnr[mid]) for mid in agg_psnr)
    check(dmax < 1e-6, f"宏平均 PSNR 与官方 aggregate 最大偏差 {dmax:.2e}")
    # 5) logical consistency
    check((df.top1_correct <= df.top5_correct).all(), "top1<=top5 全部满足")
    rank1 = (df.correct_source_rank == 1).astype(int)
    check((rank1 == df.top1_correct).all(), "rank==1 与 top1_correct 完全一致")
    samecat_n = df.groupby("component_category").size()  # not needed
    cat_counts = df[df.method == "A5"].groupby("component_category").source_id.nunique()
    check((cat_counts >= 5).all(), f"同类别候选均>=5：{cat_counts.to_dict()}")
    check((df["input_path"].str.len() > 0).all() and (df["clean_path"].str.len() > 0).all(),
          "input_path / clean_path 全部非空")
    # restored_path 恒空是已知缺失
    notes.append("EXPECTED restored_path 全部为空（项目未落盘恢复图；图 2/3/5 已补推理到 sample_restored/）")
    # 6) fig1 counts
    cnt = pd.read_csv(OUT / "cases/fig1_view_delta_counts.csv", index_col="degradation")
    total = cnt.drop(index="all")["n_views"].sum()
    check(total == 1792, f"图1 逐退化视图计数总和 = {total}（应为 1792）")
    catcols = [c for c in cnt.columns if c not in ("n_views", "mean_dpsnr", "mean_drank")]
    rowsum = cnt.drop(index="all")[catcols].sum(axis=1)
    check((rowsum == cnt.drop(index="all")["n_views"]).all(), "图1 分类计数合计等于视图数")
    src_cnt = pd.read_csv(OUT / "cases/fig1_source_delta_counts.csv")
    check(int(src_cnt["n_sources"].iloc[0]) == 256, "图1 每源聚合样本数=256")
    # fig4
    st = pd.read_csv(OUT / "cases/fig4_stability_stats.csv")
    check((st["n_full_sources"] == 256).all(), "图4 完整七类来源数=256")
    check((st["hist"].apply(lambda h: sum(map(int, h.strip('[]').split(','))) == 256)).all(),
          "图4 直方图计数合计=256")
    # 7) case plan
    plan = json.loads((OUT / "cases/case_plan.json").read_text())
    for src in plan["fig2_sources"]:
        sub = df[(df.method == "A5") & (df.source_id == src["source_id"])]
        check(len(sub) == 7, f"fig2 来源 {src['source_id'][:30]}... 7 视图齐备")
    for c in plan["fig3_cases"]:
        row = df[(df.method == "A5") & (df.source_id == c["source_id"])
                 & (df.degradation == c["degradation"])]
        check(len(row) == 1, f"fig3 案例 {c['role']} 可在统一表找到")
        for mid in ("A5", "B1"):
            p = OUT / "sample_restored" / mid / f"{c['source_id'].replace('/', '_').replace(' ', '_')}__{c['degradation']}.png"
            check(p.exists(), f"fig3 案例 {c['role']} 恢复图存在 ({mid})")
    f5 = plan["fig5"]
    check(f5.get("crop") is not None, "fig5 crop 坐标已确定")
    # 8) missing values (除 known missing 列外不应有 NaN)
    df_no_missing_col = df.drop(columns=["restored_path"])
    miss = df_no_missing_col.isna().sum()
    check(int(miss.sum()) == 0, f"除 restored_path 外无缺失值（NaN 分布：{miss[miss>0].to_dict()}）")

    md = ["# 一致性核对报告", ""]
    md += [f"- {n}" for n in notes]
    md += ["", "## 未通过项"] + (["- " + m for m in issues] if issues else ["- 无"])
    (OUT / "consistency_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n== issues ==", issues if issues else "NONE")


if __name__ == "__main__":
    main()
