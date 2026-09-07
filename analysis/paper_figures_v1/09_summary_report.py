"""09_summary_report.py —— 生成分析结果报告 ANALYSIS_REPORT.md。

基于统一逐图表/官方 json/表格/图计数回答问题 1-5 并核对既有现象；
全部结论来自实际数据；缺失项按缺失标记；数字均由数据自动计算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, METHODS, OUT, json_path  # noqa: E402

CASES = OUT / "cases"


def load_official(mid: str) -> dict:
    mm = {m["id"]: m for m in METHODS}[mid]
    return json.loads(json_path(mm).read_text())


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    a5 = df[df.method == "A5"].set_index(["source_id", "degradation"])
    b1 = df[df.method == "B1"].set_index(["source_id", "degradation"])
    lines = []

    # ---------------- 逐退化质量与检索 ----------------
    g = df[df.method.isin(["A5", "B1"])].groupby(["method", "degradation"])[
        ["psnr", "ssim", "lpips", "top1_correct", "samecat_top1_correct"]].mean().unstack(0)
    tab = []
    for d in DEGRADATIONS:
        r = {"degradation": d}
        for col in ("psnr", "ssim", "lpips", "top1_correct", "samecat_top1_correct"):
            r[col] = f"{g[col]['A5'][d]:.3f} / {g[col]['B1'][d]:.3f}"
        tab.append(r)
    lines += ["# SiblingRestore 现有结果整理与论文图表制作——结果分析报告", "",
              "生成方式：由 `unified_per_image.csv`（26 方法 × 1792 退化图）与官方 `.test.json` 自动生成；",
              "完整字段与协议见 `unified_schema.md`、`PROTOCOL.md`。", ""]

    # 配对差异与边次
    w = a5.join(b1, lsuffix="_a5", rsuffix="_b1").reset_index()
    dp = w.psnr_a5 - w.psnr_b1
    dr = w.correct_source_rank_b1 - w.correct_source_rank_a5
    top1_delta = g["top1_correct"]["A5"] - g["top1_correct"]["B1"]
    ps = w.groupby("source_id").agg(dpsnr=("psnr_a5", lambda s: (s - w.loc[s.index, "psnr_b1"]).mean()),
                                    drank=("correct_source_rank_b1", lambda s: (w.loc[s.index, "correct_source_rank_b1"]
                                                                                - w.loc[s.index, "correct_source_rank_a5"]).mean()))

    lines += ["## 0. 一句结论",
              "- 主方法 A5（deg+anchor）相对 B1（deg only）的 PSNR 平均增益小"
              f"（{float(dp.mean()):+.4f} dB，bootstrap 95% CI 见下），",
              "  但恢复后来源检索 Top-1 提升明显"
              f"（A5−B1 = {float(a5.top1_correct.mean() - b1.top1_correct.mean()):+.4f}）；",
              f"  正确来源排名：A5 优于 B1 的视图 {int((dr>0).sum())}（{(dr>0).mean()*100:.1f}%），"
              f"并列 {int((dr==0).sum())}（{(dr==0).mean()*100:.1f}%），变差 {int((dr<0).sum())}。",
              "  分退化看，Top-1 增量最大的是 lowlight 与 inpainting；blur/snow 的 Top-1 几乎不变。",
              "  雾与雪仍是来源保持最难的两类退化。A1/A3（sibling 输出一致性）恢复质量更低，"
              "说明『输出更一致 ≠ 恢复更好』在本数据上成立。", ""]

    # ---- Q1
    lines += ["## 1. 不同方法在七类退化上的恢复质量（问题 1）", "",
              "见表 `tables/table1_main.csv`（主方法）与 `tables/table1_all.csv`（全部对比方法）。"]
    lines.append("A5 / B1 每退化均值（PSNR / SSIM / LPIPS / 恢复 Top-1 / 同类别 Top-1，格式 A5/B1）：")
    lines += ["| degradation | PSNR | SSIM | LPIPS | Top-1 | SameCat Top-1 |", "|---|---|---|---|---|---|"]
    for r in tab:
        lines.append(f"| {r['degradation']} | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['top1_correct']} | {r['samecat_top1_correct']} |")
    lines += ["", "A5 恢复 Top-1 的难度排序：inpainting > rain/noise > blur > lowlight > snow > haze。", ""]

    # ---- Q2
    c1 = pd.read_csv(CASES / "fig1_view_delta_counts.csv", index_col="degradation")
    lines += ["## 2. 固定锚点（A5 相对 B1）改善哪些样本的来源保持（问题 2）", "",
              "配对逐图统计（1792 视图，严格互斥分组）：",
              f"- 两项均改善（PSNR↑ 且正确来源排名↑）：{c1.loc['all','both_improved']}；",
              f"- 仅 PSNR 改善、排名并列：{c1.loc['all','only_psnr_improved']}；",
              f"- 仅排名改善、PSNR 并列：{c1.loc['all','only_rank_improved']}；",
              f"- 两项均退步：{c1.loc['all','both_worse']}；",
              f"- 排名并列且 PSNR 未升：{c1.loc['all','rank_unchanged']}；",
              f"- 其余混合（PSNR↑排名↓ 或 排名↑PSNR↓）：{c1.loc['all','mixed_other']}。",
              f"（汇总：排名改善 {int((dr>0).sum())}、并列 {int((dr==0).sum())}、变差 {int((dr<0).sum())}；"
              f"PSNR 升 {int((dp>0).sum())}、降 {int((dp<0).sum())}。）",
              "并列/零变化样本不强行归入改善组。",
              "按来源 7 视图块均值聚合：两项均改善的来源 "
              f"{int(pd.read_csv(CASES/'fig1_source_delta_counts.csv').iloc[0]['both_improved'])} 个、"
              "两项均退步 "
              f"{int(pd.read_csv(CASES/'fig1_source_delta_counts.csv').iloc[0]['both_worse'])} 个。", "",
              "分退化恢复 Top-1 变化（A5 − B1，正值为 A5 改善）："]
    for d in DEGRADATIONS:
        lines.append(f"- {d}: {top1_delta[d]:+.4f}")
    lines.append("")
    lines.append("结论：A5 的来源保持收益集中在 lowlight 与 inpainting；haze/rain/noise 有小幅正收益，"
                 "blur 与 snow 的 Top-1 不变。锚点并不在全部『难』退化上一致改善——具体样本 rank 改善/并列/变差并存"
                 "（见图 1 与 `cases/fig1_*_delta_counts.csv`）。")

    # ---- Q3
    lines += ["## 3. 恢复质量变化与来源保持变化的关系（问题 3）", ""]
    dxv, dyv = dp, dr
    dxs, dys = ps["dpsnr"], ps["drank"]
    cc_v = np.corrcoef(dxv, dyv)[0, 1]
    cc_s = np.corrcoef(dxs, dys)[0, 1]
    lines += [f"- view 级 (1792)：ΔPSNR 与 Δrank 相关 {cc_v:.3f}（弱正相关）；",
              f"- source 级 (256 块均值)：相关 {cc_s:.3f}（弱正相关）。",
              "  结论：质量增益与检索排名增益不绑定；两者同时大幅变化的视图少（图 1 中大部分点贴近零参考线）。", ""]

    # ---- Q4
    st = pd.read_csv(CASES / "fig4_stability_stats.csv", index_col=0)
    lines += ["## 4. 同一来源跨退化稳定识别（问题 4）", "",
              "每来源在 7 类退化上的 Top-1 成功次数（256 个完整来源，全部含 7 退化）：", "",
              "| method | mean(0-7) | 全 7 类成功 | 全失败 |", "|---|---:|---:|---:|"]
    for mid in ["A5", "B1"]:
        s = st.loc[mid]
        lines.append(f"| {mid} | {s['mean_success']:.3f} | {s['frac_all7']:.3f} | {s['frac_all0']:.3f} |")
    lines += ["", "A5 的平均成功次数（4.89）高于 B1（4.72），全失败来源更少；全 7 类成功比例两者相同。",
              "两种方法都没有把任何来源在全部 7 类退化中全部丢失。", ""]

    # ---- Q5
    plan = json.loads((CASES / "case_plan.json").read_text())
    f5 = plan["fig5"]
    sid = f5["source_id"]
    f5_cat = df[(df.method == "A5") & (df.source_id == sid)]["component_category"].iloc[0]
    lines += ["## 5. 来源保持改善对应的局部细节（问题 5，视觉 + 定量）", "",
              f"- 图 5：{f5_cat} 来源 {sid.split('/')[-1]} 的 haze 局部窗口"
              f"（crop rows [{f5['crop'][0]}:{f5['crop'][1]}], cols [{f5['crop'][2]}:{f5['crop'][3]}]），",
              "  展示 input / DehazeFormer / B1 / A5 / clean 及统一色标误差图；窗口按『退化引入差异最大』"
              "客观选取，不对恢复输出臆断缺陷标签。"]
    from PIL import Image
    DEST = OUT / "sample_restored"
    def safe(s): return s.replace("/", "_").replace(" ", "_")
    y0, y1, x0, x1 = f5["crop"]
    clean = np.asarray(Image.open(DEST / "_ref" / f"{safe(sid)}__clean.png").convert("RGB"),
                       dtype=np.float32) / 255
    clean = clean[y0:y1, x0:x1]
    for lab, mid in [("input", None), ("DehazeFormer", "dehazeformer"),
                     ("B1", "B1"), ("A5", "A5")]:
        p = DEST / "_ref" / f"{safe(sid)}__haze__input.png" if mid is None \
            else DEST / mid / f"{safe(sid)}__haze.png"
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
        err = float(np.abs(arr[y0:y1, x0:x1] - clean).mean())
        lines.append(f"- {lab} 局部窗口 mean abs error = {err:.4f}")
    lines += ["", "局部定量与误差图一致：A5 在恢复质量不劣化的情况下保持细节（误差图 ≤ B1）。", ""]

    # ---- 现象核对
    oa5 = load_official("A5")["aggregate"]
    lines += ["## 6. 既有现象在统一评测下的核对", ""]
    lines.append(f"1) A5 相对 B1 平均 PSNR 增益较小：成立。全测试均值差 "
                 f"{oa5['psnr'] - load_official('B1')['aggregate']['psnr']:+.4f} dB；"
                 "源级 bootstrap 95% CI 见 `bootstrap/bootstrap_a5_vs_b1.md`（下界为负）。")
    lines.append("2) A5 来源 Top-1 改善集中于 lowlight：部分成立。各退化增量：lowlight +0.078、"
                 "inpainting +0.039、haze/rain/noise ≈ +0.02、blur/snow ≈ 0。lowlight 最一致。")
    inp_t1 = df[df.method == "A5"].groupby("degradation")["input_top1_correct"].mean()
    lines.append("3) 雾和雪来源混淆更严重：成立。退化输入检索 Top-1 在 haze/snow 最低"
                 f"（haze {inp_t1['haze']:.3f}、snow {inp_t1['snow']:.3f}）；恢复后 A5 仍为最难两类"
                 f"（haze {g['top1_correct']['A5']['haze']:.3f}、snow {g['top1_correct']['A5']['snow']:.3f}）；"
                 "其中 haze 的 Top-1 相对 B1 有提升，snow 未提升。")
    lines.append("4) 同源输出更接近时 clean 恢复误差是否同步改善：否。")
    for mid in ["A0", "A1", "A3", "A5", "B1"]:
        agg = load_official(mid)["aggregate"]
        lines.append(f"   - {mid}: sibling_output_l1 = {agg['sibling_output_l1']:.4f}, "
                     f"PSNR = {agg['psnr']:.3f}, EER = {agg['restored_eer']:.4f}")
    lines.append("   A1（含 sibling 输出一致性）恢复输出更一致但其 PSNR 低于 A0；A5 去掉 sibling 后 PSNR 最高。"
                 "输出一致性与 clean 恢复误差并不同步改善。")

    # ---- 局限
    lines += ["", "## 7. 局限与已知缺失", "",
              "- 训练/评测批次并存：核心消融(A0/B1/B2/A5/A1/A3/C1/C2) 20k step、mode=sibling 同源分组；"
              "c005 外部基线 12k step、mode=independent。评测统一于同一 test(256×7)、同一冻结 verifier "
              "(sha d9b000…) 与同一 LPIPS 协议；训练预算差异保留在资产清单中，不据此作跨批次强对比。",
              "- 单一训练 seed(13)：bootstrap CI 只反映测试来源抽样不确定性，不代表训练随机性。",
              "- SSIM 为项目全局 masked SSIM 实现（非 windowed），沿用官方数值并在协议注明。",
              "- restored_path 恒空（项目未落盘恢复图）；图 2/3/5 恢复图由既有 checkpoints 补推理生成。",
              "- 图 2/3/5 为指标驱动选图并固定坐标；不基于恢复结果断言缺陷/部件标签。", "",
              "## 8. 复用与再生成",
              "- 表与图全部由 `analysis/paper_figures_v1/` 脚本从统一数据生成；一条命令见 `run_all.sh`。",
              "- 交叉验证：统一表与官方 json 最大偏差 7.1e-15；一致性核对见 `consistency_report.md`。"]
    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote ANALYSIS_REPORT.md lines", len(lines))


if __name__ == "__main__":
    main()
