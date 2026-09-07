#!/usr/bin/env python
"""build_table1.py —— 从 SiblingRestore 真实结果自动生成论文主表 Table 1。

九宫格 3x3 布局：
  (a) Deblurring  (b) Dehazing  (c) Inpainting
  (d) Low-light En. (e) Denoising (f) Deraining
  (g) Desnowing   (h) Avg Restoration Quality  (i) Source Preservation
前三行数据来自官方 v2 逐退化/宏平均评测（unified_per_image.csv）；
(i) 来自官方冻结 verifier 聚合（.test.json：Top1/EER/AUC）。
最佳=红加粗，次佳=蓝下划线；并列按展示精度分组；同一方法顺序全表一致。

输出到 outputs/paper_table1/：
  table1_data.csv / table1_plamd.tex / table1_preview.tex / table1_notes.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "paper_table1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper_figures_v1"))

from common import METHODS, ROOT as CROOT  # noqa: E402
from common import json_path  # noqa: E402

UNI = CROOT / "outputs" / "paper_figures_v1" / "unified_per_image.csv"

# ---- 方法行（顺序全表一致；Ours 最后）----
METHOD_ROWS = [
    ("airnet", "AirNet"),
    ("dfpir", "DFPIR"),
    ("dehazeformer", "DehazeFormer"),
    ("restormer", "Restormer"),
    ("promptir", "PromptIR"),
    ("ffanet", "FFANet"),
    ("transweather", "TransWeather"),
    ("uformer", "Uformer"),
    ("A5", "SiblingRestore (Ours)"),
]
OURS_ID = "A5"

# ---- 面板定义 ----
# 表头单元格（含列对齐包装）。identity 的 (%) 单位用两行堆叠以压窄列宽。
MC = lambda s: f"\\multicolumn{{1}}{{c}}{{{s}}}"
METRIC_HEAD = ["Method",
               MC("PSNR $\\uparrow$"), MC("SSIM $\\uparrow$"), MC("LPIPS $\\downarrow$")]
ID_HEAD = ["Method",
           MC("\\shortstack{Top-1 $\\uparrow$\\\\ (\\%)}"),
           MC("\\shortstack{EER $\\downarrow$\\\\ (\\%)}"),
           MC("AUC $\\uparrow$")]
PANELS = [
    ("a", "Deblurring", "blur"),
    ("b", "Dehazing", "haze"),
    ("c", "Inpainting", "inpainting"),
    ("d", "Low-light Enhancement", "lowlight"),
    ("e", "Denoising", "noise"),
    ("f", "Deraining", "rain"),
    ("g", "Desnowing", "snow"),
]
PANEL_H = ("h", "Average Restoration Quality", None)   # macro
PANEL_I = ("i", "Source Preservation", None)            # identity


def fmt_psnr(v):  return f"{v:.2f}"
def fmt_ssim(v):  return f"{v:.4f}"
def fmt_lpips(v): return f"{v:.4f}"
def fmt_pct(v):   return f"{v*100:.2f}"
def fmt_auc(v):   return f"{v:.4f}"


def quality_panels(df: pd.DataFrame) -> dict:
    """panel -> method -> [psnr, ssim, lpips] 展示字符串 + 全精度 dict"""
    g = df.groupby(["method", "degradation"])[["psnr", "ssim", "lpips"]].mean()
    out = {}
    for tag, _, deg in PANELS:
        out[tag] = {}
        for mid, _ in METHOD_ROWS:
            r = g.loc[(mid, deg)]
            out[tag][mid] = {
                "psnr": float(r["psnr"]), "ssim": float(r["ssim"]), "lpips": float(r["lpips"]),
            }
    # macro panel h: 每退化均值后 7 类等权
    out[PANEL_H[0]] = {}
    for mid, _ in METHOD_ROWS:
        per = g.loc[mid]
        out[PANEL_H[0]][mid] = {
            "psnr": float(per["psnr"].mean()),
            "ssim": float(per["ssim"].mean()),
            "lpips": float(per["lpips"].mean()),
        }
    return out


def identity_panel(df: pd.DataFrame) -> dict:
    mm = {m["id"]: m for m in METHODS}
    out = {}
    for mid, _ in METHOD_ROWS:
        sub = df[df.method == mid]
        agg = json.loads(json_path(mm[mid]).read_text())["aggregate"]
        out[mid] = {"top1": float(sub.top1_correct.mean()),
                    "eer": float(agg["restored_eer"]),
                    "auc": float(agg["restored_roc_auc"])}
    return out


# ---------------- 排名着色 ----------------
def rank_cells(values: list[float], better_larger: bool, fmt):
    """按展示精度判定并列；返回每格样式：'best'/'second'/''。"""
    shown = [fmt(v) for v in values]
    # 稳定分组：不同底层值但同展示 → 同一显示组
    order = sorted(set(shown), reverse=better_larger)
    best_g, second_g = order[0], (order[1] if len(order) > 1 else None)
    return [("best" if s == best_g else ("second" if s == second_g else "")) for s in shown]


def decorate(fmt_val: str, style: str) -> str:
    if style == "best":
        return f"\\PTbest{{{fmt_val}}}"
    if style == "second":
        return f"\\PTsecond{{{fmt_val}}}"
    return fmt_val


# ---------------- LaTeX ----------------
def panel_table(tag: str, title: str, head, rows_cells) -> str:
    """rows_cells: list[(method_label, is_ours, cells:[str])] 每格已含装饰"""
    lines = ["\\begin{tabular}{@{}l@{\\hspace{3.2pt}}r@{\\hspace{3.2pt}}r@{\\hspace{3.2pt}}r@{}}", "\\toprule",
             f"\\multicolumn{{4}}{{c}}{{\\textbf{{({tag}) {title}}}}}\\\\",
             "\\cmidrule(lr){1-4}",
             " & ".join(head) + "\\\\",
             "\\midrule"]
    for label, is_ours, cells in rows_cells:
        lab = f"\\textbf{{{label}}}" if is_ours else label
        lines.append(" & ".join([lab] + cells) + "\\\\")
        if is_ours:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def build_tex(quality: dict, identity: dict) -> tuple[str, str]:
    # 生成九个面板 rows_cells（共用同一方法顺序）
    def rows_for(panel_data, spec):
        """spec: [(key, fmt, better_larger)]"""
        rows_cells = []
        for mid, label in METHOD_ROWS:
            vals = [panel_data[mid][k] for k, _, _ in spec]
            styles = []
            for i, (k, fmt, bl) in enumerate(spec):
                styles.append(rank_cells([panel_data[m][k] for m, _ in METHOD_ROWS], bl, fmt))
            cells = [decorate(fmt(panel_data[mid][spec[i][0]]), styles[i][[m for m, _ in METHOD_ROWS].index(mid)])
                     for i in range(len(spec))]
            rows_cells.append((label, mid == OURS_ID, cells))
        return rows_cells

    panels_tex = {}
    # a-g 与 h 用统一 metric spec
    metric_spec = [("psnr", fmt_psnr, True), ("ssim", fmt_ssim, True), ("lpips", fmt_lpips, False)]
    for tag, title, _ in PANELS:
        panels_tex[tag] = panel_table(tag, title, METRIC_HEAD,
                                      rows_for(quality[tag], metric_spec))
    panels_tex[PANEL_H[0]] = panel_table(PANEL_H[0], PANEL_H[1], METRIC_HEAD,
                                         rows_for(quality[PANEL_H[0]], metric_spec))
    id_spec = [("top1", fmt_pct, True), ("eer", fmt_pct, False), ("auc", fmt_auc, True)]
    panels_tex[PANEL_I[0]] = panel_table(PANEL_I[0], PANEL_I[1], ID_HEAD,
                                         rows_for(identity, id_spec))

    def box(tabular: str) -> str:
        return ("\\begin{minipage}[t]{0.322\\textwidth}\\centering\n"
                + tabular + "\n\\end{minipage}")

    row1 = "\\hspace{3pt}".join([box(panels_tex[k]) for k in "abc"])
    row2 = "\\hspace{3pt}".join([box(panels_tex[k]) for k in "def"])
    row3 = "\\hspace{3pt}".join([box(panels_tex[k]) for k in ("g", "h", "i")])

    body = f"""
% ---- Table 1 (auto-generated by build_table1.py; do not edit numbers by hand) ----
\\providecommand{{\\PTbest}}[1]{{\\color{{PTbest}}\\bfseries #1}}
\\providecommand{{\\PTsecond}}[1]{{\\color{{PTsecond}}\\underline{{#1}}}}
\\definecolor{{PTbest}}{{RGB}}{{198,12,12}}
\\definecolor{{PTsecond}}{{RGB}}{{0,80,190}}
\\begin{{table*}}[t]
\\centering
\\begingroup\\footnotesize
\\setlength{{\\tabcolsep}}{{0pt}}
\\caption{{Quantitative comparison on PLAMD under the all-in-one setting.
Panels (a)--(g) report restoration performance on seven degradation types
(blur, haze, inpainting, low-light, noise, rain, snow); panel (h) reports their
macro-average; panel (i) evaluates source preservation with an independent frozen
verifier. Best and second-best results are in red and blue, respectively.}}
\\vspace{{2pt}}
\\noindent
{row1}
\\par\\vspace{{7pt}}
\\noindent
{row2}
\\par\\vspace{{7pt}}
\\noindent
{row3}
\\par\\vspace{{4pt}}
\\begin{{minipage}}{{\\textwidth}}\\raggedright\\scriptsize
Panel (h) macro-average = equal-weight mean of the seven per-degradation means
(each degradation contributes equally; class sizes are equal on the test set).
Panel (i): Top-1 = fraction of restored views whose clean-source is the top
retrieval in the full 256-source gallery; EER/AUC are aggregated over the same
per-view cosine scores against that gallery. Percent columns report\\ \\%
values. All numbers originate from official frozen-verifier evaluations on the
same 256-source test split (all methods share one verifier and one LPIPS
protocol; PSNR in dB, LPIPS AlexNet 256$\\times$256). Missing results would be
shown as ``--''; no method in this table lacks any degradation. Training
protocols: all listed methods were trained on the same mixed seven-degradation
dataset (all-in-one); siblingrestore (Ours, config A5 = deg$+$fixed-anchor) used
sibling-group sampling and 20k steps, external baselines used independent
sampling and 12k steps, so results are not claimed to come from a fully
controlled common retraining budget. Single training seed (13) per method.
Noise images have native sizes differing from clean; evaluation aligns clean to
noise by Lanczos resize before scoring (project Method-B protocol).
\\end{{minipage}}
\\endgroup
\\end{{table*}}"""
    plamd = body.strip() + "\n"

    preview = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=210mm,paperheight=297mm,left=15mm,right=15mm,top=18mm,bottom=18mm]{geometry}
\usepackage{newtxtext,newtxmath}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}

""" + plamd + r"""
\end{document}
"""
    return plamd, preview


# ---------------- data.csv ----------------
def write_data_csv(quality, identity, df: pd.DataFrame) -> None:
    recs = []
    mm = {m["id"]: m for m in METHODS}
    # 指纹来源：paper_figures_v1 清单缓存 + 官方 json config 指纹
    sha_cache = json.loads((CROOT / "outputs/paper_figures_v1/checkpoint_shas.json").read_text())
    ver_sha = json.loads(json_path(mm["A5"]).read_text())["verifier_fingerprint"]["sha256"]
    def ck_sha(mid):
        return sha_cache.get(str((CROOT / mm[mid]["ckpt"]).resolve()), "")[:12]
    def cfg_sha(mid):
        return json.loads(json_path(mm[mid]).read_text())["config_fingerprint"]["sha256"]
    src_prefix = {
        "psnr": "results/... .test.csv (official v2) / unified_per_image.csv",
        "ssim": "same", "lpips": "same",
        "top1": "official .test.json aggregate", "eer": "official .test.json aggregate",
        "auc": "official .test.json aggregate",
    }
    for tag, title, deg in PANELS:
        for mid, label in METHOD_ROWS:
            v = quality[tag][mid]
            recs.append({"panel": f"({tag})", "task": title, "degradation": deg,
                         "method_id": mid, "method_label": label,
                         "psnr": v["psnr"], "ssim": v["ssim"], "lpips": v["lpips"],
                         "top1_pct": "", "eer_pct": "", "auc": "",
                         "seed": 13, "checkpoint_fingerprint_12": ck_sha(mid), "config_fingerprint": cfg_sha(mid),
                         "verifier_sha256": ver_sha, "source": src_prefix["psnr"]})
    for mid, label in METHOD_ROWS:
        v = quality[PANEL_H[0]][mid]
        recs.append({"panel": f"({PANEL_H[0]})", "task": PANEL_H[1],
                     "degradation": "macro7", "method_id": mid, "method_label": label,
                     "psnr": v["psnr"], "ssim": v["ssim"], "lpips": v["lpips"],
                     "top1_pct": "", "eer_pct": "", "auc": "",
                     "seed": 13, "checkpoint_fingerprint_12": ck_sha(mid), "config_fingerprint": cfg_sha(mid),
                     "verifier_sha256": ver_sha, "source": "macro of 7 per-degradation means"})
    for mid, label in METHOD_ROWS:
        v = identity[mid]
        recs.append({"panel": f"({PANEL_I[0]})", "task": PANEL_I[1], "degradation": "identity",
                     "method_id": mid, "method_label": label,
                     "psnr": "", "ssim": "", "lpips": "",
                     "top1_pct": v["top1"] * 100, "eer_pct": v["eer"] * 100, "auc": v["auc"],
                     "seed": 13, "checkpoint_fingerprint_12": ck_sha(mid), "config_fingerprint": cfg_sha(mid),
                     "verifier_sha256": ver_sha, "source": src_prefix["top1"]})
    data = pd.DataFrame(recs)
    data.to_csv(OUT / "table1_data.csv", index=False, float_format="%.6f")
    print("wrote table1_data.csv rows", len(data))


def write_notes() -> None:
    notes = """# Table 1 notes（模型映射 / 协议 / 缺失 / 排名 / 复现）

## 方法行与模型映射（以配置与结果指纹为准）
| 显示名 | 内部结果名 | 说明 |
|---|---|---|
| AirNet | airnet | c005 官方组 |
| DFPIR | dfpir | c005 官方组 |
| DehazeFormer | dehazeformer | c005 官方组 |
| Restormer | restormer | c005 官方组 |
| PromptIR | promptir | c005 官方组 |
| FFANet | ffanet | c005 官方组 |
| TransWeather | transweather | c005 官方组 |
| Uformer | uformer | c005 官方组 |
| **SiblingRestore (Ours)** | **A5**（= A5_no_sibling，rec+grad+deg+fixed-anchor 0.01，dim48） | ablation_core |

- A5 指纹：config sha `5f2f096d42ac`；checkpoint/verifier 指纹见 outputs/paper_figures_v1/asset_manifest 与 PROTOCOL。
- B1 等内部消融配置不放入主表（归消融表）。

## 评测协议（全部复用官方 v2 产物，未重新评测）
- 测试集：PLAMD plamd_sfr_v1 test = 256 sources × 7 退化 = 1792 恢复视图（source 级互斥）。
- 同一冻结 verifier（sha d9b000…，256 clean 全库检索）；LPIPS alex_256x256；PSNR/SSIM 官方 masked 实现。
- noise 尺寸对齐：clean Lanczos 对齐到 noise 原生尺寸（Method-B 协议）。
- 方法均在七类混合数据上 all-in-one 训练；未做测试时自集成，不加相关标记。
- 训练预算：Ours(ablation lineage)=20k steps、mode=sibling 同源分组；c005 外部基线=12k steps、independent。同一评测口径但训练预算/采样不一致 → 表题不声称公平重训练。
- 种子：全部单 seed=13，报告单次结果，不伪造均值/标准差。

## 面板 (h) 宏平均
先计算每退化在该方法上的均值（256 视图），再对七类等权平均。所有行七类均完整。

## 面板 (i) 来源保持
- Top-1：恢复视图经冻结 verifier 嵌入后在 256 clean 图库中首位命中自身 clean 来源的比例。
- EER/AUC：来源验证二分类（同源=正、跨源=负）下官方聚合的等错误率 / ROC-AUC。
- 该块是"恢复图→原始 clean 图"的来源保持检索，不作为跨视角设备身份识别表述。

## 缺失与占比格式
- 表格全单元格均有真实数值；若某方法缺少某退化类别会显示 "—" 且不参与面板 (h) 排名，本轮无此情况。
- Top-1 / EER 单元格为百分数数值（表头注明 %），无缺失补零或引用其他数据集数值。

## 排名样式
- 每子表每指标列独立排名：PSNR/SSIM/Top-1/AUC 越大越好，LPIPS/EER 越小越好。
- 最佳=红加粗（PTbest），次佳=蓝下划线（PTsecond），并列按展示精度分组判定（全精度存于 table1_data.csv）。
- Ours 行仅方法名加粗；数值颜色只按真实排名，不整行标红。

## 复现
```
.venv/bin/python analysis/paper_table1/build_table1.py   # 生成 CSV/tex
tectonic -X compile outputs/paper_table1/table1_preview.tex  # 或 pdflatex
pdftoppm -r 300 -png outputs/paper_table1/table1_preview.pdf outputs/paper_table1/table1_preview
```
"""
    (OUT / "table1_notes.md").write_text(notes, encoding="utf-8")
    print("wrote table1_notes.md")


def main() -> None:
    df = pd.read_csv(UNI)
    quality = quality_panels(df)
    identity = identity_panel(df)
    plamd, preview = build_tex(quality, identity)
    (OUT / "table1_plamd.tex").write_text(plamd, encoding="utf-8")
    (OUT / "table1_preview.tex").write_text(preview, encoding="utf-8")
    write_data_csv(quality, identity, df)
    write_notes()
    print("wrote table1_plamd.tex / table1_preview.tex")


if __name__ == "__main__":
    main()
