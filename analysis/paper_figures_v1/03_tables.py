"""03_tables.py —— 生成论文表 1/2/3 及补充表（CSV + LaTeX）。

表 1 七类退化恢复主表        : PSNR/SSIM 每退化 + 宏平均 PSNR/SSIM/LPIPS
表 2 来源保持主表            : 全库 Top1/Top5、同类别 Top1、EER、AUC（含退化输入对照行）
表 3 核心消融表              : A0/B1/B2/A5（采样方式/损失配置/质量/身份 + 七类退化分解）
补充表                      : A1/A3/A2/C1/C2（名称与真实配置一致）
效率记录                    : results/model_metrics.json 汇总（供图 6）

宏平均 = 7 类各自均值再等权平均（各类视图数一致时等于总体均值，此处保留形式定义）。
LaTeX：最优值用 textbf 命令、次优用 underline 命令，方向按指标自动处理；数值全部自动生成。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, MAIN_METHODS, METHODS, OUT, ROOT, json_path  # noqa: E402

TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DEG_LABELS = {  # tex 标题中的短名
    "blur": "Blur", "haze": "Haze", "inpainting": "Inpainting", "lowlight": "Low-light",
    "noise": "Noise", "rain": "Rain", "snow": "Snow",
}

# ---------------- 配置/显示名 ----------------
# method 顺序（表内行序）：ours 之外按论文习惯放入；此处标注友好名
def method_title(mid: str) -> str:
    if mid == "A5":
        return "Ours (A5)"
    if mid.startswith("ours_"):
        return mid  # dim48/dim64 家族名保持真实
    return mid

LOSS_TEXT = {  # 消融表"实际启用损失"
    "A0": r"rec+grad",
    "A1": r"rec+grad+$L_{\mathrm{sib}}$",
    "A2": r"rec+grad+$L_{\mathrm{deg}}$",
    "A3": r"rec+grad+$L_{\mathrm{deg}}$+$L_{\mathrm{anc}}$+$L_{\mathrm{sib}}$",
    "A5": r"rec+grad+$L_{\mathrm{deg}}$+$L_{\mathrm{anc}}$",
    "B1": r"rec+grad+$L_{\mathrm{deg}}$",
    "B2": r"rec+grad+$L_{\mathrm{anc}}$",
    "C1": r"rec+grad+$L_{\mathrm{anc}}^{\mathrm{adapt}}$",
    "C2": r"rec+grad+$L_{\mathrm{deg}}$+$L_{\mathrm{anc}}^{\mathrm{adapt}}$",
}
LOSS_TEXT_CSV = {
    "A0": "rec+grad", "A1": "rec+grad+sibling(out0.01)", "A2": "rec+grad+deg",
    "A3": "rec+grad+deg+anchor+sibling", "A5": "rec+grad+deg+anchor",
    "B1": "rec+grad+deg", "B2": "rec+grad+anchor",
    "C1": "rec+grad+adaptive_anchor", "C2": "rec+grad+deg+adaptive_anchor",
}

# 表 1 / 表 2 的方法集合
T1_MAIN = ["A5", "dehazeformer", "restormer", "promptir"]
T1_ALL = ["A5", "dehazeformer", "restormer", "promptir", "ours_dim48_anchor", "ours_dim64_anchor",
          "airnet", "clearair", "dfpir", "ffanet", "grl", "prenet", "r2r",
          "swinir", "transweather", "uformer"]

LOWER_BETTER = {"lpips", "eer", "input_eer", "restored_eer"}


def _degradation_means(df: pd.DataFrame) -> pd.DataFrame:
    """method x degradation 聚合（每退化 256 视图的均值）。"""
    g = df.groupby(["method", "degradation"], as_index=False)[
        ["psnr", "ssim", "lpips", "top1_correct", "top5_correct",
         "samecat_top1_correct"]].mean()
    return g


def macro_row(df: pd.DataFrame, method: str) -> dict:
    """宏平均 = 先每退化均值再七类等权平均。"""
    sub = df[df.method == method]
    per = sub.groupby("degradation")[["psnr", "ssim", "lpips"]].mean()
    return {k: float(per[k].mean()) for k in ("psnr", "ssim", "lpips")}


def _fmt_tex(v, nd=2):
    return f"{v:.{nd}f}"


def _best_flags(vals: list[float], higher_better: bool):
    """返回 (best_idx, second_idx)，严格比较；并列取第一个。"""
    order = sorted(vals, reverse=higher_better)
    best, second = order[0], (order[1] if len(order) > 1 else None)
    b_idx = vals.index(best)
    s_idx = None
    if second is not None and len(vals) > 1:
        s_idx = vals.index(second) if second != best else None
        if s_idx == b_idx:
            s_idx = None
    return b_idx, s_idx


def _decorate(v: float, nd, higher_better, rank_in_col):
    txt = _fmt_tex(v, nd)
    if rank_in_col == 0:
        return f"\\textbf{{{txt}}}"
    if rank_in_col == 1:
        return f"\\underline{{{txt}}}"
    return txt


def write_table1(df: pd.DataFrame, methods: list[str], tag: str) -> None:
    """表 1：每退化 PSNR/SSIM + 宏平均。"""
    deg_means = _degradation_means(df)
    rows = []
    for mid in methods:
        row = {"method": mid}
        mm = deg_means[deg_means.method == mid].set_index("degradation")
        for d in DEGRADATIONS:
            row[f"psnr_{d}"] = float(mm.loc[d, "psnr"])
            row[f"ssim_{d}"] = float(mm.loc[d, "ssim"])
        mr = macro_row(df, mid)
        row.update({f"macro_{k}": mr[k] for k in ("psnr", "ssim", "lpips")})
        rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLE_DIR / f"table1_{tag}.csv", index=False)

    # ---- LaTeX ----
    ncols = 1 + 2 * len(DEGRADATIONS) + 3
    colspec = "l" + "c" * (ncols - 1)
    tex = [f"% Table 1 ({tag}) — 自动生成，禁止手改数值",
           "\\begin{table*}[t]", "\\centering",
           "\\caption{Seven-degradation restoration quality (test, 256 sources). "
           "Per-degradation mean over 256 views. Macro: equal-weight mean over the seven "
           "degradation means. Protocol: v2 masked PSNR/SSIM, LPIPS AlexNet 256$\\times$256. "
           "Bold=best, underline=second. Train budget differs across cohorts "
           "(see asset manifest).}",
           "\\begin{tabular}{%s}" % colspec, "\\toprule"]
    # header rows
    head1 = ["Method"] + [f"\\multicolumn{{2}}{{c}}{{{DEG_LABELS[d]}}}" for d in DEGRADATIONS] \
            + ["\\multicolumn{3}{c}{Macro}"]
    head2 = [""] + ["PSNR", "SSIM"] * len(DEGRADATIONS) + ["PSNR", "SSIM", "LPIPS"]
    tex.append(" & ".join(head1) + " \\\\")
    tex.append(" & ".join(head2) + " \\\\")
    tex.append("\\midrule")
    # 数值列（决定 bold/underline 顺序），每列一列值
    col_keys = [f"psnr_{d}" for d in DEGRADATIONS] + [f"ssim_{d}" for d in DEGRADATIONS] \
               + ["macro_psnr", "macro_ssim", "macro_lpips"]
    higher = [True] * len(DEGRADATIONS) + [True] * len(DEGRADATIONS) + [True, True, False]
    for i, (ck, hb) in enumerate(zip(col_keys, higher)):
        vals = [tbl.loc[j, ck] for j in range(len(tbl))]
        b, s = _best_flags(vals, hb)
        col_dec = 2 if ck.startswith("psnr") else 4
        rank_map = {j: (0 if j == b else (1 if j == s else None)) for j in range(len(tbl))}
        tbl[f"__fmt_{i}"] = [rank_map[j] for j in range(len(tbl))]
        tbl[f"__dec_{i}"] = col_dec
    for j in range(len(tbl)):
        cells = [tbl.loc[j, "method"]]
        for i, ck in enumerate(col_keys):
            nd = tbl.loc[j, f"__dec_{i}"]
            rk = tbl.loc[j, f"__fmt_{i}"]
            cells.append(_decorate(tbl.loc[j, ck], nd, higher[i], rk))
        tex.append(" & ".join(cells) + " \\\\")
    tex.append("\\bottomrule")
    tex += ["\\end{tabular}", "\\end{table*}"]
    (TABLE_DIR / f"table1_{tag}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("table1", tag, "rows", len(tbl))


def write_table2(df: pd.DataFrame, methods: list[str], tag: str) -> None:
    """表 2：来源保持。含退化输入参考行 + 各方法恢复行。"""
    meta = {m["id"]: m for m in METHODS}
    rows = []
    # input row 取同一退化输入（与方法无关）
    inp = df[df.method == "A5"].copy()
    res_json = json.loads(json_path(meta["A5"]).read_text())["aggregate"]
    rows.append({
        "row_type": "degraded_input", "method": "Degraded input",
        "cohort": "all", "top1": inp.input_top1_correct.mean(),
        "top5": inp.input_top5_correct.mean(),
        "samecat_top1": inp.input_samecat_top1_correct.mean(),
        "eer": res_json["input_eer"], "auc": res_json["input_roc_auc"]})
    for mid in methods:
        sub = df[df.method == mid]
        res_json = json.loads(json_path(meta[mid]).read_text())["aggregate"]
        rows.append({
            "row_type": "restored", "method": mid,
            "cohort": meta[mid]["cohort"],
            "top1": sub.top1_correct.mean(), "top5": sub.top5_correct.mean(),
            "samecat_top1": sub.samecat_top1_correct.mean(),
            "eer": res_json["restored_eer"], "auc": res_json["restored_roc_auc"]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLE_DIR / f"table2_{tag}.csv", index=False)

    # LaTeX：仅对 restored 行做最优/次优；input 行单独参考。
    keys = ["top1", "top5", "samecat_top1", "eer", "auc"]
    higher = [True, True, True, False, True]
    fmt_map = {"top1": 4, "top5": 4, "samecat_top1": 4, "eer": 4, "auc": 4}
    rank_cols = {}
    restored_idx = [i for i, r in enumerate(rows) if r["row_type"] == "restored"]
    for ck, hb in zip(keys, higher):
        vals = [rows[i][ck] for i in restored_idx]
        b, s = _best_flags(vals, hb)
        rm = {j: (0 if j == b else (1 if j == s else None)) for j in restored_idx}
        rank_cols[ck] = rm
    tex = ["% Table 2 source identity retention (auto-generated)",
           "\\begin{table*}[t]", "\\centering",
           "\\caption{Source-identity retention on the test set (full clean gallery of 256 "
           "sources; same-subcategory candidate set in parentheses). Degraded-input row is "
           "method independent and included as reference. EER/AUC from the official frozen-"
           "verifier aggregate. Bold=best among restored rows, underline=second.}",
           "\\begin{tabular}{lcccccc}", "\\toprule",
           "Method", "Cohort", "Top-1", "Top-5", "SameCat Top-1", "EER$\\downarrow$", "AUC$\\uparrow$",
           "\\\\", "\\midrule"]
    for i, r in enumerate(rows):
        cells = [r["method"], r["cohort"]]
        for ck in keys:
            v = r[ck]
            rank = rank_cols.get(ck, {}).get(i)
            txt = _decorate(v, 4, higher[keys.index(ck)], rank)
            cells.append(txt)
        tex.append(" & ".join(cells) + " \\\\")
        if r["row_type"] == "degraded_input":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TABLE_DIR / f"table2_{tag}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("table2", tag, "rows", len(tbl))


def write_table3(df: pd.DataFrame, mids: list[str], tag: str, core: bool) -> None:
    """表 3：消融。mids 顺序即行序。"""
    meta = {m["id"]: m for m in METHODS}
    deg_means = _degradation_means(df)
    rows = []
    for mid in mids:
        sub = df[df.method == mid]
        agg = json.loads(json_path(meta[mid]).read_text())["aggregate"]
        row = {"method": mid, "sampling": "sibling-group(2 deg/src)",
               "losses": LOSS_TEXT_CSV[mid],
               "psnr": sub.psnr.mean(), "ssim": sub.ssim.mean(),
               "lpips": sub.lpips.mean(), "top1": sub.top1_correct.mean(),
               "eer": agg["restored_eer"], "auc": agg["restored_roc_auc"]}
        rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLE_DIR / f"table3_{tag}.csv", index=False)
    # per-degradation breakdown
    parts = []
    for mid in mids:
        mm = deg_means[deg_means.method == mid].set_index("degradation")
        for d in DEGRADATIONS:
            parts.append({"method": mid, "degradation": d,
                          "psnr": float(mm.loc[d, "psnr"]), "ssim": float(mm.loc[d, "ssim"]),
                          "lpips": float(mm.loc[d, "lpips"]),
                          "top1": float(mm.loc[d, "top1_correct"])})
    pd.DataFrame(parts).to_csv(TABLE_DIR / f"table3_{tag}_perdegradation.csv", index=False)

    # LaTeX main summary
    keys = ["psnr", "ssim", "lpips", "top1", "eer", "auc"]
    higher = [True, True, False, True, False, True]
    nmap = {"psnr": "PSNR$\\uparrow$", "ssim": "SSIM$\\uparrow$", "lpips": "LPIPS$\\downarrow$",
            "top1": "Top-1$\\uparrow$", "eer": "EER$\\downarrow$", "auc": "AUC$\\uparrow$"}
    dec = {"psnr": 2, "ssim": 4, "lpips": 4, "top1": 4, "eer": 4, "auc": 4}
    rank_cols = {}
    for ck, hb in zip(keys, higher):
        vals = [tbl.loc[j, ck] for j in range(len(tbl))]
        b, s = _best_flags(vals, hb)
        rank_cols[ck] = {j: (0 if j == b else (1 if j == s else None)) for j in range(len(tbl))}
    tex = ["% Table 3 ablation (auto-generated)",
           "\\begin{table}[t]", "\\centering",
           "\\caption{Core ablation on the 2$\\times$2 design (Deg $\\times$ Anchor). "
           "Sampling: sibling grouping (2 same-source degraded views per group). "
           "All models share the same dim-48 backbone, crop 256, seed 13, 20k steps "
           "(mode=sibling). Bold=best, underline=second.}",
           "\\begin{tabular}{ll" + "c" * len(keys) + "}", "\\toprule",
           "Config & Enabled losses & " + " & ".join(nmap[k] for k in keys) + "\\\\",
           "\\midrule"]
    for j in range(len(tbl)):
        cells = [tbl.loc[j, "method"], LOSS_TEXT[tbl.loc[j, "method"]]]
        for ck in keys:
            cells.append(_decorate(tbl.loc[j, ck], dec[ck], higher[keys.index(ck)],
                                   rank_cols[ck][j]))
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / f"table3_{tag}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("table3", tag, "rows", len(tbl))


def write_supplementary(df: pd.DataFrame) -> None:
    """补充消融表：A1/A3/A2/C1/C2（真实配置名）。"""
    mids = ["A1", "A2", "A3", "C1", "C2"]
    meta = {m["id"]: m for m in METHODS}
    rows = []
    for mid in mids:
        sub = df[df.method == mid]
        cfg = json.loads((ROOT / "configs/ablation_core").glob(mid + "_*.json").__next__().read_text())
        agg = json.loads(json_path(meta[mid]).read_text())["aggregate"]
        rows.append({"method": mid, "losses": LOSS_TEXT_CSV[mid],
                     "sampling": "sibling-group", "max_steps": 20000,
                     "anchor_type": (cfg.get("anchor_config") or {}).get("type", "none"),
                     "psnr": sub.psnr.mean(), "ssim": sub.ssim.mean(),
                     "lpips": sub.lpips.mean(), "top1": sub.top1_correct.mean(),
                     "eer": agg["restored_eer"]})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_supplementary_ablation.csv", index=False)
    keys = ["psnr", "ssim", "lpips", "top1", "eer"]
    higher = [True, True, False, True, False]
    dec = {"psnr": 2, "ssim": 4, "lpips": 4, "top1": 4, "eer": 4}
    nmap = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "top1": "Top-1", "eer": "EER"}
    rank_cols = {}
    for ck, hb in zip(keys, higher):
        vals = [rows[i][ck] for i in range(len(rows))]
        b, s = _best_flags(vals, hb)
        rank_cols[ck] = {i: (0 if i == b else (1 if i == s else None)) for i in range(len(rows))}
    tex = ["% Supplementary ablation (auto-generated)",
           "\\begin{table}[t]", "\\centering",
           "\\caption{Supplementary configs with real training settings "
           "(names unchanged from config files). A1/A3 contain the sibling "
           "output-consistency loss; C1/C2 use adaptive frozen anchors (tau=0.1). "
           "All: sibling grouping, 20k steps, seed 13, dim-48.}",
           "\\begin{tabular}{lcccccc}", "\\toprule",
           "Config & Enabled losses & PSNR & SSIM & LPIPS & Top-1 & EER\\\\",
           "\\midrule"]
    for i, r in enumerate(rows):
        cells = [r["method"], LOSS_TEXT[r["method"]]]
        for ck in keys:
            cells.append(_decorate(r[ck], dec[ck], higher[keys.index(ck)], rank_cols[ck][i]))
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "table_supplementary_ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("supplementary ablation rows", len(rows))


def write_efficiency_records() -> None:
    """效率记录：results/model_metrics.json + 本清单可配对的质量指标。"""
    src = ROOT / "results" / "model_metrics.json"
    if not src.exists():
        print("[warn] model_metrics.json missing")
        return
    mm = json.loads(src.read_text())
    if isinstance(mm, list):
        rows = mm
    elif isinstance(mm, dict):
        if all(isinstance(v, dict) for v in mm.values()):
            rows = [{"model": k, **v} for k, v in mm.items()]
        else:
            rows = mm.get("metrics", [])
    else:
        rows = []
    eff = pd.DataFrame(rows)
    if eff.empty:
        print("[warn] model_metrics empty")
        return
    eff.to_csv(TABLE_DIR / "efficiency_records.csv", index=False)
    print("efficiency records rows", len(eff))


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    # sanity: input-side metrics 应与方法无关（同一批退化输入）
    inp_means = df.groupby("method")["input_top1_correct"].mean()
    print("input top1 spread across methods:", float(inp_means.max() - inp_means.min()))

    write_table1(df, T1_MAIN, "main")
    write_table1(df, T1_ALL, "all")
    write_table2(df, T1_MAIN, "main")
    write_table2(df, T1_ALL, "all")
    write_table3(df, list(MAIN_METHODS), "core", core=True)
    write_supplementary(df)
    write_efficiency_records()


if __name__ == "__main__":
    main()
