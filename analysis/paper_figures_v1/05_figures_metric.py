"""05_figures_metric.py —— 图1 逐图收益散点 / 图4 同源跨退化稳定性 / 图6 质量-效率。

输出 PDF / SVG / PNG(300dpi) 到 outputs/paper_figures_v1/figures/，
分类计数 CSV 到 outputs/paper_figures_v1/cases/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, OUT  # noqa: E402

FIG = OUT / "figures"
CASES = OUT / "cases"
FIG.mkdir(parents=True, exist_ok=True)
CASES.mkdir(parents=True, exist_ok=True)

DEG_COLORS = {
    "blur": "#1f77b4", "haze": "#ff7f0e", "inpainting": "#2ca02c",
    "lowlight": "#d62728", "noise": "#9467bd", "rain": "#8c564b", "snow": "#17becf",
}
DEG_LABELS = {"lowlight": "Low-light", "inpainting": "Inpainting"}
DEG_TITLE = {d: (DEG_LABELS.get(d) or d.capitalize()) for d in DEGRADATIONS}


def save(fig, name: str):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def cat_of(dp, dr):
    """dp = ΔPSNR (A5-B1), dr = Δrank (B1_rank-A5_rank; >0 rank improves)。"""
    if dp > 0 and dr > 0:
        return "both_improved"
    if dp > 0 and dr == 0:
        return "only_psnr_improved"
    if dr > 0 and dp == 0:
        return "only_rank_improved"
    if dp < 0 and dr < 0:
        return "both_worse"
    if dr == 0:
        return "rank_unchanged"
    return "mixed_other"


CATS = ["both_improved", "only_psnr_improved", "only_rank_improved",
        "rank_unchanged", "both_worse", "mixed_other"]
CAT_LABEL = {
    "both_improved": "both improved", "only_psnr_improved": "only PSNR improved",
    "only_rank_improved": "only rank improved", "rank_unchanged": "rank unchanged",
    "both_worse": "both worse", "mixed_other": "opposite-sign / other"}


def fig1(df: pd.DataFrame) -> None:
    a = df[df.method == "A5"].set_index(["source_id", "degradation"])
    b = df[df.method == "B1"].set_index(["source_id", "degradation"])
    common = a.index.intersection(b.index)
    dA, dB = a.loc[common], b.loc[common]
    x = dA["psnr"].to_numpy() - dB["psnr"].to_numpy()
    y = dB["correct_source_rank"].to_numpy() - dA["correct_source_rank"].to_numpy()
    deg = dA.index.get_level_values("degradation").to_numpy()
    src = dA.index.get_level_values("source_id").to_numpy()

    # 每退化计数（view 级）
    recs = []
    for d in DEGRADATIONS:
        m = deg == d
        cnt = pd.Series([cat_of(a_, b_) for a_, b_ in zip(x[m], y[m])]).value_counts()
        row = {"degradation": d, "n_views": int(m.sum())}
        row.update({c: int(cnt.get(c, 0)) for c in CATS})
        row["mean_dpsnr"] = float(x[m].mean()); row["mean_drank"] = float(y[m].mean())
        recs.append(row)
    allrow = {"degradation": "all", "n_views": len(x)}
    cnt_all = pd.Series([cat_of(a_, b_) for a_, b_ in zip(x, y)]).value_counts()
    allrow.update({c: int(cnt_all.get(c, 0)) for c in CATS})
    allrow["mean_dpsnr"] = float(x.mean()); allrow["mean_drank"] = float(y.mean())
    recs.append(allrow)
    counts = pd.DataFrame(recs).set_index("degradation")
    counts.to_csv(CASES / "fig1_view_delta_counts.csv")

    # 每 source 聚合（7 视图块均值；整体一张表，避免同源多退化图被视为独立证据）
    per_source = pd.DataFrame({"source": src, "dpsnr": x, "drank": y})
    ps = per_source.groupby("source")[["dpsnr", "drank"]].mean()
    cnt_all_s = pd.Series([cat_of(v.dpsnr, v.drank) for v in ps.itertuples()]).value_counts()
    row_all = {"degradation": "all", "n_sources": len(ps)}
    row_all.update({c: int(cnt_all_s.get(c, 0)) for c in CATS})
    row_all["mean_dpsnr"] = float(ps.dpsnr.mean()); row_all["mean_drank"] = float(ps.drank.mean())
    pd.DataFrame([row_all]).set_index("degradation").to_csv(CASES / "fig1_source_delta_counts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
    for ax, use_src, title in ((axes[0], False, "Per-view (1792 test views)"),
                               (axes[1], True, "Per-source block mean (256 sources)")):
        if use_src:
            xs = ps["dpsnr"].to_numpy(); ys = ps["drank"].to_numpy()
            dcol = None
            for d in DEGRADATIONS:
                # 无法区分 source 主要退化；直接单色
                pass
            ax.scatter(xs, ys, s=14, alpha=0.55, color="#444444", edgecolors="none", zorder=2)
        else:
            for d in DEGRADATIONS:
                m = deg == d
                ax.scatter(x[m], y[m], s=9, alpha=0.5, color=DEG_COLORS[d],
                           label=DEG_TITLE[d], edgecolors="none", zorder=2)
        ax.axhline(0, color="k", lw=0.9, ls="--", zorder=1)
        ax.axvline(0, color="k", lw=0.9, ls="--", zorder=1)
        ax.set_xlabel("$\\Delta$ PSNR = PSNR$_{A5}$ $-$ PSNR$_{B1}$ (dB)")
        ax.set_ylabel("$\\Delta$ correct-source rank =\nrank$_{B1}$ $-$ rank$_{A5}$  (positive = A5 improves)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(title="Degradation", fontsize=7, markerscale=1.6, loc="upper right",
                   framealpha=0.9)
    fig.suptitle("A5 vs B1 per-image gain: restoration quality vs. source-retrieval rank",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig1_scatter_a5_vs_b1")
    print("fig1 done")


def fig4(df: pd.DataFrame) -> None:
    """同源跨退化稳定性：每 source 在 7 类退化中 Top1 成功次数 0..7。"""
    methods = ["A5", "B1", "dehazeformer", "restormer"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.2
    stats = {}
    for i, mid in enumerate(methods):
        sub = df[df.method == mid]
        succ = sub.groupby("source_id")["top1_correct"].sum()
        # 仅统计 7 退化齐全的来源（测试集全部 256 均齐全）
        full = sub.groupby("source_id").size()
        assert (full == 7).all()
        hist = np.bincount(succ.to_numpy().astype(int), minlength=8)
        stats[mid] = {
            "n_full_sources": int(len(succ)),
            "hist": hist.tolist(),
            "frac_all7": float((succ == 7).mean()),
            "frac_all0": float((succ == 0).mean()),
            "mean_success": float(succ.mean()),
        }
        ax.bar(np.arange(8) + (i - 1.5) * width, hist, width=width,
               label=("Ours (A5)" if mid == "A5" else mid))
    ax.set_xticks(range(8))
    ax.set_xlabel("Number of degradations with correct Top-1 retrieval (per source, out of 7)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Per-source cross-degradation identity stability (256 test sources)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig4_source_stability_top1")
    pd.DataFrame.from_dict(stats, orient="index").to_csv(CASES / "fig4_stability_stats.csv")
    print("fig4 done:", {k: round(v["mean_success"], 3) for k, v in stats.items()})


def fig6() -> None:
    """质量-效率散点：仅使用本项目统一口径记录的效率（256x256, batch1, fp32, V100）。"""
    eff = pd.read_csv(OUT / "tables/efficiency_records.csv").rename(columns={"model": "method"})
    uni = pd.read_csv(OUT / "unified_per_image.csv")
    deg_m = uni.groupby(["method", "degradation"])[["psnr", "lpips"]].mean()
    macro = deg_m.groupby("method").mean().reset_index().rename(
        columns={"psnr": "macro_psnr", "lpips": "macro_lpips"})
    top1 = uni.groupby("method")["top1_correct"].mean().rename("macro_top1").reset_index()
    q = macro.merge(top1, on="method")
    d = eff.merge(q, on="method", how="inner")
    # A5 与 ours_dim48_anchor 为同一 dim-48 架构（本项目记录以其测量）；A5 单独给出
    a5q = q[q.method == "A5"].iloc[0]
    a5e = eff[eff.method == "ours_dim48_anchor"].iloc[0]
    a5row = pd.DataFrame([{**a5e.to_dict(), "method": "A5",
                           "macro_psnr": a5q["macro_psnr"], "macro_top1": a5q["macro_top1"],
                           "macro_lpips": a5q["macro_lpips"]}])
    d = pd.concat([d, a5row], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = {"A5": "Ours (A5)", "ours_dim48_anchor": "Ours dim48", "ours_dim64_anchor": "Ours dim64"}
    colors = {"A5": "#c00000", "ours_dim48_anchor": "#c00000", "ours_dim64_anchor": "#e07b00"}
    for ax, (yk, ylab) in zip(axes, [("macro_psnr", "Macro PSNR (dB) $\\uparrow$"),
                                     ("macro_top1", "Restored Top-1 $\\uparrow$")]):
        area = np.pi * (d["params_M"] / 2.0)  # 面积 ∝ 参数量
        ax.scatter(d["latency_ms"], d[yk], s=area, alpha=0.75, c=[colors.get(m, "#1f77b4") for m in d["method"]])
        for _, r in d.iterrows():
            off = 4 if r["method"] == "A5" else 0
            ax.annotate(labels.get(r["method"], r["method"]),
                        (r["latency_ms"], r[yk]), fontsize=7,
                        xytext=(3, off + 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms, log scale, 256$\\times$256 batch=1)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        # size legend (面积∝参数量)
        for p, lab in [(2.63, "2.6M"), (26.13, "26M"), (40, "40M")]:
            ax.scatter([], [], s=np.pi * p / 2.0, c="grey", alpha=0.6, label=lab)
        ax.legend(title="Params", fontsize=7, loc="lower right")
    fig.suptitle("Efficiency vs quality / source retention\n"
                 "(latency from project-wide consistent protocol: 256$\\times$256, batch 1, fp32, V100)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_quality_efficiency")
    d.to_csv(CASES / "fig6_efficiency_scatter_data.csv", index=False)
    print("fig6 done:", len(d), "points")


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    fig1(df)
    fig4(df)
    fig6()


if __name__ == "__main__":
    main()
