"""Rich four-panel A5 vs B1 restoration--identity trade-off figure.

This extends the existing Figure 1 comparison without changing the input data
or the paired-comparison logic.  The statistical unit for source-level
bootstrap is source_id, with all seven degradation views kept together.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, OUT  # noqa: E402

FIG = OUT / "figures"
BOOT = OUT / "bootstrap"
FIG.mkdir(parents=True, exist_ok=True)
BOOT.mkdir(parents=True, exist_ok=True)

SEED = 20260907
N_BOOT = 20_000

DEG_LABEL = {
    "blur": "Blur", "haze": "Haze", "inpainting": "Inpainting",
    "lowlight": "Low-light", "noise": "Noise", "rain": "Rain", "snow": "Snow",
}
DEG_COLORS = {
    "blur": "#4C78A8", "haze": "#72B7B2", "inpainting": "#E6A24A",
    "lowlight": "#9C87C5", "noise": "#D9826B", "rain": "#6F8F72", "snow": "#7F9CCB",
}

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
QUAD_LABEL = {
    "Q1": "Joint improvement",
    "Q2": "Identity-only improvement",
    "Q3": "Both degraded",
    "Q4": "PSNR-only improvement",
}
QUAD_COLOR = {
    "Q1": "#E7F1E8", "Q2": "#E8EEF7", "Q3": "#F2E9E7", "Q4": "#F4EFE2",
}


def _set_style() -> None:
    # Times New Roman is requested; Nimbus Roman/Liberation Serif are the
    # installed metrically compatible fallbacks on this host.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "Liberation Serif"],
        "font.size": 8.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _quadrant(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Four requested quadrants; equality belongs to the non-improved side."""
    out = np.empty(len(dx), dtype=object)
    out[(dx > 0) & (dy > 0)] = "Q1"
    out[(dx <= 0) & (dy > 0)] = "Q2"
    out[(dx <= 0) & (dy <= 0)] = "Q3"
    out[(dx > 0) & (dy <= 0)] = "Q4"
    return out


def _paired_data(df: pd.DataFrame) -> pd.DataFrame:
    a5 = df[df["method"] == "A5"][['source_id', 'degradation', 'psnr', 'correct_source_rank']]
    b1 = df[df["method"] == "B1"][['source_id', 'degradation', 'psnr', 'correct_source_rank']]
    a5 = a5.rename(columns={"psnr": "psnr_a5", "correct_source_rank": "rank_a5"})
    b1 = b1.rename(columns={"psnr": "psnr_b1", "correct_source_rank": "rank_b1"})
    paired = a5.merge(b1, on=["source_id", "degradation"], validate="one_to_one")
    assert len(paired) == 1792, f"Expected 1792 paired views, got {len(paired)}"
    paired["delta_psnr"] = paired["psnr_a5"] - paired["psnr_b1"]
    paired["delta_rank"] = paired["rank_b1"] - paired["rank_a5"]
    paired["quadrant"] = _quadrant(paired["delta_psnr"].to_numpy(), paired["delta_rank"].to_numpy())
    return paired


def _source_data(paired: pd.DataFrame) -> pd.DataFrame:
    source = (paired.groupby("source_id", as_index=False)
              .agg(mean_delta_psnr=("delta_psnr", "mean"),
                   mean_delta_rank=("delta_rank", "mean")))
    joint = (paired.assign(joint=(paired["delta_psnr"] > 0) & (paired["delta_rank"] > 0))
             .groupby("source_id")["joint"].sum()
             .rename("joint_improvement_count"))
    source = source.merge(joint, on="source_id", validate="one_to_one")
    source["quadrant"] = _quadrant(source["mean_delta_psnr"].to_numpy(), source["mean_delta_rank"].to_numpy())
    assert len(source) == 256, f"Expected 256 source points, got {len(source)}"
    return source


def _bootstrap(source: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source bootstrap for panel-b statistics and degradation-wise ratios."""
    rng = np.random.default_rng(SEED)
    n_source = len(source)
    ids = source["source_id"].to_numpy()
    idx = rng.integers(0, n_source, size=(N_BOOT, n_source))

    x = source["mean_delta_psnr"].to_numpy()
    y = source["mean_delta_rank"].to_numpy()
    rho = np.array([spearmanr(x[s], y[s]).statistic for s in idx])
    rho = rho[np.isfinite(rho)]
    joint_source = ((x > 0) & (y > 0)).astype(float)
    stats = [
        ("spearman_rho", "Spearman rho", float(spearmanr(x, y).statistic), rho),
        ("mean_delta_psnr", "Mean delta PSNR", float(x.mean()), x[idx].mean(axis=1)),
        ("mean_delta_rank", "Mean delta Rank", float(y.mean()), y[idx].mean(axis=1)),
        ("joint_improvement_source_ratio", "Joint-improvement source ratio", float(joint_source.mean()), joint_source[idx].mean(axis=1)),
    ]
    rows = []
    for metric, label, estimate, boots in stats:
        rows.append({
            "metric": metric, "label": label, "estimate": estimate,
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "bootstrap_unit": "source_id", "n_sources": n_source,
            "n_boot": N_BOOT, "seed": SEED,
        })
    bootstrap_ci = pd.DataFrame(rows)

    deg_rows = []
    for d in DEGRADATIONS:
        sub = paired[paired["degradation"] == d]
        sub = sub.set_index("source_id").loc[ids]
        joint = ((sub["delta_psnr"].to_numpy() > 0) & (sub["delta_rank"].to_numpy() > 0)).astype(float)
        boots = joint[idx].mean(axis=1)
        deg_rows.append({
            "degradation": d, "label": DEG_LABEL[d],
            "joint_improvement_ratio": float(joint.mean()),
            "joint_improvement_percent": float(joint.mean() * 100),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "ci_low_percent": float(np.percentile(boots, 2.5) * 100),
            "ci_high_percent": float(np.percentile(boots, 97.5) * 100),
            "bootstrap_unit": "source_id", "n_sources": n_source,
            "n_boot": N_BOOT, "seed": SEED,
        })
    deg_ci = pd.DataFrame(deg_rows)
    return bootstrap_ci, deg_ci


def _summary(paired: pd.DataFrame, source: pd.DataFrame, bootstrap_ci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grain, sub in [("per_view", paired), ("per_source", source)]:
        q = sub["quadrant"].value_counts()
        n = len(sub)
        for code in QUADRANTS:
            rows.append({
                "grain": grain, "metric": "quadrant", "outcome": code,
                "label": QUAD_LABEL[code], "count": int(q.get(code, 0)),
                "percentage": float(q.get(code, 0) / n * 100), "n": n,
                "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "delta_psnr": np.nan, "delta_rank": np.nan, "outlier_score": np.nan,
            })
    for _, r in bootstrap_ci.iterrows():
        rows.append({
            "grain": "per_source", "metric": "bootstrap_statistic", "outcome": r["metric"],
            "label": r["label"], "count": np.nan, "percentage": np.nan, "n": int(r["n_sources"]),
            "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "delta_psnr": np.nan, "delta_rank": np.nan, "outlier_score": np.nan,
        })
    # Keep the selected outliers auditable without changing the raw input data.
    sx = paired["delta_psnr"].abs() / max(paired["delta_psnr"].abs().max(), 1e-12)
    sy = paired["delta_rank"].abs() / max(paired["delta_rank"].abs().max(), 1e-12)
    out = paired.assign(outlier_score=sx + sy).nlargest(5, "outlier_score")
    for rank, (_, r) in enumerate(out.iterrows(), start=1):
        rows.append({
            "grain": "per_view", "metric": "outlier", "outcome": f"rank_{rank}",
            "label": f"{r['source_id']} / {r['degradation']}", "count": np.nan,
            "percentage": np.nan, "n": len(paired), "estimate": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
            "delta_psnr": r["delta_psnr"], "delta_rank": r["delta_rank"],
            "outlier_score": r["outlier_score"],
        })
    return pd.DataFrame(rows)


def _decorate_scatter(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axvspan(xlim[0], 0, color=QUAD_COLOR["Q2"], alpha=0.52, zorder=0)
    ax.axvspan(0, xlim[1], color=QUAD_COLOR["Q1"], alpha=0.52, zorder=0)
    ax.axhspan(0, ylim[1], color="white", alpha=0.01, zorder=0)
    # Explicit quadrant rectangles make the four backgrounds independent.
    ax.fill_between([xlim[0], 0], ylim[0], 0, color=QUAD_COLOR["Q3"], alpha=0.52, zorder=0)
    ax.fill_between([0, xlim[1]], ylim[0], 0, color=QUAD_COLOR["Q4"], alpha=0.52, zorder=0)
    ax.axvline(0, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.grid(color="#B8B8B8", lw=0.45, alpha=0.35, zorder=0)


def _panel_a(fig, spec, paired: pd.DataFrame) -> None:
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec,
                                  width_ratios=[4.8, 1.15], height_ratios=[1.05, 4.8],
                                  wspace=0.08, hspace=0.08)
    ax_histx = fig.add_subplot(sub[0, 0])
    ax = fig.add_subplot(sub[1, 0])
    ax_histy = fig.add_subplot(sub[1, 1], sharey=ax)
    x = paired["delta_psnr"].to_numpy()
    y = paired["delta_rank"].to_numpy()
    pad_x = max((x.max() - x.min()) * 0.08, 0.02)
    pad_y = max((y.max() - y.min()) * 0.08, 0.25)
    xlim = (x.min() - pad_x, x.max() + pad_x)
    ylim = (y.min() - pad_y, y.max() + pad_y)
    _decorate_scatter(ax, xlim, ylim)
    for d in DEGRADATIONS:
        m = paired["degradation"].to_numpy() == d
        ax.scatter(x[m], y[m], s=8.5, color=DEG_COLORS[d], alpha=0.58,
                   edgecolors="none", label=DEG_LABEL[d], zorder=2)
    counts = paired["quadrant"].value_counts()
    n = len(paired)
    xpos = {"Q1": 0.74, "Q2": 0.24, "Q3": 0.24, "Q4": 0.74}
    ypos = {"Q1": 0.84, "Q2": 0.84, "Q3": 0.16, "Q4": 0.16}
    for code in QUADRANTS:
        c = int(counts.get(code, 0))
        ax.text(xpos[code], ypos[code], f"{QUAD_LABEL[code]}\n{c:,} ({c/n:.1%})",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
                color="#3F3F3F", linespacing=1.15, zorder=3)
    # The five labels are selected by combined normalized absolute changes.
    sx = np.abs(x) / max(np.max(np.abs(x)), 1e-12)
    sy = np.abs(y) / max(np.max(np.abs(y)), 1e-12)
    out_idx = np.argsort(sx + sy)[-5:][::-1]
    for i in out_idx:
        label = textwrap.fill(f"{paired.iloc[i]['source_id']} / {paired.iloc[i]['degradation']}", 25)
        ax.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords="offset points",
                    fontsize=5.7, color="#303030",
                    arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#666666"},
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#B0B0B0", "lw": 0.35, "alpha": 0.88},
                    zorder=4)
    ax.set_xlabel(r"$\Delta$PSNR = PSNR$_{A5}$ $-$ PSNR$_{B1}$ (dB)")
    ax.set_ylabel(r"$\Delta$Rank = rank$_{B1}$ $-$ rank$_{A5}$")
    ax_histx.text(0.0, 1.15, "(a) View-level effects", transform=ax_histx.transAxes,
                  ha="left", va="bottom", fontsize=10.0, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              frameon=False, handletextpad=0.3, columnspacing=0.8, markerscale=1.35)
    ax_histx.hist(x, bins=34, color="#A7B9C9", edgecolor="white", linewidth=0.25)
    ax_histx.axvline(0, color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax_histx.set_xlim(*xlim)
    ax_histx.set_ylabel("Count", fontsize=7)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.spines["right"].set_visible(False); ax_histx.spines["top"].set_visible(False)
    ax_histy.hist(y, bins=28, orientation="horizontal", color="#B9B2C7", edgecolor="white", linewidth=0.25)
    ax_histy.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax_histy.set_ylim(*ylim)
    ax_histy.set_xlabel("Count", fontsize=7)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.spines["right"].set_visible(False); ax_histy.spines["top"].set_visible(False)


def _panel_b(fig, spec, source: pd.DataFrame, bootstrap_ci: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    cmap = LinearSegmentedColormap.from_list("joint_count", ["#D9E2E8", "#AABFD0", "#7398B2", "#506F8C"])
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
    sc = ax.scatter(source["mean_delta_psnr"], source["mean_delta_rank"],
                    c=source["joint_improvement_count"], cmap=cmap, norm=norm,
                    s=19, alpha=0.9, edgecolors="white", linewidths=0.25, zorder=2)
    x = source["mean_delta_psnr"].to_numpy(); y = source["mean_delta_rank"].to_numpy()
    px = max((x.max() - x.min()) * 0.10, 0.02); py = max((y.max() - y.min()) * 0.10, 0.15)
    _decorate_scatter(ax, (x.min() - px, x.max() + px), (y.min() - py, y.max() + py))
    # Re-draw points over the pale quadrant fills.
    ax.scatter(source["mean_delta_psnr"], source["mean_delta_rank"],
               c=source["joint_improvement_count"], cmap=cmap, norm=norm,
               s=19, alpha=0.9, edgecolors="white", linewidths=0.25, zorder=2)
    ax.set_xlabel(r"Mean $\Delta$PSNR (dB)")
    ax.set_ylabel(r"Mean $\Delta$Rank")
    ax.set_title("(b) Source-level effects", loc="left", pad=5, fontweight="bold")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.035,
                        ticks=range(8))
    cbar.set_label("Number of jointly improved degradations", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.7)
    lookup = bootstrap_ci.set_index("metric")
    r = lookup.loc["spearman_rho"]
    p = lookup.loc["mean_delta_psnr"]
    k = lookup.loc["mean_delta_rank"]
    j = lookup.loc["joint_improvement_source_ratio"]
    text = (f"Spearman $\\rho$ = {r['estimate']:.3f}\n"
            f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]\n"
            f"Mean $\\Delta$PSNR = {p['estimate']:.3f} dB\n"
            f"95% CI [{p['ci_low']:.3f}, {p['ci_high']:.3f}]\n"
            f"Mean $\\Delta$Rank = {k['estimate']:.3f}\n"
            f"95% CI [{k['ci_low']:.3f}, {k['ci_high']:.3f}]\n"
            f"Joint-improvement source ratio = {j['estimate']:.1%}\n"
            f"95% CI [{j['ci_low']:.1%}, {j['ci_high']:.1%}]")
    ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            linespacing=1.22, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#A5A5A5", "lw": 0.5, "alpha": 0.93})


def _panel_c(fig, spec, paired: pd.DataFrame, source: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    totals = [len(paired), len(source)]
    data = [paired["quadrant"].value_counts(), source["quadrant"].value_counts()]
    left = np.zeros(2)
    # Keep the requested order from top to bottom: Per-view, Per-source.
    y = np.array([1, 0])
    for code in QUADRANTS:
        vals = np.array([d.get(code, 0) / n for d, n in zip(data, totals)])
        ax.barh(y, vals, left=left, color=QUAD_COLOR[code], edgecolor="white", linewidth=0.75,
                height=0.52, label=f"{code} {QUAD_LABEL[code]}")
        for yi, l, v in zip(y, left, vals):
            if v >= 0.055:
                ax.text(l + v / 2, yi, f"{v:.1%}", ha="center", va="center", fontsize=7.0, color="#3F3F3F")
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_yticks(y); ax.set_yticklabels(["Per-view\n(n=1,792)", "Per-source\n(n=256)"])
    ax.set_xlabel("Share of observations")
    ax.set_title("(c) Outcome composition", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#B8B8B8", lw=0.45, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=0.8, handletextpad=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _panel_d(fig, spec, deg_ci: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    plot = deg_ci.set_index("degradation").loc[list(reversed(DEGRADATIONS))].reset_index()
    y = np.arange(len(plot))
    est = plot["joint_improvement_percent"].to_numpy()
    lo = plot["ci_low_percent"].to_numpy()
    hi = plot["ci_high_percent"].to_numpy()
    ax.hlines(y, lo, hi, color="#536D7A", lw=1.35, zorder=2)
    ax.plot(est, y, "o", color="#344E5C", markersize=4.5, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Joint-improvement ratio (%)")
    ax.set_xlim(0, max(100, float(hi.max()) * 1.12))
    ax.set_title("(d) Degradation-wise joint improvement", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#B8B8B8", lw=0.45, alpha=0.38)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.02, "Point: estimate; line: 95% source-bootstrap CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color="#555555")


def main() -> None:
    _set_style()
    df = pd.read_csv(OUT / "unified_per_image.csv")
    paired = _paired_data(df)
    source = _source_data(paired)
    bootstrap_ci, deg_ci = _bootstrap(source, paired)
    summary = _summary(paired, source, bootstrap_ci)

    summary.to_csv(BOOT / "tradeoff_summary.csv", index=False)
    deg_ci.to_csv(BOOT / "degradation_joint_improvement.csv", index=False)
    bootstrap_ci.to_csv(BOOT / "tradeoff_bootstrap_ci.csv", index=False)

    fig = plt.figure(figsize=(15.2, 10.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, left=0.065, right=0.965, bottom=0.08, top=0.91,
                  wspace=0.28, hspace=0.38)
    _panel_a(fig, gs[0, 0], paired)
    _panel_b(fig, gs[0, 1], source, bootstrap_ci)
    _panel_c(fig, gs[1, 0], paired, source)
    _panel_d(fig, gs[1, 1], deg_ci)
    fig.suptitle("Restoration–Identity Trade-off Analysis", fontsize=16, fontweight="bold", y=0.965)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"restoration_identity_tradeoff_rich.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote:")
    for p in [FIG / "restoration_identity_tradeoff_rich.png",
              FIG / "restoration_identity_tradeoff_rich.pdf",
              FIG / "restoration_identity_tradeoff_rich.svg",
              BOOT / "tradeoff_summary.csv",
              BOOT / "degradation_joint_improvement.csv",
              BOOT / "tradeoff_bootstrap_ci.csv"]:
        print(p)
    print(f"paired views={len(paired)}, sources={len(source)}, degradations={paired['degradation'].nunique()}")


if __name__ == "__main__":
    main()
