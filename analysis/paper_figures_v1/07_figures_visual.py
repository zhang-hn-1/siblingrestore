"""07_figures_visual.py —— 图2 同源多退化对比 / 图3 来源检索案例 / 图5 局部细节。

输入：case_plan.json + outputs/paper_figures_v1/sample_restored 下补推理恢复图
      + 统一逐图表(unified_per_image.csv) 中的 PSNR/rank/cos 标注。
所有图像统一对齐到 clean 原生尺寸（noise 已 bilinear 回 clean 尺寸）。
同源内各方法同位置像素空间对齐；局部框各方法坐标一致。
误差图使用统一数值范围与色标，不逐张独立归一化。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEGRADATIONS, OUT, ROOT  # noqa: E402

FIG = OUT / "figures"
DEST = OUT / "sample_restored"
DEG_TITLE = {"lowlight": "Low-light", "inpainting": "Inpainting"}

GROUPS = json.loads((ROOT / "data/plamd_sfr_v1/metadata/source_groups.json").read_text())
GMAP = {g["source_id"]: g for g in GROUPS}
CAT_SHORT = {"good": "good", "rust": "rust", "bird-nest": "bird-nest"}


def safe(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def load_png(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def load_clean(sid: str) -> np.ndarray:
    return load_png(DEST / "_ref" / f"{safe(sid)}__clean.png")


def cell_path(mid: str, sid: str, deg: str) -> Path:
    return DEST / mid / f"{safe(sid)}__{deg}.png"


def imshow(ax, arr, **kw):
    ax.imshow(arr, interpolation="lanczos", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def annotate(ax, arr, text, color="white", loc="tl", fontsize=7):
    x = 4 if loc == "tl" else arr.shape[1] - 4
    ax.text(x, 4, text, color=color, fontsize=fontsize,
            ha="left" if loc == "tl" else "right", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))


def save(fig, name: str):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _per_view_scores(mid: str, sid: str, deg: str) -> tuple[np.ndarray, np.ndarray]:
    """从官方 .sources.jsonl 取该 (source,deg) 的 input/restored score 向量。"""
    import json as _json
    from common import sources_path
    mp = {m["id"]: m for m in __import__("common").METHODS}
    with open(sources_path(mp[mid]), encoding="utf-8") as fh:
        for ln in fh:
            rec = _json.loads(ln)
            if rec["source_id"] != sid:
                continue
            for v in rec["views"]:
                if v["degradation"] == deg:
                    return (np.asarray(v["input_scores"]), np.asarray(v["restored_scores"]))
    raise KeyError((mid, sid, deg))


def fig2(df: pd.DataFrame) -> None:
    plan = json.loads((OUT / "cases/case_plan.json").read_text())
    metric = df.set_index(["method", "source_id", "degradation"])
    srcs = plan["fig2_sources"]
    fig, axes = plt.subplots(4 * len(srcs), 7, figsize=(30, 6.4 * len(srcs)))
    axes = np.atleast_2d(axes)
    for bi, src in enumerate(srcs):
        sid = src["source_id"]
        cat = src["category"]
        clean = load_clean(sid)
        base = 4 * bi
        row_lab = {0: "Degraded input", 1: "B1 (deg only)",
                   2: "Ours A5 (deg+anchor)", 3: "Clean reference"}
        for ri in range(4):
            for c, deg in enumerate(DEGRADATIONS):
                ax = axes[base + ri, c]
                if ri == 0:
                    arr = load_png(DEST / "_ref" / f"{safe(sid)}__{deg}__input.png")
                elif ri == 3:
                    arr = clean
                else:
                    mid = "B1" if ri == 1 else "A5"
                    arr = load_png(cell_path(mid, sid, deg))
                    mrow = metric.loc[(mid, sid, deg)]
                    annotate(ax, arr, f"PSNR {mrow.psnr:.2f}", loc="tl")
                    if mrow.top1_correct:
                        annotate(ax, arr, "Top-1", color="#8dff8d", loc="tr")
                    else:
                        annotate(ax, arr, f"r={int(mrow.correct_source_rank)}",
                                 color="#ffb0b0", loc="tr")
                imshow(ax, arr)
                if ri == 0:
                    ax.set_title(DEG_TITLE.get(deg, deg.capitalize()), fontsize=11)
            axes[base + ri, 0].set_ylabel(f"{cat}\n{row_lab[ri]}", fontsize=9)
    fig.tight_layout(h_pad=0.45, w_pad=0.45)
    save(fig, "fig2_same_source_multi_deg")
    print("fig2 done")


def fig3(df: pd.DataFrame) -> None:
    plan = json.loads((OUT / "cases/case_plan.json").read_text())
    metric = df.set_index(["method", "source_id", "degradation"])
    cases = plan["fig3_cases"]
    role_t = {"improvement": "Improvement (B1 wrong, A5 correct)",
              "regression": "Regression (B1 correct, A5 wrong)",
              "unchanged": "Unchanged (both correct Top-1)"}
    # 每个 case 一个 2 行 x 3 列块：行0 = input/B1/A5 恢复；行1 = own clean/B1 match/A5 match
    fig, axes = plt.subplots(2, 3 * len(cases), figsize=(11.5 * len(cases), 9))
    axes = np.atleast_2d(axes)
    for ci, case in enumerate(cases):
        sid, deg = case["source_id"], case["degradation"]
        col0 = 3 * ci
        clean = load_clean(sid)
        inp = load_png(DEST / "_ref" / f"{safe(sid)}__{deg}__input.png")
        # row 0
        imshow(axes[0, col0], inp)
        axes[0, col0].set_title("Degraded input", fontsize=9)
        for k, mid in enumerate(["B1", "A5"]):
            arr = load_png(cell_path(mid, sid, deg))
            imshow(axes[0, col0 + 1 + k], arr)
            mrow = metric.loc[(mid, sid, deg)]
            axes[0, col0 + 1 + k].set_title(
                f"{mid} restored\nPSNR {mrow.psnr:.2f} | rank {int(mrow.correct_source_rank)}\n"
                f"cos(own) {mrow.own_clean_similarity:.3f}", fontsize=8)
        # row 1
        imshow(axes[1, col0], clean)
        axes[1, col0].set_title("Source clean", fontsize=9)
        for k, mid in enumerate(["B1", "A5"]):
            mrow = metric.loc[(mid, sid, deg)]
            ret = mrow.retrieved_source_id
            _, rscores = _per_view_scores(mid, sid, deg)
            bank_ids = [g["source_id"] for g in GROUPS if g["split"] == "test"]
            ridx = bank_ids.index(ret)
            cos_ret = float(rscores[ridx])
            img = clean if ret == sid else load_png(
                ROOT / "data/plamd_sfr_v1" / GMAP[ret]["clean_path"])
            imshow(axes[1, col0 + 1 + k], img)
            mark = "own source" if ret == sid else GMAP[ret]["subcategory"]
            axes[1, col0 + 1 + k].set_title(
                f"{mid} top match\n{ret.split('/')[-1][:20]}\n({mark}) cos {cos_ret:.3f}",
                fontsize=7)
        # 列组标题
        m_b = metric.loc[("B1", sid, deg)]
        m_a = metric.loc[("A5", sid, deg)]
        fig.text((col0 + 1.5) / (3 * len(cases)), 0.985,
                 f"{role_t[case['role']]}  |  {case['category']} / {deg}\n"
                 f"B1: r={int(m_b.correct_source_rank)}  A5: r={int(m_a.correct_source_rank)}",
                 ha="center", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=0.6, w_pad=0.9, rect=[0, 0, 1, 0.96])
    save(fig, "fig3_source_retrieval_cases")
    print("fig3 done")


def fig5(df: pd.DataFrame) -> None:
    plan = json.loads((OUT / "cases/case_plan.json").read_text())
    f5 = plan["fig5"]
    sid, deg = f5["source_id"], f5["degradation"]
    y0, y1, x0, x1 = f5["crop"]
    clean_full = load_clean(sid)
    rows = [("Degraded input", None), ("DehazeFormer", "dehazeformer"),
            ("B1 (deg only)", "B1"), ("Ours A5 (deg+anchor)", "A5"), ("Clean", None)]
    crop = {}
    for lab, mid in rows:
        if mid is None:
            arr = load_png(DEST / "_ref" / f"{safe(sid)}__{deg}__input.png")
        elif lab == "Clean":
            arr = clean_full
        else:
            arr = load_png(cell_path(mid, sid, deg))
        crop[lab] = arr[y0:y1, x0:x1]
    err_keys = ["Degraded input", "DehazeFormer", "B1 (deg only)", "Ours A5 (deg+anchor)"]
    err_imgs = [np.abs(crop[k] - crop["Clean"]).mean(axis=2) for k in err_keys]
    vmax = max(float(np.percentile(e, 99.0)) for e in err_imgs) or 1e-6
    fig, axes = plt.subplots(5, 3, figsize=(12, 15),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    axf = axes[0, 2]
    imshow(axf, clean_full)
    axf.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="red", lw=1.2))
    axf.set_title("Full clean image (red: crop)", fontsize=8)
    for i in range(1, 5):
        axes[i, 2].axis("off")
    axes[4, 2].text(0, 0.5,
                    f"crop: rows [{y0}:{y1}], cols [{x0}:{x1}]\n"
                    "window centred on the largest\ndegradation-induced difference\n"
                    "(visual analysis only; no defect label\nis asserted from restored outputs).",
                    va="center", fontsize=9)
    metric = df.set_index(["method", "source_id", "degradation"])
    err_labels = ["|input$-$clean|", "|DehazeFormer$-$clean|", "|B1$-$clean|", "|A5$-$clean|"]
    for r, (lab, mid) in enumerate(rows):
        imshow(axes[r, 0], crop[lab])
        axes[r, 0].set_ylabel(lab, fontsize=9)
        if mid in ("dehazeformer", "B1", "A5"):
            mrow = metric.loc[(mid, sid, deg)]
            axes[r, 0].set_title(f"PSNR {mrow.psnr:.2f} dB", fontsize=8)
        if r < 4:
            axes[r, 1].imshow(err_imgs[r], cmap="magma", vmin=0, vmax=vmax)
            axes[r, 1].set_title(err_labels[r], fontsize=8)
            axes[r, 1].set_xticks([])
            axes[r, 1].set_yticks([])
    for ax in axes[:, 2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="magma"),
                 ax=axes[:, 1], fraction=0.04, pad=0.02).set_label(
        "mean abs. difference (shared scale)")
    fig.suptitle(f"Local detail comparison — {GMAP[sid]['subcategory']} source ({deg}), "
                 f"source {sid.split('/')[-1][:28]}", fontsize=12)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "fig5_local_detail")
    print("fig5 done")


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    fig2(df)
    fig3(df)
    fig5(df)


if __name__ == "__main__":
    main()
