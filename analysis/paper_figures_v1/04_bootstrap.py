"""04_bootstrap.py —— A5 与 B1 的源级配对 bootstrap 95% CI。

统计单元 = source_id（同一 source 的全部 7 张退化图作为一个块），
对每 source 计算块内均值差 delta(A5 - B1)，再对 256 个 source 做有放回重采样，
固定随机种子，重采样 20000 次，取 2.5%/97.5% 分位数作为 95% CI。

说明：A5/B1 均只有单一训练 seed(13)；CI 反映的是"测试来源抽样"的不确定性，
不代表训练随机性。CI 用于配对差值的区间估计，不单独作为"显著因果"证据。
指标：psnr/ssim/lpips/restored_top1/samecat_top1/correct_source_rank/restored_margin。
输出: outputs/paper_figures_v1/bootstrap/bootstrap_a5_vs_b1.csv / .md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402

BOOT_DIR = OUT / "bootstrap"
BOOT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("psnr", "PSNR (dB)", True),
    ("ssim", "SSIM", True),
    ("lpips", "LPIPS", False),
    ("top1_correct", "Restored Top-1", True),
    ("samecat_top1_correct", "Same-cat Top-1", True),
    ("correct_source_rank", "Correct-source rank", False),
    ("restored_margin", "Restored margin", True),
]

SEED = 20260907
N_BOOT = 20_000


def main() -> None:
    df = pd.read_csv(OUT / "unified_per_image.csv")
    a5 = df[df.method == "A5"].set_index(["source_id", "degradation"])
    b1 = df[df.method == "B1"].set_index(["source_id", "degradation"])
    assert set(a5.index) == set(b1.index)
    source_ids = np.array(sorted({sid for sid, _ in a5.index}))

    # 每 source 块内均值差
    block = {}
    for sid in source_ids:
        dA = a5.loc[sid]
        dB = b1.loc[sid]
        block[sid] = {m: float(dA[m].mean() - dB[m].mean()) for m, _, _ in METRICS}
    block_df = pd.DataFrame.from_dict(block, orient="index")
    block_df.index.name = "source_id"
    block_df.to_csv(BOOT_DIR / "per_source_delta_a5_minus_b1.csv")

    rng = np.random.default_rng(SEED)
    n_src = len(source_ids)
    # 预生成索引：20000 x 256
    idx = rng.integers(0, n_src, size=(N_BOOT, n_src))
    rows = []
    for m, label, _ in METRICS:
        vals = block_df[m].to_numpy()
        sample_means = vals[idx].mean(axis=1)
        ci = (np.percentile(sample_means, 2.5), np.percentile(sample_means, 97.5))
        rows.append({
            "metric": m, "label": label, "delta_mean": float(vals.mean()),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
            "pct_sources_delta_gt0": float((vals > 0).mean()),
        })
    res = pd.DataFrame(rows)
    res.to_csv(BOOT_DIR / "bootstrap_a5_vs_b1.csv", index=False)

    lines = [
        "# A5 vs B1：源级配对 bootstrap（95% CI）",
        "",
        f"- 配对统计单元：source_id（每 source 7 张退化图作为一个块，块内先求均值差）。",
        f"- 重采样：{N_BOOT} 次，固定随机种子 `{SEED}`，测试来源 256 个。",
        "- 单一训练 seed(13)：CI 反映测试来源抽样的不确定性，不代表训练随机性。",
        "- 方向（A5 − B1 为正表示 A5 更优的指标）：PSNR/SSIM/Top1/SameCat/margin 越大越好，LPIPS/rank 越小越好。",
        "",
        "| metric | delta mean | 95% CI | 正差 source 占比 |",
        "|---|---:|---:|---:|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['label']} | {r['delta_mean']:.4f} | "
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['pct_sources_delta_gt0']:.3f} |")
    (BOOT_DIR / "bootstrap_a5_vs_b1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
