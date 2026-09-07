"""11_write_docs.py —— 生成 PROTOCOL.md（评测协议/指纹/运行命令）与目录 README。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import METHODS, OUT, ROOT, VERIFIER_PATH, VERIFIER_SHA, csv_path, json_path, means_dir  # noqa: E402

DOC = OUT / "PROTOCOL.md"


def main() -> None:
    L = []
    L += ["# 评测协议、指纹与运行命令（paper_figures_v1）", ""]
    L += ["## 1. 数据与评测口径（全部来自官方 v2 管线，未重新评测）", "",
          "- 数据包：`data/plamd_sfr_v1`（contract）→ `data/PLAMD_SFR_v1_method_b`（图像本体）。",
          "- 划分：test = 256 sources（895 train / 128 val / 256 test，source 级互斥，seed 2026）。",
          "- 每 source：1 张 clean + 7 张退化图（blur/haze/inpainting/lowlight/noise/rain/snow）→ 1792 恢复视图。",
          "- 评测脚本：`scripts/evaluate_frozen_verifier_v2.py`；数值即该脚本产物 `.test.csv/.json`。",
          "- PSNR/SSIM：RGB float [0,1]，masked 口径（reflection-pad 不计）；SSIM 为全局 masked 实现（非 windowed）。",
          "- LPIPS：AlexNet `alex_256x256`，双图 bilinear 到 256×256，输入 x*2-1。",
          "- noise（Method B）：clean 用 PIL Lanczos resize 到 noise 原生尺寸后同源评估；恢复视图 bilinear 回 clean 尺寸。",
          "- 推理：分块恢复 `restore_tiled(512, 32)` 重叠均值融合。",
          "- 检索：恢复图 embedding（verifier tiled 512/32）与 256 个 test clean anchor 的余弦，候选含自身；",
          "  并列按更小索引优先（与官方 argmax 一致）；`correct_source_rank` 定义与此严格一致。",
          f"- 冻结评测 verifier：`{VERIFIER_PATH.name}`，sha256 `{VERIFIER_SHA}`。", ""]
    L += ["## 2. 指纹表（method → checkpoint sha256 / config sha / cohort）", "",
          "| method | cohort | checkpoint_sha256(前12) | config_sha(12) | 逐图CSV行数 |", "|---|---|---|---|---|"]
    import json as _json
    cache = _json.loads((OUT / "checkpoint_shas.json").read_text())
    for m in METHODS:
        res = _json.loads(json_path(m).read_text())
        ckpt = ROOT / m["ckpt"]
        csha = cache.get(str(ckpt.resolve()), "")[:12]
        L.append(f"| {m['id']} | {m['cohort']} | `{csha}` | `{res['config_fingerprint']['sha256']}` | 1792 |")
    L += ["", "注：全 26 方法同一 verifier、同一 data_pkg 指纹"
              "（audit_index `a4b095234bc8…` / groups `990880d4c2a6…`）；config 各不相同。", ""]
    L += ["## 3. 生成资产与命令", "",
          "产出：", "- `unified_per_image.csv`（26 方法 × 1792 行，schema 见 `unified_schema.md`）",
          "- `tables/table{1,2,3}_*.csv/.tex`、`table_supplementary_ablation.*`、`efficiency_records.csv`",
          "- `bootstrap/bootstrap_a5_vs_b1.*`", "- `figures/fig{1,4,6}_*`（纯指标）与 `fig{2,3,5}_*`（样本恢复图）",
          "- `cases/case_plan.json|csv`、`fig1_*_delta_counts.csv`、`fig4_stability_stats.csv`",
          "- `consistency_report.md`、`ANALYSIS_REPORT.md`", "",
          "离线再生成（无 GPU）：", "```bash", "bash analysis/paper_figures_v1/run_all.sh", "```",
          "含样本恢复图推理（GPU，需要显卡，默认 cuda:3）：", "```bash",
          "RUN_INFERENCE=1 DEVICE=cuda:3 bash analysis/paper_figures_v1/run_all.sh", "```", "",
          "对应官方命令（如需从 checkpoint 重新生成任意方法的 v2 结果）：", "```bash",
          "cd /home/zhanghangning/siblingrestore-pilot-server",
          ".venv/bin/python scripts/evaluate_frozen_verifier_v2.py \\",
          "    --checkpoint runs/ablation_core/A5_no_sibling_seed13/best.pt \\",
          "    --verifier runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt \\",
          "    --data-root data/plamd_sfr_v1 --split test --device cuda \\",
          "    --method A5_no_sibling --seed 13 --lpips \\",
          "    --output results/ablation_core/A5_no_sibling.13.test.json", "```",
          "（其余方法改 `--checkpoint`/`--method` 与输出前缀即可；输出与现网一致时才可覆盖，否则写新前缀。）",
          "", "## 4. 本轮规则与说明",
          "- 未启动任何训练；未修改/覆盖任何历史结果；未改动数据划分与生成。",
          "- 恢复图未在本项目落盘（restored_path 列为空=已知缺失）；图 2/3/5 的恢复图为补推理产物。",
          "- 选图规则在拼图前锁定（`10_select_cases.py` 头注释），案例坐标与依据见 `case_plan.json`。",
          "- 仅用 test 结果分析；未用测试集选择权重/检查点。"]
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", DOC)

    readme = ROOT / "analysis" / "paper_figures_v1" / "README.md"
    R = ["# paper_figures_v1 —— 现有结果整理 + 逐图评测 + 论文图表", "",
         "在 SiblingRestore 项目内只读复用官方 v2 评测产物，补算官方未保存的逐图检索指标，",
         "自动生成表 1/2/3（CSV+LaTeX）、图 1–6（PDF/SVG/300dpi PNG）与分析报告。",
         "所有脚本顺序见 `run_all.sh`；协议/指纹/命令见 `outputs/paper_figures_v1/PROTOCOL.md`。", "",
         "| 脚本 | 内容 |", "|---|---|",
         "| `01_assets_manifest.py` | 资产清单/可用性/指纹 |",
         "| `02_build_unified_table.py` | 统一逐图结果表（rank/top5/同类别重算） |",
         "| `03_tables.py` | 表 1/2/3 + 补充表 + 效率记录（CSV/LaTeX） |",
         "| `04_bootstrap.py` | A5 vs B1 源级块状 bootstrap 95% CI |",
         "| `05_figures_metric.py` | 图 1（散点）/图 4（稳定性）/图 6（效率） |",
         "| `10_select_cases.py` | 固定规则的案例选择（拼图前锁定） |",
         "| `06_infer_samples.py` | 用既有 checkpoint 补推理少量恢复图（GPU） |",
         "| `07_figures_visual.py` | 图 2（同源多退化）/图 3（检索案例）/图 5（局部细节） |",
         "| `08_consistency_check.py` | 一致性核对 |",
         "| `09_summary_report.py` | 结果分析报告 |",
         "| `11_write_docs.py` | PROTOCOL/README |", "",
         "离线重生成：`bash run_all.sh`；需要样本恢复图时：`RUN_INFERENCE=1 DEVICE=cuda:3 bash run_all.sh`。"]
    readme.write_text("\n".join(R) + "\n", encoding="utf-8")
    print("wrote", readme)


if __name__ == "__main__":
    main()
