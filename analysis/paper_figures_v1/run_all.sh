#!/usr/bin/env bash
# 一条命令重新生成表格与分析图（不含需要 GPU 的样本推理；需要时设置 RUN_INFERENCE=1）。
# 用法:
#   bash analysis/paper_figures_v1/run_all.sh          # 纯离线：表 + 图1/4/6 + 报告
#   RUN_INFERENCE=1 bash analysis/paper_figures_v1/run_all.sh  # 额外补推理图2/3/5（需 GPU）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"

echo "[1/8] 资产清单"                && "$PY" analysis/paper_figures_v1/01_assets_manifest.py
echo "[2/8] 统一逐图结果表"          && "$PY" analysis/paper_figures_v1/02_build_unified_table.py
echo "[3/8] 表格（CSV + LaTeX）"     && "$PY" analysis/paper_figures_v1/03_tables.py
echo "[4/8] A5 vs B1 源级 bootstrap" && "$PY" analysis/paper_figures_v1/04_bootstrap.py
echo "[5/8] 指标图（图1/4/6）"       && "$PY" analysis/paper_figures_v1/05_figures_metric.py
if [[ "${RUN_INFERENCE:-0}" == "1" ]]; then
  echo "[6a/8] 案例选择(固定规则)"   && "$PY" analysis/paper_figures_v1/10_select_cases.py
  echo "[6b/8] 样本恢复图推理(GPU)"  && "$PY" analysis/paper_figures_v1/06_infer_samples.py --device "${DEVICE:-cuda:3}"
  echo "[7/8] 视觉图（图2/3/5）"     && "$PY" analysis/paper_figures_v1/07_figures_visual.py
else
  echo "[6/8] 跳过样本推理与图2/3/5（需要时 RUN_INFERENCE=1）"
fi
echo "[7/8] 一致性核对"              && "$PY" analysis/paper_figures_v1/08_consistency_check.py
echo "[8/8] 分析报告"                && "$PY" analysis/paper_figures_v1/09_summary_report.py
echo "完成。产物目录：outputs/paper_figures_v1/"
