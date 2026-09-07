# paper_figures_v1 —— 现有结果整理 + 逐图评测 + 论文图表

在 SiblingRestore 项目内只读复用官方 v2 评测产物，补算官方未保存的逐图检索指标，
自动生成表 1/2/3（CSV+LaTeX）、图 1–6（PDF/SVG/300dpi PNG）与分析报告。
所有脚本顺序见 `run_all.sh`；协议/指纹/命令见 `outputs/paper_figures_v1/PROTOCOL.md`。

| 脚本 | 内容 |
|---|---|
| `01_assets_manifest.py` | 资产清单/可用性/指纹 |
| `02_build_unified_table.py` | 统一逐图结果表（rank/top5/同类别重算） |
| `03_tables.py` | 表 1/2/3 + 补充表 + 效率记录（CSV/LaTeX） |
| `04_bootstrap.py` | A5 vs B1 源级块状 bootstrap 95% CI |
| `05_figures_metric.py` | 图 1（散点）/图 4（稳定性）/图 6（效率） |
| `10_select_cases.py` | 固定规则的案例选择（拼图前锁定） |
| `06_infer_samples.py` | 用既有 checkpoint 补推理少量恢复图（GPU） |
| `07_figures_visual.py` | 图 2（同源多退化）/图 3（检索案例）/图 5（局部细节） |
| `08_consistency_check.py` | 一致性核对 |
| `09_summary_report.py` | 结果分析报告 |
| `11_write_docs.py` | PROTOCOL/README |

离线重生成：`bash run_all.sh`；需要样本恢复图时：`RUN_INFERENCE=1 DEVICE=cuda:3 bash run_all.sh`。
