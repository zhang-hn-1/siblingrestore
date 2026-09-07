# Table 1 notes（模型映射 / 协议 / 缺失 / 排名 / 复现）

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
