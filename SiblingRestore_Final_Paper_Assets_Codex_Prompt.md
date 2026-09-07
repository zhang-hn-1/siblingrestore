# SiblingRestore论文最终图表整合与复用：Codex总提示词

## 任务目标

仓库：`https://github.com/zhang-hn-1/siblingrestore`

目标：把当前仓库中已经完成的实验图、统计图、可视化和表格统一串联成一套论文主文素材。论文主线固定为：

1. Restoration Fidelity
2. Identity Fidelity
3. Efficiency

核心原则：**先审计、优先复用、只补缺口、禁止重复训练、禁止改变真实数据。**

---

## 0. 硬性规则

### 必须先做

执行：

```bash
git status
git log --oneline --decorate -20
git rev-parse HEAD
```

然后扫描：

```text
artifacts/
artifacts/identity_evaluation/
artifacts/identity_evaluation/paper_figures/
artifacts/identity_evaluation/visualization/
paper_visualization/
experiment_data/
results/
results/ablation_core/
results/campaigns/
EXPERIMENT_DATA_SUMMARY.md
ABLATION_REPORT_FINAL.md
```

查找：

```text
*.png *.pdf *.svg *.csv *.json *.md *.tex
identity cosine recovery embedding PCA tSNE UMAP tradeoff bootstrap pareto efficiency ablation visual comparison degradation cross_verifier
```

### 资产状态

先生成：

`paper_assets/PAPER_ASSET_MANIFEST.md`

每个论文槽位只能标记为：

- `REUSE`：已有同等内容，直接复制/重命名/编排，不重画。
- `REFINE`：已有图和数据逻辑正确，只统一字体、图例、标题、尺寸、panel组合。
- `GENERATE`：仓库没有同等内容，才允许新增。
- `SUPPLEMENT`：已有内容有价值，但放补充材料。

### 严格禁止

- 不重新训练A5、A0、B1、B2、Restormer、PromptIR、DehazeFormer或其他恢复模型。
- 不修改任何`best.pt`。
- 不修改split。
- 不重新生成PLAMD退化图。
- 不为了突出Ours筛选有利样本。
- 不修改真实实验值。
- 不把1792 views当作1792个独立source做统计推断。
- 已有同等图表时，不得重新设计另一套。

允许：从已有CSV/JSON做离线汇总、bootstrap、组合panel、统一命名和轻量排版精修。

---

# 1. 最终主文结构

主文固定为：

- 7 Figures
- 5 Tables

Supplementary放cross-verifier完整表、bootstrap完整表、Adaptive Anchor、历史Sibling、更多visual case。

论文叙事顺序必须是：

```text
Motivation
→ Method
→ Main Restoration Quality
→ Per-Degradation Results
→ Qualitative Results
→ Identity Preservation
→ Restoration–Identity Trade-off
→ Ablation
→ Efficiency
```

---

# FIGURE 1 — Motivation and Problem Definition

论文位置：Introduction。

## 作用

说明同一个电力设备source在7种退化下同时产生pixel degradation和identity drift，而UAV部署还要求轻量化。

## 审计

先搜索已有`motivation / overview / PLAMD / seven degradation / intro figure / degradation montage`。

若已有类似拼图：`REUSE`或`REFINE`，不要重新随机选一套样本。

## 最终布局

顶部：

```text
Clean | Blur | Haze | Inpainting | Low-light | Noise | Rain | Snow
```

中部概念流：

```text
Heterogeneous degradation
        ↓
Pixel corruption + identity drift
        ↓
Restoration
```

底部三个目标：

```text
Restoration Fidelity: PSNR↑ / SSIM↑ / LPIPS↓
Identity Fidelity: Top1↑ / EER↓ / Cosine↑
Efficiency: Params↓ / MACs↓ / Latency↓
```

Figure1不画详细网络。

输出：`fig1_motivation.png/pdf/svg`

---

# FIGURE 2 — Overall Architecture

论文位置：Method。

## 作用

准确展示最终A5：轻量恢复主干 + Degradation Awareness + Identity Anchor，并明确identity verifier仅训练阶段使用。

## 审计

搜索`architecture / framework / method / network / diagram`。如果已有架构图，不推翻主体逻辑，只做精修。

## 最终逻辑

```text
Degraded UAV Image
      ↓
Lightweight Encoder
      ↓
Shared Restoration Feature
   ┌──┴──┐
   │     │
Restoration Path   Degradation-aware representation
   │     │
   └──┬──┘
      ↓
Lightweight Decoder
      ↓
Restored Image
      └ - - - - - - → Frozen Identity Verifier
                         ↓
                    Clean-source Anchor
                         ↓
                   Identity Constraint
                    [Training Only]
```

Frozen verifier支路必须虚线并标`Training Only`，避免读者误以为部署时需要verifier。

输出：`fig2_method_overview.png/pdf/svg`

---

# TABLE 1 — Main Comparison

论文位置：Experiments → Main Results。

## 作用

主表只讲恢复质量 + 主identity指标，不塞完整效率。

方法按类型分组：

```text
General Restoration
All-in-One Restoration
Ours
```

列：

```text
Method | PSNR↑ | SSIM↑ | LPIPS↓ | Top1↑ | EER↓
```

优先读取`EXPERIMENT_DATA_SUMMARY.md`和现有results。

已知关键值：

```text
A5: 28.791 / 0.9458 / 0.1433 / 0.6981 / 0.0799
DehazeFormer LPIPS = 0.1376
```

注意：A5不是LPIPS第一。禁止错误加粗。

排版：booktabs、无竖线、最佳bold、第二名underline。

输出：`table1_main_comparison.tex/csv/md`

---

# TABLE 2 — Seven-Degradation Quantitative Comparison

论文位置：Experiments → Per-Degradation Results。

## 作用

做成高密度AiOIR风格大表。

列按7退化分组：

```text
Blur | Haze | Inpainting | Low-light | Noise | Rain | Snow | Average
```

每个退化下：`PSNR / SSIM`。

如果双栏宽度不足，可拆成Table2(a)/(b)，但仍作为同一张表。

主文这张表不塞Top1/EER，identity单独处理。

只读取已有per-degradation结果，不重新评估。

输出：`table2_per_degradation.tex`

---

# FIGURE 3 — Seven-Degradation Qualitative Comparison

论文位置：Experiments → Qualitative Results。

## 审计

先搜索`visual_comparison / qualitative / comparison_grid / restored_images`。

若已有定性图，直接用现有图做基础；不要重新随机选样本。

## 最终布局

Rows = 7 degradations：

```text
Blur
Haze
Inpainting
Low-light
Noise
Rain
Snow
```

Columns：

```text
Input | Restormer | PromptIR | DehazeFormer | Ours | GT
```

每行至少一个局部放大区域，重点观察：

```text
power line
insulator edge
metal fitting
small structural detail
texture boundary
```

如果某baseline没有现成输出，先审计已有inference结果；确实不存在则标记缺口，禁止为了做图重新训练。

输出：`fig3_visual_comparison.png/pdf`

---

# TABLE 3 — Identity Preservation Comparison

论文位置：Experiments → Identity Preservation。

## 作用

作为identity方向的主表。

至少包含：

```text
Degraded
Restormer
DehazeFormer
Ours
```

如PromptIR已有完整identity指标可加入。

列：

```text
Top1↑ | AUC↑ | EER↓ | Own-anchor Cosine↑ | Margin↑
```

使用主identity verifier结果。

已有A5 aggregate：

```text
Degraded: Top1 0.303, EER 0.169, AUC 0.908, Cosine 0.884, Margin -0.097
A5:       Top1 0.698, EER 0.080, AUC 0.975, Cosine 0.978, Margin -0.012
```

Cross-verifier不单独放主文表，只在正文/caption保留一句：

```text
An independently trained ResNet18 verifier further confirms the transferability of the identity-preserving effect, with A5 achieving the lowest EER of 0.01707.
```

完整cross-verifier表放Supplementary Table S1。

输出：`table3_identity_preservation.tex`

---

# FIGURE 4 — Identity Embedding + Identity Recovery

论文位置：Experiments → Identity Preservation。

## 这是优先复用项目

检查：

```text
artifacts/identity_evaluation/visualization/
artifacts/identity_evaluation/paper_figures/
embedding_2d.csv
identity_cosine*
identity_recovery*
```

如果已有degraded/Restormer/Ours PCA以及identity cosine/recovery图，不重新做另一套随机embedding。

## 最终4-panel

```text
(a) Degraded embedding
(b) Restormer embedding
(c) Ours embedding
(d) Per-degradation identity recovery
```

前三图规则：

```text
颜色 = source ID
marker = degradation
星号 = clean anchor
```

如果当前PCA图已经固定source、固定投影，则优先REUSE。不要仅因UMAP更好看就重新换算法。

Panel(d)优先直接读取已有identity cosine结果。

已知示例：

```text
Haze: 0.7958 → 0.9366, Gain +0.1408
Snow: 0.6165 → 0.9380, Gain +0.3215
```

如果已有Restormer per-degradation cosine则加入三条线：`Degraded / Restormer / Ours`。如果没有，但已有per-view CSV可离线汇总，允许汇总，不允许重跑恢复模型。

输出：`fig4_identity_analysis.png/pdf`

---

# FIGURE 5 — Restoration–Identity Trade-off

论文位置：Experiments → Analysis。

## 目标

证明Pixel Fidelity和Identity Fidelity并非完全等价目标。

对比：A5 vs B1。

## 审计

搜索：

```text
tradeoff delta_psnr delta_rank quadrant Spearman joint improvement
```

如果已有双散点图，直接以现有图为基础升级，不重新定义统计逻辑。

## 最终4-panel

### (a) View-level effects

```text
1792 views
x = PSNR_A5 - PSNR_B1
y = rank_B1 - rank_A5
颜色 = 7 degradation types
```

四象限显示count和percentage：

```text
Q1 Joint improvement
Q2 Identity-only improvement
Q3 Both degraded
Q4 PSNR-only improvement
```

### (b) Source-level effects

```text
256 source means
```

必须显示：

```text
Spearman rho
source-level bootstrap 95% CI
```

### (c) Outcome composition

100% stacked bar：

```text
Per-view
Per-source
```

四段对应Q1-Q4。

### (d) Degradation-wise joint improvement

7退化forest plot，横轴Joint-improvement ratio，误差线为source-level bootstrap 95% CI。

如果已有CSV，直接读取；不要重新定义threshold。

输出：`fig5_restoration_identity_tradeoff.png/pdf`

---

# TABLE 4 — Core 2×2 Ablation

论文位置：Experiments → Ablation。

行固定：

```text
A0
B1
B2
A5
```

列：

```text
Deg | Anchor | PSNR↑ | SSIM↑ | LPIPS↓ | Top1↑ | EER↓
```

使用仓库现有真实结果：

```text
A0: 28.441 / 0.9428 / 0.1563 / 0.6462 / 0.0842
B1: 28.749 / 0.9464 / 0.1464 / 0.6735 / 0.0804
B2: 27.914 / 0.9389 / 0.1637 / 0.6434 / 0.0884
A5: 28.791 / 0.9458 / 0.1433 / 0.6981 / 0.0799
```

按真实最佳值加粗，不要因为A5是Ours就全部加粗。

输出：`table4_ablation.tex`

---

# FIGURE 6 — Quality–Efficiency–Identity Pareto

论文位置：Experiments → Efficiency。

## 审计

搜索`pareto / efficiency / bubble / model_metrics / params / MACs`。已有Pareto图则REUSE或REFINE。

## 数据源

优先：

```text
results/model_metrics.json
EXPERIMENT_DATA_SUMMARY.md
已有main result CSV/JSON
```

方法至少：

```text
Restormer
PromptIR
DehazeFormer
DFPIR
AirNet
SwinIR
Ours
```

## 绘图定义

```text
x = Params(M)
y = PSNR(dB)
bubble size = MACs(G)
bubble color = Top1
```

突出Ours：

```text
2.63M Params
25.52G MACs
28.791 dB
Top1 0.6981
```

不要通过筛除不利方法制造优势。

输出：`fig6_quality_efficiency_identity_pareto.png/pdf`

---

# TABLE 5 — Efficiency Comparison

论文位置：Experiments → Efficiency。

列：

```text
Method | Params(M)↓ | MACs(G)↓ | Latency(ms)↓ | VRAM(GB)↓ | Throughput(ips)↑ | PSNR↑
```

至少包含：

```text
Ours
Restormer
PromptIR
DehazeFormer
SwinIR
AirNet
```

Ours已知：

```text
2.63M / 25.52G / 46.42ms / 0.20GB / 21.54ips / 28.791dB
```

输出：`table5_efficiency.tex`

---

# FIGURE 7 — Ablation and Interaction Analysis

论文位置：Experiments → Ablation / Discussion。

## 目标

把已有零散消融串联起来，不新增训练：

```text
Deg贡献恢复质量
Anchor负责身份约束
Deg + Anchor取得最佳综合结果
Adaptive Anchor未超过Fixed
过强Anchor损害PSNR
Sibling output consistency已被证明有害
```

## 审计

搜索：

```text
ablation anchor_weight adaptive_anchor sibling A0 B1 B2 A5 C1 C2
```

## 推荐3-panel

(a) Deg × Anchor interaction：A0/B1/B2/A5，展示PSNR和Top1。

(b) Anchor sensitivity：已有`lambda=0.01`和`lambda=0.1`时，只画真实2点或2-bar，不伪造中间lambda，不画假曲线。

(c) Design history：A0 / A1 sibling / A3 deg+anchor+sibling / A5 deg+anchor no sibling，解释最终为什么移除Sibling output consistency。

旧Sibling不要再作为最终核心贡献。

输出：`fig7_ablation_interaction.png/pdf`

---

# 2. Supplementary

## Table S1 — Cross-Verifier Generalization

完整独立ResNet18结果：

```text
Degraded 0.9437 / 0.9297 / 0.9916 / 0.03718
Restormer 0.9663 / 0.9621 / 0.9962 / 0.02300
DehazeFormer 0.9716 / 0.9738 / 0.9976 / 0.01732
A5 0.9705 / 0.9654 / 0.9972 / 0.01707
```

列：Cosine / Top1 / ROC-AUC / EER。

注意：A5只在EER上最好，不要错误加粗其他列。

## Table S2 — Source-level Bootstrap

至少保留：

```text
Ours vs Restormer
Ours vs DehazeFormer
```

报告delta和95%CI。

## Table S3 — Adaptive Anchor

C1/C2完整结果。

## Table S4 — Historical Sibling Ablation

A0/A1/A2/A3/A4/A5。

## Figure S1 — More Qualitative Cases

复用现有额外case。

## Figure S2 — Full Per-Degradation Identity Metrics

完整cosine/margin等。

## Figure S3 — Additional Embedding Visualization

已有PCA则直接复用。

---

# 3. 全局视觉规范

所有图统一：

```text
white background
300 dpi
PDF优先vector
SVG保留编辑版
Times New Roman for English
LaTeX风格公式
```

禁止：重阴影、glow、玻璃效果、3D、装饰性渐变。

如果已有图风格已经形成，不因“统一”而完全重画。优先保留现有palette和版式。

建立：`paper_assets/style_manifest.json`

固定方法视觉编码：

```text
Ours
Restormer
DehazeFormer
PromptIR
DFPIR
Degraded
Clean
```

固定7种退化的颜色/marker映射，并在Fig4/Fig5/Fig7中保持一致。

---

# 4. 统一输出目录

```text
paper_assets/
├── main/
│   ├── figures/
│   └── tables/
├── supplementary/
│   ├── figures/
│   └── tables/
├── source_data/
├── style_manifest.json
├── PAPER_ASSET_MANIFEST.md
├── PAPER_ASSET_INDEX.md
├── CAPTIONS.md
└── FINAL_PAPER_ASSET_AUDIT.md
```

对于REUSE资产：只复制到目标目录，保留原文件不动，并在manifest中记录原路径。

---

# 5. 自动生成索引和caption

生成`paper_assets/PAPER_ASSET_INDEX.md`：

| ID | Section | File | Reused from | Data source | Main/Supp | Ready |
|---|---|---|---|---|---|---|

每个图表加一句功能描述。

生成`paper_assets/CAPTIONS.md`，每个caption三句：

1. 这张图/表展示什么。
2. 如何读图。
3. 核心结果是什么。

禁止使用未经支持的`best on all metrics`、`significantly better`等表述。

---

# 6. 最终验收Gate

必须逐项PASS：

```text
Gate A — Reuse-first
已有等价工作全部复用，没有因为“更漂亮”重复生成。

Gate B — Data fidelity
每个图表数值都能追溯到CSV/JSON/MD。

Gate C — No retraining
没有启动任何恢复模型训练。

Gate D — Cross-verifier honesty
没有声称A5在独立ResNet18全部指标第一。

Gate E — LPIPS honesty
没有声称A5 LPIPS第一。

Gate F — Statistical unit
bootstrap按source_id进行。

Gate G — Story order
Quality → Identity → Trade-off → Ablation → Efficiency。
```

最终生成：`paper_assets/FINAL_PAPER_ASSET_AUDIT.md`

必须回答：

```text
1. 哪些图完全复用了已有工作？
2. 哪些图只是REFINE/组合？
3. 哪些图确实新生成？
4. 哪些表直接来自已有结果？
5. 是否发生任何重新训练？
6. 是否存在数据缺口？
7. 7 Figures + 5 Tables是否全部READY？
8. Supplementary包含哪些防审稿内容？
9. 每个主文图表对应哪个核心论点？
```

如果某素材缺数据：明确标`MISSING_DATA`，不要自行训练或伪造。

---

# 7. 最终论文核心故事

整套素材必须让读者自然得到：

```text
为什么电力巡检恢复需要身份保持
        ↓
方法如何联合退化感知和source identity约束
        ↓
恢复质量是否足够强
        ↓
七种退化是否都成立
        ↓
身份是否确实被恢复
        ↓
identity结论是否跨verifier成立
        ↓
Pixel fidelity与Identity fidelity是否是不同目标
        ↓
Deg和Anchor为什么需要组合
        ↓
最终模型是否真正轻量
```

最终主线固定为：

**High Restoration Fidelity + Identity Preservation + Lightweight Deployment**

如果论文名称继续使用SiblingRestore，则Sibling应作为“同源多退化数据关系/训练背景”解释，不再把已被消融证明有害的Sibling output consistency作为最终核心模块。
