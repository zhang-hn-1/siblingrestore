# SiblingRestore 实验数据总表

更新时间：2026-09-06

本文档汇总当前工作区中已经完成或有完整记录的 SiblingRestore 训练与评估实验。不同实验批次的数据规模、训练步数、模型结构和 verifier 版本并不完全相同，表格中的协议说明用于避免跨批次误比较。

## 1. 结论摘要

当前最强候选是 **Safe-refine**：以 M3 ALCRB checkpoint 初始化，加入受限 gated residual 和 identity-safe verifier distillation。

在当前 `plamd_sfr_v1`、`seed13`、独立 frozen verifier、`v2 + LPIPS` 口径下：

- Val：PSNR `28.2325`，Restored Top-1 `0.7344`，ROC-AUC `0.9726`，EER `0.0735`，LPIPS `0.1457`。
- Test：PSNR `28.8228`，Restored Top-1 `0.6802`，ROC-AUC `0.9729`，EER `0.0837`，LPIPS `0.1442`。
- 相比原始 M3 ALCRB，Safe-refine 在当前 val 和 test 上的主要指标均同方向改善。

Safe-refine 是从 M3 继续训练的模型，不属于 M0-M7 从零训练的严格公平对照组；正式的稳定性结论仍应补充多 seed 和独立 embedding 验证。

## 2. 当前主线：M0-M7 与 Safe-refine

### 2.1 训练摘要

共同协议：`plamd_sfr_v1`、`sibling`、`seed13`、`formal_12k`、12,000 steps、crop 256。

| 模型 | 配置 | Best step | Best PSNR | Last PSNR | 状态 |
|---|---|---:|---:|---:|---|
| M0 | Fixed baseline | 12000 | 27.2517 | 27.2517 | 完成 |
| M1 | Transformer refine | 12000 | 27.3176 | 27.3176 | 完成 |
| M2 | NAF refine | 12000 | 27.3796 | 27.3796 | 完成 |
| M3 | ALCRB | 11500 | 27.4819 | 27.4731 | 完成 |
| M4 | RERH | 12000 | 27.3542 | 27.3542 | 完成 |
| M5 | LMRB | 11000 | 27.0728 | 26.9955 | 完成 |
| M6 | HFRB | 12000 | 27.4559 | 27.4559 | 完成 |
| M7 | ALCRB + HFRB | 12000 | 27.2879 | 27.2879 | 完成 |
| **Safe-refine** | M3 + gated residual + identity-safe loss | **12000** | **28.2325** | **28.2325** | 完成 |

### 2.2 Val：独立 frozen verifier

评估规模：128 sources、896 restored views。身份指标使用 `v2` evaluator 和 frozen verifier；LPIPS 使用 AlexNet 256x256 协议。

| 模型 | PSNR ↑ | SSIM ↑ | Restored Top-1 ↑ | ROC-AUC ↑ | EER ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| M0 baseline | 27.2517 | 0.93568 | 0.6897 | 0.9659 | 0.08043 | 0.17376 |
| M1 Transformer | 27.3176 | 0.93615 | 0.6998 | **0.9672** | 0.08193 | 0.17345 |
| M2 NAF | 27.3796 | 0.93688 | 0.6987 | 0.9662 | 0.08044 | 0.17049 |
| **M3 ALCRB** | **27.4819** | **0.93791** | **0.7009** | 0.9669 | 0.08314 | **0.16668** |
| M4 RERH | 27.3542 | 0.93703 | 0.6987 | 0.9671 | **0.07946** | 0.17175 |
| M5 LMRB | 27.0728 | 0.93463 | 0.6719 | 0.9659 | 0.08527 | 0.17195 |
| M6 HFRB | 27.4559 | 0.93703 | 0.6987 | 0.9660 | 0.08347 | 0.16925 |
| M7 ALCRB + HFRB | 27.2879 | 0.93605 | 0.6920 | 0.9666 | 0.08080 | 0.17079 |
| **Safe-refine** | **28.2325** | **0.94557** | **0.7344** | **0.9726** | **0.07354** | **0.14565** |

Safe-refine 相对 M3 val：PSNR `+0.7506 dB`、SSIM `+0.00766`、Top-1 `+0.03348`、ROC-AUC `+0.00570`、EER `-0.00960`、LPIPS `-0.02103`。

### 2.3 Test：M3 与 Safe-refine

评估规模：256 sources、1792 restored views。两者使用同一 frozen verifier 和同一 `v2 + LPIPS` 评估口径。

| 指标 | M3 ALCRB | Safe-refine | 变化 |
|---|---:|---:|---:|
| PSNR ↑ | 28.0478 | **28.8228** | **+0.7750** |
| SSIM ↑ | 0.93720 | **0.94536** | **+0.00816** |
| Restored Top-1 ↑ | 0.63170 | **0.68025** | **+0.04855** |
| ROC-AUC ↑ | 0.96678 | **0.97292** | **+0.00614** |
| EER ↓ | 0.09394 | **0.08373** | **-0.01021** |
| LPIPS ↓ | 0.16757 | **0.14420** | **-0.02337** |

### 2.4 Test：逐退化 M3 对比 Safe-refine

| 退化 | M3 PSNR | Safe PSNR | PSNR 变化 | M3 Top-1 | Safe Top-1 | Top-1 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 28.2530 | 28.6233 | +0.3703 | 0.6602 | 0.7227 | +0.0625 |
| Haze | 23.3583 | 24.4291 | +1.0708 | 0.2812 | 0.3633 | +0.0820 |
| Inpainting | 30.4534 | 30.9804 | +0.5269 | 0.8594 | 0.8906 | +0.0312 |
| Lowlight | 28.7129 | 29.8966 | +1.1838 | 0.4805 | 0.5273 | +0.0469 |
| Noise | 27.5706 | 27.8758 | +0.3052 | 0.8477 | 0.8594 | +0.0117 |
| Rain | 33.0390 | 33.9951 | +0.9561 | 0.8750 | 0.9102 | +0.0352 |
| Snow | 24.9475 | 25.9594 | +1.0120 | 0.4180 | 0.4883 | +0.0703 |

收益最大的退化是 Lowlight、Haze、Snow 和 Rain；Blur、Noise 的 PSNR 提升相对较小，但 Top-1 仍然改善。

## 3. 历史核心消融实验

协议：`plamd_sfr_v1`、单 seed13、约 20,000 steps、256 test sources / 1792 views。该协议与当前 12k M0-M7 主线不同。

| 模型 | 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Top-1 ↑ | EER ↓ |
|---|---|---:|---:|---:|---:|---:|
| A0 | rec + grad | 28.4412 | 0.94285 | 0.15629 | 0.6462 | 0.08416 |
| A1 | rec + grad + sibling | 27.9525 | 0.93950 | 0.15779 | 0.6456 | 0.08904 |
| A2 | rec + grad + degradation | 28.5478 | 0.94353 | 0.14996 | 0.6713 | 0.08387 |
| A3 | Deg + Anchor + Sibling | 28.5512 | 0.94297 | 0.14768 | 0.6808 | 0.08198 |
| **A5** | **Deg + Anchor，无 sibling** | **28.7905** | **0.94577** | **0.14334** | **0.6981** | **0.07989** |
| B1 | Deg only | 28.7486 | 0.94640 | 0.14645 | 0.6735 | 0.08039 |
| B2 | Anchor only | 27.9137 | 0.93888 | 0.16369 | 0.6434 | 0.08838 |
| C1 | Adaptive Anchor only | 28.0692 | 0.94123 | 0.15555 | 0.6328 | 0.08895 |
| C2 | Deg + Adaptive Anchor | 28.6019 | 0.94437 | 0.15016 | 0.6702 | 0.08257 |

### 核心结论

- Deg 是主要 PSNR 贡献：A0 → B1 约 `+0.3074 dB`。
- Anchor 单独使用会损害 PSNR：A0 → B2 约 `-0.5275 dB`。
- Deg + Anchor 组合后取得最佳结果，说明两者存在协同效应。
- Anchor 权重 `0.01` 有效；提高到 `0.1` 时 val PSNR 降至约 `27.34`，训练被停止。
- Sibling output consistency 会损害恢复质量：A1 低于 A0；A5 去掉 sibling loss 后更好。

## 4. 旧版 500 集基线对比

协议：`plamd_vari_grip_500`，355 train / 80 val / 65 test，6 种退化，旧版统一 frozen verifier。该批次不能与当前 7 退化、128/256-source 结果直接横比。

| 模型 | 参数量 | PSNR ↑ | Top-1 ↑ | Margin ↑ | ROC-AUC ↑ | EER ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ours dim64 + anchor 12k | 4.68M | **26.89** | **0.896** | **0.1272** | **0.9896** | **0.0460** |
| Ours dim48 + anchor 12k | 2.63M | 26.64 | 0.885 | 0.1197 | 0.9860 | 0.0524 |
| Restormer | 26.13M | 25.92 | 0.881 | 0.1157 | 0.9851 | 0.0494 |
| PromptIR | 35.59M | 25.85 | 0.881 | 0.1160 | 0.9857 | 0.0497 |
| Ours dim48 + anchor 5k | 2.63M | 25.39 | 0.890 | 0.1222 | 0.9871 | 0.0456 |
| Ours dim48 two-stage anchor | 2.63M | 25.20 | 0.875 | 0.1131 | 0.9842 | 0.0531 |
| Ours dim48 | 2.63M | 25.20 | 0.860 | 0.1070 | 0.9820 | 0.0611 |
| Ours two-stage v2 | 1.19M | 24.84 | 0.885 | 0.1115 | 0.9832 | 0.0525 |
| Ours dim32 incumbent | 1.19M | 24.81 | 0.865 | 0.1041 | 0.9804 | 0.0623 |
| Ours dim32 MSC | 1.19M | 24.64 | 0.835 | 0.0979 | 0.9779 | 0.0717 |
| Ours dim32 + anchor | 1.19M | 24.57 | 0.885 | 0.1168 | 0.9855 | 0.0537 |
| PReNet | 0.46M | 22.57 | 0.887 | 0.1156 | 0.9836 | 0.0492 |
| GRL | 6.45M | 20.00 | 0.894 | 0.1157 | 0.9849 | 0.0509 |

重要说明：GRL 的 crop 128 训练导致恢复强度偏弱，低丢失率不代表真实质量优势。

## 5. Adaptive Anchor 专项实验

| 模型 | 配置 | PSNR | SSIM | LPIPS | Top-1 | EER |
|---|---|---:|---:|---:|---:|---:|
| C1 | Adaptive Anchor only | 28.0692 | 0.94123 | 0.15555 | 0.6328 | 0.08895 |
| C2 | Deg + Adaptive Anchor | 28.6019 | 0.94437 | 0.15016 | 0.6702 | 0.08257 |
| A5 | Deg + fixed Anchor | **28.7905** | **0.94577** | **0.14334** | **0.6981** | **0.07989** |

Adaptive Anchor 当前不如固定 Anchor，主要原因是 `tau=0.10` 过严，active ratio 约为 `2%–6%`，大多数样本几乎没有获得有效的 adaptive supervision。

## 6. v0.2 三 seed 实验

协议：16 sources、6 种退化、seeds `13/37/73`、5,000 steps，使用 source-level cluster bootstrap。该批次用于验证训练流程和随机种子稳定性，不与当前主线直接横比。

| 实验 | PSNR mean ± sample std | SSIM mean ± sample std |
|---|---:|---:|
| group | 24.3297 ± 0.2944 | 0.90277 ± 0.00297 |
| source_0003 | 23.9605 ± 0.7605 | 0.89856 ± 0.00989 |
| degradation_001 | 24.4192 ± 0.0514 | 0.90381 ± 0.00186 |

`degradation_001` 相比 `group` 的平均 PSNR 增益约 `+0.089 dB`，但 95% bootstrap CI 为 `[-0.226, +0.431]`，没有显著性结论。小规模 16-source pilot 的统计效力不足。

## 7. 早期 Pilot

早期 pilot 使用 16 sources / 96 views，主要验证数据适配、训练流程和评估管线，不作为论文指标。

| 实验 | PSNR | SSIM |
|---|---:|---:|
| Pilot independent | 24.3269 | 0.9016 |
| Pilot group | 24.0782 | 0.9008 |
| Pilot sibling | 24.0782 | 0.9008 |
| Pilot degradation | 24.3609 | 0.9025 |

## 8. 数据口径与可比性

| 批次 | 数据规模 | 退化数 | 训练步数 | 主要用途 | 与当前 safe-refine 可比性 |
|---|---:|---:|---:|---|---|
| 早期 Pilot | 16 sources | 6 | 探索性 | 管线验证 | 不可直接横比 |
| v0.2 multiseed | 16 sources | 6 | 5000 | 稳定性探索 | 不可直接横比 |
| 500 集基线 | 355/80/65 train/val/test | 6 | 5000/12000 | 外部模型对比 | 不可直接横比 |
| 核心消融 | 约 1400 train / 256 test | 7 | 约 20000 | Deg/Anchor/Sibling 归因 | 部分可比，训练协议不同 |
| M0-M7 12k | 895/128/256 train/val/test | 7 | 12000 | 模块公平对照 | 当前主线 |
| Safe-refine | 895/128/256 train/val/test | 7 | 12000，M3 初始化 | 当前最佳候选 | 当前主线候选 |

身份指标的正式评估应同时记录：

- verifier checkpoint SHA256；
- 数据包 audit fingerprint；
- 模型 config hash；
- split、source 数量和 restored view 数量；
- 评估脚本版本和 tile 参数。

## 9. 结果文件索引

### 当前主线

- `results/psnr_modules_12k/`
- `results/psnr_safe_refine/m3_alcrb_safe_seed13.13.val.json`
- `results/psnr_safe_refine/m3_alcrb_safe_seed13.13.test.json`
- `configs/psnr_safe_refine_m3/seed13.json`

### 历史实验

- `results/ablation_core/`
- `experiment_data/README.md`
- `runs/baselines_500/COMPARISON_SUMMARY.md`
- `runs/ablation_v02_multiseed/multiseed_report.md`
- `ABLATION_REPORT_FINAL.md`
- `EXPERIMENT_DATA_SUMMARY.md`

## 10. 后续建议

1. 对 M3 和 Safe-refine 分别补 `seed37`、`seed73`，每个 seed 使用对应的 M3 初始化。
2. 用独立于训练 verifier 的第二个 verifier 或外部视觉 embedding 做身份验证。
3. 做 M3 continuation、只训练 gated residual、全量 safe-refine 三组归因实验。
4. 用 source-level paired bootstrap 计算 PSNR、Top-1、ROC-AUC、EER 和 margin 的置信区间。
5. 在最终模型冻结后，再进行一次不参与调参的 test 汇报。

当前最稳妥的表述是：

> 在当前单 seed、固定数据划分和 frozen verifier 口径下，Safe-refine 同时取得了最高的恢复质量和身份保持指标；多 seed 与独立 embedding 验证是下一步需要补充的稳健性证据。
