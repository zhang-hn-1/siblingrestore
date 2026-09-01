# 数据包说明

本文件夹包含 SiblingRestore 全部实验的核心数据，供论文表格整理使用。

## 目录结构

```
experiment_data/
├── README.md                        ← 本说明
├── ablation/                        ← 消融实验 test 集结果（9 个模型）
│   ├── A0_recon.13.test.json        ← baseline (rec+grad)
│   ├── A1_sibling.13.test.json      ← +Sibling（已证有害，移除）
│   ├── A2_degradation.13.test.json  ← +Deg
│   ├── A3_full_components.13.test.json ← Deg+Anchor+Sibling
│   ├── A5_no_sibling.13.test.json   ← ★ 最佳: Deg+Anchor (28.791)
│   ├── B1_deg_only.13.test.json     ← Deg only
│   ├── B2_anchor_only.13.test.json  ← Anchor only
│   ├── C1_adaptive_anchor_only.13.test.json ← Adaptive only
│   └── C2_deg_adaptive_anchor.13.test.json  ← Deg+Adaptive
├── baselines/                       ← 15 个基线/对比模型 test 集结果
│   ├── airnet / clearair / dehazeformer / dfpir / ffanet / grl
│   ├── ours_dim48_anchor / ours_dim64_anchor  ← 早期版本
│   ├── prenet / promptir / r2r / restormer / swinir
│   └── transweather / uformer
├── metrics/
│   └── model_metrics.json           ← 轻量级指标（Params/MACs/Latency/VRAM/Throughput）
└── adaptive_anchor_ablation/        ← Adaptive Anchor 专项实验
    ├── adaptive_anchor_ablation.csv / .json  ← 6 模型指标
    ├── adaptive_anchor_diagnostics.csv       ← 训练诊断（active ratio/margin/weight）
    └── adaptive_anchor_summary.md            ← 结论报告
```

## 每个 JSON 的 aggregate 字段含义

| 字段 | 含义 |
|------|------|
| `psnr` | 恢复图像 PSNR (dB) ↑ |
| `ssim` | 恢复图像 SSIM ↑ |
| `lpips` | AlexNet 256×256 LPIPS ↓ |
| `restored_to_clean_top1` | frozen verifier Top-1 身份检索率 ↑ |
| `restored_eer` | 等错误率 ↓ |
| `restored_roc_auc` | ROC AUC ↑ |
| `restored_own_anchor_cos` | 与自身 source 锚点余弦相似度 ↑ |
| `restored_margin` | 与最近 impostor 的 margin（越大越安全） |
| `identity_gain_margin` | 恢复后 margin 增益 |
| `per_degradation` | 7 种退化（blur/haze/inpainting/lowlight/noise/rain/snow）分别的指标 |

## 核心结论速览

1. **最佳模型 A5** = rec + grad(0.05) + Deg(0.01) + Anchor(0.01)
   - PSNR 28.791 / SSIM 0.9458 / LPIPS 0.1433 / Top1 0.6981 / EER 0.0799
2. **Deg 是 PSNR 核心贡献**（+0.31 dB），**Anchor 需与 Deg 组合**（协同 +0.57 dB）
3. **Anchor 权重必须小**（0.01 最优，0.1 崩溃 -1.45 dB）
4. **Sibling 模块有害**，已移除
5. **轻量优势**: 2.63M 参数 = restormer 的 1/10，PSNR 反超 +0.69

## 来源

- 训练: `train.py --config configs/ablation_core/*.json`
- 评估: `scripts/evaluate_frozen_verifier_v2.py`（test split, seed 13, LPIPS AlexNet）
- 指标: `scripts/measure_model_metrics.py`（256×256）
