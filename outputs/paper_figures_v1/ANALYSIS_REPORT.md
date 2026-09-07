# SiblingRestore 现有结果整理与论文图表制作——结果分析报告

生成方式：由 `unified_per_image.csv`（26 方法 × 1792 退化图）与官方 `.test.json` 自动生成；
完整字段与协议见 `unified_schema.md`、`PROTOCOL.md`。

## 0. 一句结论
- 主方法 A5（deg+anchor）相对 B1（deg only）的 PSNR 平均增益小（+0.0419 dB，bootstrap 95% CI 见下），
  但恢复后来源检索 Top-1 提升明显（A5−B1 = +0.0246）；
  正确来源排名：A5 优于 B1 的视图 280（15.6%），并列 1267（70.7%），变差 245。
  分退化看，Top-1 增量最大的是 lowlight 与 inpainting；blur/snow 的 Top-1 几乎不变。
  雾与雪仍是来源保持最难的两类退化。A1/A3（sibling 输出一致性）恢复质量更低，说明『输出更一致 ≠ 恢复更好』在本数据上成立。

## 1. 不同方法在七类退化上的恢复质量（问题 1）

见表 `tables/table1_main.csv`（主方法）与 `tables/table1_all.csv`（全部对比方法）。
A5 / B1 每退化均值（PSNR / SSIM / LPIPS / 恢复 Top-1 / 同类别 Top-1，格式 A5/B1）：
| degradation | PSNR | SSIM | LPIPS | Top-1 | SameCat Top-1 |
|---|---|---|---|---|---|
| blur | 28.766 / 28.624 | 0.954 / 0.952 | 0.170 / 0.180 | 0.734 / 0.734 | 0.766 / 0.785 |
| haze | 24.430 / 24.678 | 0.907 / 0.913 | 0.061 / 0.066 | 0.406 / 0.387 | 0.449 / 0.465 |
| inpainting | 31.061 / 31.043 | 0.974 / 0.974 | 0.091 / 0.092 | 0.918 / 0.879 | 0.938 / 0.898 |
| lowlight | 29.659 / 29.290 | 0.974 / 0.971 | 0.134 / 0.142 | 0.617 / 0.539 | 0.672 / 0.621 |
| noise | 27.810 / 27.920 | 0.944 / 0.945 | 0.159 / 0.152 | 0.871 / 0.855 | 0.887 / 0.879 |
| rain | 33.835 / 33.883 | 0.981 / 0.980 | 0.090 / 0.095 | 0.863 / 0.844 | 0.891 / 0.879 |
| snow | 25.971 / 25.802 | 0.887 / 0.889 | 0.298 / 0.299 | 0.477 / 0.477 | 0.539 / 0.523 |

A5 恢复 Top-1 的难度排序：inpainting > rain/noise > blur > lowlight > snow > haze。

## 2. 固定锚点（A5 相对 B1）改善哪些样本的来源保持（问题 2）

配对逐图统计（1792 视图，严格互斥分组）：
- 两项均改善（PSNR↑ 且正确来源排名↑）：176；
- 仅 PSNR 改善、排名并列：648；
- 仅排名改善、PSNR 并列：0；
- 两项均退步：144；
- 排名并列且 PSNR 未升：619；
- 其余混合（PSNR↑排名↓ 或 排名↑PSNR↓）：205。
（汇总：排名改善 280、并列 1267、变差 245；PSNR 升 925、降 867。）
并列/零变化样本不强行归入改善组。
按来源 7 视图块均值聚合：两项均改善的来源 74 个、两项均退步 60 个。

分退化恢复 Top-1 变化（A5 − B1，正值为 A5 改善）：
- blur: +0.0000
- haze: +0.0195
- inpainting: +0.0391
- lowlight: +0.0781
- noise: +0.0156
- rain: +0.0195
- snow: +0.0000

结论：A5 的来源保持收益集中在 lowlight 与 inpainting；haze/rain/noise 有小幅正收益，blur 与 snow 的 Top-1 不变。锚点并不在全部『难』退化上一致改善——具体样本 rank 改善/并列/变差并存（见图 1 与 `cases/fig1_*_delta_counts.csv`）。
## 3. 恢复质量变化与来源保持变化的关系（问题 3）

- view 级 (1792)：ΔPSNR 与 Δrank 相关 0.204（弱正相关）；
- source 级 (256 块均值)：相关 0.157（弱正相关）。
  结论：质量增益与检索排名增益不绑定；两者同时大幅变化的视图少（图 1 中大部分点贴近零参考线）。

## 4. 同一来源跨退化稳定识别（问题 4）

每来源在 7 类退化上的 Top-1 成功次数（256 个完整来源，全部含 7 退化）：

| method | mean(0-7) | 全 7 类成功 | 全失败 |
|---|---:|---:|---:|
| A5 | 4.887 | 0.129 | 0.008 |
| B1 | 4.715 | 0.129 | 0.016 |

A5 的平均成功次数（4.89）高于 B1（4.72），全失败来源更少；全 7 类成功比例两者相同。
两种方法都没有把任何来源在全部 7 类退化中全部丢失。

## 5. 来源保持改善对应的局部细节（问题 5，视觉 + 定量）

- 图 5：rust 来源 Fotos 23-10-2020_DJI_0032_vari_grip_750 的 haze 局部窗口（crop rows [34:226], cols [34:226]），
  展示 input / DehazeFormer / B1 / A5 / clean 及统一色标误差图；窗口按『退化引入差异最大』客观选取，不对恢复输出臆断缺陷标签。
- input 局部窗口 mean abs error = 0.0927
- DehazeFormer 局部窗口 mean abs error = 0.0551
- B1 局部窗口 mean abs error = 0.0354
- A5 局部窗口 mean abs error = 0.0309

局部定量与误差图一致：A5 在恢复质量不劣化的情况下保持细节（误差图 ≤ B1）。

## 6. 既有现象在统一评测下的核对

1) A5 相对 B1 平均 PSNR 增益较小：成立。全测试均值差 +0.0419 dB；源级 bootstrap 95% CI 见 `bootstrap/bootstrap_a5_vs_b1.md`（下界为负）。
2) A5 来源 Top-1 改善集中于 lowlight：部分成立。各退化增量：lowlight +0.078、inpainting +0.039、haze/rain/noise ≈ +0.02、blur/snow ≈ 0。lowlight 最一致。
3) 雾和雪来源混淆更严重：成立。退化输入检索 Top-1 在 haze/snow 最低（haze 0.137、snow 0.148）；恢复后 A5 仍为最难两类（haze 0.406、snow 0.477）；其中 haze 的 Top-1 相对 B1 有提升，snow 未提升。
4) 同源输出更接近时 clean 恢复误差是否同步改善：否。
   - A0: sibling_output_l1 = 0.0434, PSNR = 28.441, EER = 0.0842
   - A1: sibling_output_l1 = 0.0472, PSNR = 27.953, EER = 0.0890
   - A3: sibling_output_l1 = 0.0439, PSNR = 28.551, EER = 0.0820
   - A5: sibling_output_l1 = 0.0424, PSNR = 28.791, EER = 0.0799
   - B1: sibling_output_l1 = 0.0418, PSNR = 28.749, EER = 0.0804
   A1（含 sibling 输出一致性）恢复输出更一致但其 PSNR 低于 A0；A5 去掉 sibling 后 PSNR 最高。输出一致性与 clean 恢复误差并不同步改善。

## 7. 局限与已知缺失

- 训练/评测批次并存：核心消融(A0/B1/B2/A5/A1/A3/C1/C2) 20k step、mode=sibling 同源分组；c005 外部基线 12k step、mode=independent。评测统一于同一 test(256×7)、同一冻结 verifier (sha d9b000…) 与同一 LPIPS 协议；训练预算差异保留在资产清单中，不据此作跨批次强对比。
- 单一训练 seed(13)：bootstrap CI 只反映测试来源抽样不确定性，不代表训练随机性。
- SSIM 为项目全局 masked SSIM 实现（非 windowed），沿用官方数值并在协议注明。
- restored_path 恒空（项目未落盘恢复图）；图 2/3/5 恢复图由既有 checkpoints 补推理生成。
- 图 2/3/5 为指标驱动选图并固定坐标；不基于恢复结果断言缺陷/部件标签。

## 8. 复用与再生成
- 表与图全部由 `analysis/paper_figures_v1/` 脚本从统一数据生成；一条命令见 `run_all.sh`。
- 交叉验证：统一表与官方 json 最大偏差 7.1e-15；一致性核对见 `consistency_report.md`。
