# Adaptive Anchor vs Fixed Anchor 消融总结

**日期**: 2026-08-31  
**协议**: seed=13, batch=4, sibling=2, crop=256, 20000 steps, dim=48, verifier 冻结  
**tau**: 0.10  
**评估**: test split (256 sources, 1792 views), frozen verifier evaluator + LPIPS(AlexNet, 256x256)

---

## 1. 结果表

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Top1 ↑ | EER ↓ | idMargin ↑ |
|-------|---------|-------:|-------:|--------:|-------:|------:|-----------:|
| A0    | baseline            | **28.441** | 0.9428 | 0.1563 | 0.6462 | 0.0842 | +0.0810 |
| B1    | deg only            | 28.749 | 0.9464 | 0.1464 | 0.6735 | 0.0804 | +0.0853 |
| B2    | fixed anchor only   | 27.914 | 0.9389 | 0.1637 | 0.6434 | 0.0884 | +0.0795 |
| A5    | deg + fixed anchor  | **28.791** | 0.9458 | 0.1433 | **0.6981** | **0.0799** | +0.0846 |
| C1    | adaptive only       | 28.069 | 0.9412 | 0.1555 | 0.6328 | 0.0890 | +0.0808 |
| C2    | deg + adaptive      | 28.602 | 0.9444 | 0.1502 | 0.6702 | 0.0826 | +0.0844 |

---

## 2. Comparison A: A0 → B2 → C1（无 Deg，固定 vs 自适应）

| 路径 | PSNR 变化 | Top1 变化 | EER 变化 |
|------|----------:|----------:|---------:|
| A0 → B2 (加 Fixed) | **-0.527** | -0.0028 | +0.0042 |
| A0 → C1 (加 Adaptive) | **-0.372** | -0.0134 | +0.0048 |
| B2 → C1 (Fixed→Adaptive) | **+0.155** | -0.0106 | +0.0006 |

**回答 Q1: Fixed Anchor 是否伤害 restoration？**
✅ **是，确认伤害**。B2 vs A0 下降 0.527 dB PSNR，同时 Top1 无改善（-0.003）、EER 变差（+0.004）。单独使用 Fixed Anchor 只有害无利。

**回答 Q2: Adaptive Anchor 是否缓解这种副作用？**
⚠️ **部分缓解**。C1 比 B2 PSNR 恢复 +0.155 dB（仍有 -0.372 vs A0 的净损失），说明 adaptive 的"选择性施压"确实减轻了过度约束。但身份指标没有改善（Top1 -0.011, EER +0.001），**在无 Deg 场景下 adaptive 无法帮助身份保持**。

---

## 3. Comparison B: B1 → A5 → C2（有 Deg，固定 vs 自适应）

| 路径 | PSNR 变化 | Top1 变化 | EER 变化 |
|------|----------:|----------:|---------:|
| B1 → A5 (加 Fixed) | +0.042 | **+0.0246** | **-0.0005** |
| B1 → C2 (加 Adaptive) | -0.147 | -0.0033 | +0.0022 |
| A5 → C2 (Fixed→Adaptive) | **-0.189** | **-0.0279** | **+0.0027** |

**回答 Q3: Adaptive Anchor 是否改善 Top1/EER？**
❌ **没有**。在 Deg 辅助下，Fixed Anchor（A5）才是身份提升的关键（Top1 +0.025, EER -0.001），而 Adaptive（C2）反而丢失了全部身份收益（Top1 -0.003, EER +0.002），且 PSNR 也比 A5 低 0.19 dB。

---

## 4. 机制诊断：为什么 Adaptive 表现不佳

训练诊断（20000 步全程）：

| 指标 | C1 | C2 |
|------|----|----|
| active_ratio 均值 | 0.057 | 0.057 |
| active_ratio 最后 100 步 | **0.019** | **0.021** |
| mean margin 均值 | 0.587 | 0.588 |
| mean weight 均值 | 0.042 | 0.042 |

**根因：tau=0.10 过于严格**。真实 margin 分布集中在 0.59 附近（远大于 tau），导致 98% 的样本 weight=0。Adaptive Anchor 实际只对 ~2-5% 的"最危险"样本施加约束，其余样本完全没有 anchor 监督——等价于一个极弱化版的 Fixed Anchor。

这解释了：
- C1 vs B2：约束少了 → PSNR 伤害减小（+0.155），但身份收益也没了（Top1 -0.011）
- C2 vs A5：约束少了 → 身份收益全丢（Top1 -0.028），且 Deg 与几乎不存在的 anchor 组合后没有 synergy

**结论**：Adaptive 机制本身逻辑正确（选择性地对高风险样本施压），但 tau 选择不当使它在训练后期几乎完全关闭。

---

## 5. 判定标准评估

| 情况 | 条件 | 结果 |
|------|------|------|
| 情况 A | C2 PSNR ≥ A5 且 Top1↑ 或 EER↓ | ❌ PSNR -0.19, Top1 -0.028 |
| 情况 B | C2 PSNR 降 ≤0.05 且 source 明显改善 | ❌ 降 0.19, source 反而变差 |
| 情况 C | C1 相比 B2 恢复 PSNR 且保持/改善 Top1/EER | ⚠️ PSNR +0.155 ✓，但 Top1 -0.011 ✗ |

**回答 Q4: C2 是否值得进入下一阶段 3-seed 实验？**
❌ **不建议在当前 tau=0.10 下进入 3-seed**。C2 在 PSNR 和身份指标上均不优于 A5。

---

## 6. 建议（如果继续 Adaptive 方向）

1. **tau 需要匹配真实 margin 分布**（~0.5-0.7），当前 0.10 导致 active ratio 仅 ~2%
2. 可用 **margin 分位数自适应 tau**（如 batch 内 margin 的 30th 百分位），保证稳定 ~30% active ratio
3. 或改用 **soft weight**（如 sigmoid 形）而非 hard clip，避免梯度断崖
4. 若目标是身份保持，Fixed Anchor（A5, 28.791/0.698）仍是当前最佳配置

---

## 7. 结论

1. **Fixed Anchor 确实伤害 restoration**（单独使用 -0.53 dB PSNR）✅ 确认
2. **Adaptive Anchor 部分缓解了 PSNR 副作用**（+0.155 dB vs Fixed）⚠️
3. **Adaptive Anchor 未改善 Top1/EER**，在 Deg 组合下反而丢失 Fixed 的全部身份收益 ❌
4. **C2 在当前 tau 下不值得进入 3-seed**；如需继续，先修 tau / 权重函数 ❌

当前最佳配置维持 **A5 (deg + fixed anchor, tau=0.01 权重)**：PSNR 28.791 / Top1 0.6981 / EER 0.0799。
