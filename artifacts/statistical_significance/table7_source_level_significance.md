# Table 7. Source-level Statistical Significance

Positive Δ always means Ours is better. Confidence intervals are obtained by paired source-level bootstrap over 256 independent clean sources with 10,000 resamples. All degradation views belonging to the same source are resampled jointly.

| Comparison | Metric | Δ Improvement | 95% CI | Significant |
|---|---|---:|---:|:---:|
| Ours vs Restormer | PSNR | +0.6860 dB | [0.5923, 0.7794] | true |
| Ours vs Restormer | LPIPS | +0.0063 | [0.0048, 0.0078] | true |
| Ours vs Restormer | Top1 | +0.0419 | [0.0240, 0.0592] | true |
| Ours vs Restormer | Margin | +0.0045 | [0.0014, 0.0077] | true |
| Ours vs DehazeFormer | PSNR | +0.4188 dB | [0.3182, 0.5218] | true |
| Ours vs DehazeFormer | LPIPS | -0.0058 | [-0.0077, -0.0039] | false |
| Ours vs DehazeFormer | Top1 | +0.0273 | [0.0078, 0.0469] | true |
| Ours vs DehazeFormer | Margin | +0.0050 | [0.0014, 0.0086] | true |
| Ours vs PromptIR | PSNR | +1.2650 dB | [1.1426, 1.3841] | true |
| Ours vs PromptIR | LPIPS | +0.0146 | [0.0127, 0.0163] | true |
| Ours vs PromptIR | Top1 | +0.0647 | [0.0452, 0.0843] | true |
| Ours vs PromptIR | Margin | +0.0056 | [0.0022, 0.0092] | true |
