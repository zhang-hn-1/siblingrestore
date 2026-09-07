# Statistical Significance Report

## 1. Statistical unit and protocol

- Independent unit: source; 256 clean sources, 7 jointly resampled views per source, 1792 total views.
- Paired bootstrap: 10,000 repetitions, with replacement, seed 13, percentile 95% CI.
- Positive Δ means Ours is better for every metric; LPIPS/EER use baseline minus Ours.

## 2. Data integrity audit

- Required methods: Ours, Restormer, DehazeFormer, PromptIR; valid sources: 256.
- Missing statistical samples: 0; protocol issues: 0.
- Same source IDs, seven degradations, test-only split, and official frozen verifier were checked.

## 3. Aggregate sanity check

- Ours: PASS.
  - PSNR recomputed 28.79050786 vs official 28.79050786 (difference 0.00000000).
  - SSIM recomputed 0.94577006 vs official 0.94577006 (difference 0.00000000).
  - LPIPS recomputed 0.14333723 vs official 0.14333723 (difference 0.00000000).
  - Top1 recomputed 0.69810268 vs official 0.69810268 (difference 0.00000000).
  - AUC recomputed 0.97493616 vs official 0.97493615 (difference 0.00000001).
  - EER recomputed 0.07988774 vs official 0.07988774 (difference 0.00000000).
- Restormer: PASS.
  - PSNR recomputed 28.10455210 vs official 28.10455210 (difference 0.00000000).
  - SSIM recomputed 0.94330823 vs official 0.94330823 (difference 0.00000000).
  - LPIPS recomputed 0.14961439 vs official 0.14961439 (difference 0.00000000).
  - Top1 recomputed 0.65625000 vs official 0.65625000 (difference 0.00000000).
  - AUC recomputed 0.97016282 vs official 0.97016282 (difference 0.00000000).
  - EER recomputed 0.09017529 vs official 0.09017529 (difference 0.00000000).
- DehazeFormer: PASS.
  - PSNR recomputed 28.37168088 vs official 28.37168088 (difference 0.00000000).
  - SSIM recomputed 0.94334599 vs official 0.94334599 (difference 0.00000000).
  - LPIPS recomputed 0.13756044 vs official 0.13756044 (difference 0.00000000).
  - Top1 recomputed 0.67075893 vs official 0.67075893 (difference 0.00000000).
  - AUC recomputed 0.96919127 vs official 0.96919126 (difference 0.00000001).
  - EER recomputed 0.09418658 vs official 0.09418658 (difference 0.00000000).
- PromptIR: PASS.
  - PSNR recomputed 27.52552907 vs official 27.52552907 (difference 0.00000000).
  - SSIM recomputed 0.93806412 vs official 0.93806412 (difference 0.00000000).
  - LPIPS recomputed 0.15789029 vs official 0.15789029 (difference 0.00000000).
  - Top1 recomputed 0.63337054 vs official 0.63337054 (difference 0.00000000).
  - AUC recomputed 0.96740844 vs official 0.96740843 (difference 0.00000001).
  - EER recomputed 0.09438682 vs official 0.09438682 (difference 0.00000000).

## 4. Table 7 headline results

- Ours_vs_Restormer PSNR: Δ=+0.685956, 95% CI [0.592275, 0.779444], significant=True.
- Ours_vs_Restormer LPIPS: Δ=+0.006277, 95% CI [0.004772, 0.007771], significant=True.
- Ours_vs_Restormer Top1: Δ=+0.041853, 95% CI [0.023996, 0.059152], significant=True.
- Ours_vs_Restormer Margin: Δ=+0.004524, 95% CI [0.001378, 0.007706], significant=True.
- Ours_vs_DehazeFormer PSNR: Δ=+0.418827, 95% CI [0.318155, 0.521848], significant=True.
- Ours_vs_DehazeFormer LPIPS: Δ=-0.005777, 95% CI [-0.007689, -0.003866], significant=False.
- Ours_vs_DehazeFormer Top1: Δ=+0.027344, 95% CI [0.007812, 0.046875], significant=True.
- Ours_vs_DehazeFormer Margin: Δ=+0.005000, 95% CI [0.001398, 0.008607], significant=True.
- Ours_vs_PromptIR PSNR: Δ=+1.264979, 95% CI [1.142639, 1.384093], significant=True.
- Ours_vs_PromptIR LPIPS: Δ=+0.014553, 95% CI [0.012748, 0.016322], significant=True.
- Ours_vs_PromptIR Top1: Δ=+0.064732, 95% CI [0.045201, 0.084277], significant=True.
- Ours_vs_PromptIR Margin: Δ=+0.005617, 95% CI [0.002236, 0.009189], significant=True.
- Aggregate metrics whose 95% CI crosses zero: none.
- Aggregate metrics with a CI entirely above zero: Ours_vs_Restormer/PSNR, Ours_vs_Restormer/SSIM, Ours_vs_Restormer/LPIPS, Ours_vs_Restormer/Top1, Ours_vs_Restormer/Cosine, Ours_vs_Restormer/Margin, Ours_vs_DehazeFormer/PSNR, Ours_vs_DehazeFormer/SSIM, Ours_vs_DehazeFormer/Top1, Ours_vs_DehazeFormer/Cosine, Ours_vs_DehazeFormer/Margin, Ours_vs_PromptIR/PSNR, Ours_vs_PromptIR/SSIM, Ours_vs_PromptIR/LPIPS, Ours_vs_PromptIR/Top1, Ours_vs_PromptIR/Cosine, Ours_vs_PromptIR/Margin, Ours_vs_Restormer/AUC, Ours_vs_Restormer/EER, Ours_vs_DehazeFormer/AUC, Ours_vs_DehazeFormer/EER, Ours_vs_PromptIR/AUC, Ours_vs_PromptIR/EER.
- Aggregate metrics with a CI entirely below zero (baseline favored under the Δ convention): Ours_vs_DehazeFormer/LPIPS.

## 5. Full aggregate results

All eight requested metrics are retained in `full_statistical_results.csv`; no non-significant metric was hidden from the supplementary analysis.

## 6. Per-degradation analysis

Per-degradation source-level CIs cover PSNR, SSIM, LPIPS, Top1, Cosine, and Margin. AUC/EER remain aggregate gallery-level metrics and are recomputed at aggregate bootstrap level, not split into pseudo-source values.

- Non-significant per-degradation natural-metric comparisons: 65/126.
- Per-degradation CIs crossing zero: Ours_vs_Restormer/blur/LPIPS, Ours_vs_Restormer/blur/Top1, Ours_vs_Restormer/blur/Cosine, Ours_vs_Restormer/blur/Margin, Ours_vs_Restormer/haze/SSIM, Ours_vs_Restormer/haze/Top1, Ours_vs_Restormer/haze/Margin, Ours_vs_Restormer/inpainting/Top1, Ours_vs_Restormer/lowlight/LPIPS, Ours_vs_Restormer/noise/LPIPS, Ours_vs_Restormer/noise/Top1, Ours_vs_Restormer/noise/Cosine, Ours_vs_Restormer/noise/Margin, Ours_vs_Restormer/rain/Top1, Ours_vs_Restormer/snow/Cosine, Ours_vs_Restormer/snow/Margin, Ours_vs_DehazeFormer/blur/PSNR, Ours_vs_DehazeFormer/blur/SSIM, Ours_vs_DehazeFormer/blur/Top1, Ours_vs_DehazeFormer/blur/Cosine, Ours_vs_DehazeFormer/blur/Margin, Ours_vs_DehazeFormer/haze/Top1, Ours_vs_DehazeFormer/inpainting/Top1, Ours_vs_DehazeFormer/lowlight/Top1, Ours_vs_DehazeFormer/noise/Top1, Ours_vs_DehazeFormer/noise/Margin, Ours_vs_DehazeFormer/snow/SSIM, Ours_vs_DehazeFormer/snow/Top1, Ours_vs_DehazeFormer/snow/Cosine, Ours_vs_DehazeFormer/snow/Margin, Ours_vs_PromptIR/blur/LPIPS, Ours_vs_PromptIR/blur/Top1, Ours_vs_PromptIR/blur/Cosine, Ours_vs_PromptIR/blur/Margin, Ours_vs_PromptIR/haze/SSIM, Ours_vs_PromptIR/haze/Top1, Ours_vs_PromptIR/haze/Cosine, Ours_vs_PromptIR/haze/Margin, Ours_vs_PromptIR/inpainting/Top1, Ours_vs_PromptIR/inpainting/Cosine, Ours_vs_PromptIR/inpainting/Margin, Ours_vs_PromptIR/noise/PSNR, Ours_vs_PromptIR/noise/SSIM, Ours_vs_PromptIR/rain/Top1.
- Per-degradation CIs entirely below zero (baseline favored): Ours_vs_Restormer/inpainting/PSNR, Ours_vs_Restormer/inpainting/SSIM, Ours_vs_Restormer/inpainting/LPIPS, Ours_vs_Restormer/inpainting/Cosine, Ours_vs_Restormer/inpainting/Margin, Ours_vs_Restormer/noise/PSNR, Ours_vs_Restormer/noise/SSIM, Ours_vs_DehazeFormer/blur/LPIPS, Ours_vs_DehazeFormer/inpainting/PSNR, Ours_vs_DehazeFormer/inpainting/SSIM, Ours_vs_DehazeFormer/inpainting/LPIPS, Ours_vs_DehazeFormer/inpainting/Cosine, Ours_vs_DehazeFormer/inpainting/Margin, Ours_vs_DehazeFormer/lowlight/LPIPS, Ours_vs_DehazeFormer/noise/PSNR, Ours_vs_DehazeFormer/noise/SSIM, Ours_vs_DehazeFormer/noise/LPIPS, Ours_vs_DehazeFormer/noise/Cosine, Ours_vs_PromptIR/inpainting/PSNR, Ours_vs_PromptIR/inpainting/SSIM, Ours_vs_PromptIR/inpainting/LPIPS.

## 7. Multiple comparisons

Supplementary raw p-values use a centered-bootstrap two-sided tail calculation: the bootstrap distribution is centered by subtracting the observed point estimate, and both tails at least as extreme as the observed statistic are doubled conservatively. Benjamini-Hochberg q-values are computed over aggregate and per-degradation tests.

## 8. Source bootstrap versus naive view bootstrap

The source-level result is the formal analysis. The naive view-level comparison is diagnostic only and resamples 1792 views independently. If its CI is narrower, that is evidence that treating correlated degradation views as independent underestimates uncertainty.
- Across 18 natural-metric diagnostics, naive view-level CIs are narrower in 10 and wider in 8; the effect is not uniform, but independent-view resampling does not preserve source clustering and is not used for the paper result.
- Ours_vs_Restormer/PSNR: source CI width 0.187169; view CI width 0.216188; view/source ratio 1.155.
- Ours_vs_Restormer/SSIM: source CI width 0.002161; view CI width 0.002150; view/source ratio 0.995.
- Ours_vs_Restormer/LPIPS: source CI width 0.003000; view CI width 0.002930; view/source ratio 0.977.
- Ours_vs_Restormer/Top1: source CI width 0.035156; view CI width 0.036272; view/source ratio 1.032.
- Ours_vs_Restormer/Cosine: source CI width 0.006662; view CI width 0.006598; view/source ratio 0.990.
- Ours_vs_Restormer/Margin: source CI width 0.006328; view CI width 0.006658; view/source ratio 1.052.
- Ours_vs_DehazeFormer/PSNR: source CI width 0.203693; view CI width 0.218985; view/source ratio 1.075.
- Ours_vs_DehazeFormer/SSIM: source CI width 0.002330; view CI width 0.002394; view/source ratio 1.028.
- Ours_vs_DehazeFormer/LPIPS: source CI width 0.003823; view CI width 0.003465; view/source ratio 0.906.
- Ours_vs_DehazeFormer/Top1: source CI width 0.039062; view CI width 0.038518; view/source ratio 0.986.
- Ours_vs_DehazeFormer/Cosine: source CI width 0.007887; view CI width 0.007755; view/source ratio 0.983.
- Ours_vs_DehazeFormer/Margin: source CI width 0.007210; view CI width 0.007121; view/source ratio 0.988.
- Ours_vs_PromptIR/PSNR: source CI width 0.241455; view CI width 0.316490; view/source ratio 1.311.
- Ours_vs_PromptIR/SSIM: source CI width 0.002614; view CI width 0.002613; view/source ratio 0.999.
- Ours_vs_PromptIR/LPIPS: source CI width 0.003574; view CI width 0.003687; view/source ratio 1.032.
- Ours_vs_PromptIR/Top1: source CI width 0.039076; view CI width 0.041295; view/source ratio 1.057.
- Ours_vs_PromptIR/Cosine: source CI width 0.007062; view CI width 0.006595; view/source ratio 0.934.
- Ours_vs_PromptIR/Margin: source CI width 0.006953; view CI width 0.006662; view/source ratio 0.958.
- AUC/EER are omitted from the naive diagnostic because the formal gallery-level bootstrap is the required evidence and pseudo-source AUC/EER are invalid.

## 9. Paper-safe conclusions

1. Ours improves aggregate PSNR over Restormer by 0.6860 dB (95% CI [0.5923, 0.7794]).
2. LPIPS, Top1, and Margin should be described as statistically significant only for comparisons whose source-level 95% CI is entirely above zero; DehazeFormer has a significantly lower LPIPS than Ours in this frozen test result.
3. The source-level bootstrap preserves the seven correlated degradation views within each source, so its intervals are the appropriate uncertainty statement for this test set.
