# Identity Evaluation Extensions

- Evaluation split: SFR v1 test (256 sources, 1792 views)
- Bootstrap unit: `source_id`
- Bootstrap iterations: `10000`
- Restoration inference rerun: **no**

## Per-Degradation Identity Cosine

| Method | Degradation | Degraded Cosine | Restored Cosine | Gain | N |
|---|---|---:|---:|---:|---:|
| A5 | blur | 0.991395 | 0.992833 | 0.000443 | 256 |
| A5 | haze | 0.795792 | 0.936580 | 0.120249 | 256 |
| A5 | inpainting | 0.902365 | 0.997809 | 0.086711 | 256 |
| A5 | lowlight | 0.921723 | 0.985948 | 0.055554 | 256 |
| A5 | noise | 0.968912 | 0.997438 | 0.023449 | 256 |
| A5 | rain | 0.990396 | 0.997319 | 0.005287 | 256 |
| A5 | snow | 0.616457 | 0.937954 | 0.300199 | 256 |
| Restormer | blur | 0.991395 | 0.994340 | 0.001590 | 256 |
| Restormer | haze | 0.795792 | 0.922979 | 0.107946 | 256 |
| Restormer | inpainting | 0.902365 | 0.998145 | 0.087557 | 256 |
| Restormer | lowlight | 0.921723 | 0.969339 | 0.040991 | 256 |
| Restormer | noise | 0.968912 | 0.997027 | 0.023095 | 256 |
| Restormer | rain | 0.990396 | 0.995201 | 0.003512 | 256 |
| Restormer | snow | 0.616457 | 0.932149 | 0.295533 | 256 |
| DehazeFormer | blur | 0.991395 | 0.990414 | -0.001940 | 256 |
| DehazeFormer | haze | 0.795792 | 0.912192 | 0.099176 | 256 |
| DehazeFormer | inpainting | 0.902365 | 0.999232 | 0.088471 | 256 |
| DehazeFormer | lowlight | 0.921723 | 0.979140 | 0.050145 | 256 |
| DehazeFormer | noise | 0.968912 | 0.998009 | 0.023557 | 256 |
| DehazeFormer | rain | 0.990396 | 0.988415 | -0.001975 | 256 |
| DehazeFormer | snow | 0.616457 | 0.933730 | 0.299459 | 256 |

## Source-Level Bootstrap

| Comparison | Metric | Mean Delta | Lower 95% | Upper 95% | Sources |
|---|---|---:|---:|---:|---:|
| A5_vs_Restormer | psnr | 0.685956 | 0.592162 | 0.780491 | 256 |
| A5_vs_Restormer | ssim | 0.002462 | 0.001389 | 0.003569 | 256 |
| A5_vs_Restormer | lpips | -0.006277 | -0.007793 | -0.004797 | 256 |
| A5_vs_Restormer | restored_to_clean_top1 | 0.041853 | 0.023996 | 0.059710 | 256 |
| A5_vs_Restormer | restored_margin | 0.004524 | 0.001282 | 0.007772 | 256 |
| A5_vs_Restormer | identity_gain_margin | 0.004524 | 0.001362 | 0.007785 | 256 |
| A5_vs_DehazeFormer | psnr | 0.418827 | 0.317280 | 0.519959 | 256 |
| A5_vs_DehazeFormer | ssim | 0.002424 | 0.001258 | 0.003578 | 256 |
| A5_vs_DehazeFormer | lpips | 0.005777 | 0.003849 | 0.007689 | 256 |
| A5_vs_DehazeFormer | restored_to_clean_top1 | 0.027344 | 0.007812 | 0.046875 | 256 |
| A5_vs_DehazeFormer | restored_margin | 0.005000 | 0.001382 | 0.008551 | 256 |
| A5_vs_DehazeFormer | identity_gain_margin | 0.005000 | 0.001378 | 0.008670 | 256 |
