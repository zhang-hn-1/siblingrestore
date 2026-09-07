# Identity Evaluation Extensions

- Evaluation split: SFR v1 test (256 sources, 1792 views)
- Bootstrap unit: `source_id`
- Bootstrap iterations: `10000`
- Restoration inference rerun: **no**

## Per-Degradation Identity Cosine

| Method | Degradation | Degraded Cosine | Restored Cosine | Gain | N |
|---|---|---:|---:|---:|---:|
| A5 | blur | 0.991395 | 0.992833 | 0.001438 | 256 |
| A5 | haze | 0.795792 | 0.936580 | 0.140788 | 256 |
| A5 | inpainting | 0.902365 | 0.997809 | 0.095445 | 256 |
| A5 | lowlight | 0.921723 | 0.985948 | 0.064225 | 256 |
| A5 | noise | 0.968912 | 0.997438 | 0.028527 | 256 |
| A5 | rain | 0.990396 | 0.997319 | 0.006922 | 256 |
| A5 | snow | 0.616457 | 0.937954 | 0.321497 | 256 |
| Restormer | blur | 0.991395 | 0.994340 | 0.002945 | 256 |
| Restormer | haze | 0.795792 | 0.922979 | 0.127187 | 256 |
| Restormer | inpainting | 0.902365 | 0.998145 | 0.095780 | 256 |
| Restormer | lowlight | 0.921723 | 0.969339 | 0.047617 | 256 |
| Restormer | noise | 0.968912 | 0.997027 | 0.028115 | 256 |
| Restormer | rain | 0.990396 | 0.995201 | 0.004805 | 256 |
| Restormer | snow | 0.616457 | 0.932149 | 0.315691 | 256 |
| DehazeFormer | blur | 0.991395 | 0.990414 | -0.000981 | 256 |
| DehazeFormer | haze | 0.795792 | 0.912192 | 0.116401 | 256 |
| DehazeFormer | inpainting | 0.902365 | 0.999232 | 0.096867 | 256 |
| DehazeFormer | lowlight | 0.921723 | 0.979140 | 0.057418 | 256 |
| DehazeFormer | noise | 0.968912 | 0.998009 | 0.029097 | 256 |
| DehazeFormer | rain | 0.990396 | 0.988415 | -0.001981 | 256 |
| DehazeFormer | snow | 0.616457 | 0.933730 | 0.317273 | 256 |

## Source-Level Bootstrap

| Comparison | Metric | Mean Delta | Lower 95% | Upper 95% | Sources |
|---|---|---:|---:|---:|---:|
| A5_vs_Restormer | psnr | 0.685956 | 0.592014 | 0.779072 | 256 |
| A5_vs_Restormer | ssim | 0.002462 | 0.001445 | 0.003592 | 256 |
| A5_vs_Restormer | lpips | -0.006277 | -0.007778 | -0.004758 | 256 |
| A5_vs_Restormer | restored_to_clean_top1 | 0.041853 | 0.023996 | 0.059710 | 256 |
| A5_vs_Restormer | restored_margin | 0.004524 | 0.001290 | 0.007818 | 256 |
| A5_vs_Restormer | identity_gain_margin | 0.004524 | 0.001346 | 0.007782 | 256 |
| A5_vs_DehazeFormer | psnr | 0.418827 | 0.316441 | 0.520598 | 256 |
| A5_vs_DehazeFormer | ssim | 0.002424 | 0.001289 | 0.003610 | 256 |
| A5_vs_DehazeFormer | lpips | 0.005777 | 0.003861 | 0.007746 | 256 |
| A5_vs_DehazeFormer | restored_to_clean_top1 | 0.027344 | 0.007812 | 0.046317 | 256 |
| A5_vs_DehazeFormer | restored_margin | 0.005000 | 0.001306 | 0.008669 | 256 |
| A5_vs_DehazeFormer | identity_gain_margin | 0.005000 | 0.001318 | 0.008558 | 256 |
