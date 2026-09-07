# table4_ablation

Core 2x2 Deg x Anchor ablation. All rows use the existing sibling-group protocol and seed; no new training was performed.

| Config | Deg | Anchor | PSNR | SSIM | LPIPS | Top1 | EER |
|---|---|---|---|---|---|---|---|
| A0 | No | No | 28.441 | 0.9428 | 0.1563 | 0.6462 | 0.0842 |
| B1 | Yes | No | 28.749 | 0.9464 | 0.1464 | 0.6735 | 0.0804 |
| B2 | No | Yes | 27.914 | 0.9389 | 0.1637 | 0.6434 | 0.0884 |
| A5 | Yes | Yes | 28.791 | 0.9458 | 0.1433 | 0.6981 | 0.0799 |
