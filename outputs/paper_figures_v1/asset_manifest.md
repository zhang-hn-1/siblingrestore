# 资产清单与可用性报告（paper_figures_v1）

生成时间见文件 mtime；协议：test 256 sources × 7 退化 = 1792 views；
verifier sha256 `d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6`；LPIPS alex_256x256；
恢复图实物（restored images）全项目均未落盘 → `restored_images_saved=False`，
图 2/3/5 需要恢复图的环节将用已有 checkpoint 补跑少量推理。

| method | cohort | 训练元信息 | ckpt | 汇总json | 逐图csv | sources分 | ckpt_sha256(前12) | PSNR | Top1 | EER |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | ablation_core | rec+grad; mode=sibling; 20k | Y | Y | Y | Y | `819ed9b271e4` | 28.4412 | 0.6462 | 0.0842 |
| A1 | ablation_core | rec+grad+sibling; mode=sibling; 20k | Y | Y | Y | Y | `0abcf9591530` | 27.9525 | 0.6456 | 0.0890 |
| A2 | ablation_core | rec+grad+deg; mode=sibling; 20k | Y | Y | Y | Y | `e744accbe42e` | 28.5478 | 0.6713 | 0.0839 |
| A3 | ablation_core | rec+grad+deg+anchor+sibling; mode=sibling; 20k | Y | Y | Y | Y | `3600aea0c693` | 28.5512 | 0.6808 | 0.0820 |
| A5 | ablation_core | rec+grad+deg+anchor; mode=sibling; 20k | Y | Y | Y | Y | `221250dc3332` | 28.7905 | 0.6981 | 0.0799 |
| B1 | ablation_core | rec+grad+deg; mode=sibling; 20k | Y | Y | Y | Y | `2924dc96b84d` | 28.7486 | 0.6735 | 0.0804 |
| B2 | ablation_core | rec+grad+anchor; mode=sibling; 20k | Y | Y | Y | Y | `4151b7f6bb79` | 27.9137 | 0.6434 | 0.0884 |
| C1 | ablation_core | rec+grad+adaptive_anchor; mode=sibling; 20k | Y | Y | Y | Y | `d0d66ae881cf` | 28.0692 | 0.6328 | 0.0890 |
| C2 | ablation_core | rec+grad+deg+adaptive_anchor; mode=sibling; 20k | Y | Y | Y | Y | `9f2bcb31f3bc` | 28.6019 | 0.6702 | 0.0826 |
| airnet | c005 | 12k independent | Y | Y | Y | Y | `386282930220` | 24.7851 | 0.4442 | 0.1357 |
| clearair | c005 | 12k independent | Y | Y | Y | Y | `e6b40930cb85` | 21.0695 | 0.2584 | 0.1896 |
| dehazeformer | c005 | 12k independent | Y | Y | Y | Y | `9d775373ef7e` | 28.3717 | 0.6708 | 0.0942 |
| dfpir | c005 | 12k independent | Y | Y | Y | Y | `dfa4d7c6cf4c` | 27.2762 | 0.5954 | 0.0998 |
| ffanet | c005 | 12k independent | Y | Y | Y | Y | `cdbe92d4fc95` | 26.6121 | 0.5301 | 0.1082 |
| grl | c005 | 12k independent | Y | Y | Y | Y | `4a5f17626df9` | 20.4327 | 0.3092 | 0.1678 |
| ours_dim48_anchor | c005 | 12k independent | Y | Y | Y | Y | `75f34474ef12` | 28.7192 | 0.6786 | 0.0827 |
| ours_dim64_anchor | c005 | 12k independent | Y | Y | Y | Y | `e0f2184ae81a` | 28.1141 | 0.6702 | 0.0849 |
| prenet | c005 | 12k independent | Y | Y | Y | Y | `ce83f215be02` | 24.3834 | 0.3711 | 0.1660 |
| promptir | c005 | 12k independent | Y | Y | Y | Y | `fce760bb5b3f` | 27.5255 | 0.6334 | 0.0944 |
| r2r | c005 | 12k independent | Y | Y | Y | Y | `7f293730d0e0` | 26.1536 | 0.5519 | 0.1117 |
| restormer | c005 | 12k independent | Y | Y | Y | Y | `cdc94e587131` | 28.1046 | 0.6562 | 0.0902 |
| swinir | c005 | 12k independent | Y | Y | Y | Y | `e355ba1c50ee` | 24.7790 | 0.3912 | 0.1591 |
| transweather | c005 | 12k independent | Y | Y | Y | Y | `f752218e787d` | 23.7907 | 0.3181 | 0.1522 |
| uformer | c005 | 12k independent | Y | Y | Y | Y | `99b434a0782c` | 25.6141 | 0.4883 | 0.1383 |
| M3_alcrb | psnr_modules_12k | 12k, formal, M3 ALCRB | Y | Y | Y | Y | `34ab20a22eac` | 28.0478 | 0.6317 | 0.0939 |
| m3_alcrb_safe_seed13 | psnr_safe_refine | 12k, safe-refine (M3 init) | Y | Y | Y | Y | `70fa82f52e39` | 28.8228 | 0.6802 | 0.0837 |
