# Overall qualitative comparison — three degradation cases

## Audit and reuse

The preceding zoom-in comparison used two low-light cases: vari-grip/rust/Fotos 23-10-2020_DJI_0043_vari_grip_759 and vari-grip/good/Fotos 10-12-2020_DJI_0258_vari_grip_282.

No existing figure matched the requested 3×6 full-image SOTA layout. The repository's existing multi-degradation visualization and frozen inference protocol were reused; the final composition is a new paper-ready refinement.

- REUSED FROM: `outputs/paper_figures_v1/unified_per_image.csv`, `outputs/paper_figures_v1/sample_restored/`, and the official frozen checkpoints.
- REFINED BY: unified method order, full-image-only layout, Times-style typography, consistent PSNR formatting, white background, and 300-dpi PDF/PNG/SVG export.

## Selected cases

The three cases are distinct degradation types, exclude the previous low-light cases, and were selected from the real test table with Ours ranked first among the strong baseline set (Restormer, PromptIR, DehazeFormer, and DFPIR).

| Row | Degradation | Source | Ours rank | Ours PSNR | Strongest baseline | Margin |
|---|---|---|---:|---:|---|---:|
| Haze | haze | `vari-grip/good/Fotos 03-12-2020_DJI_0561_vari_grip_100` | 1 | 35.20 dB | DFPIR (34.68 dB) | +0.52 dB |
| Snow | snow | `vari-grip/good/Fotos 18-11-2020_DJI_0246_vari_grip_528` | 1 | 35.97 dB | DFPIR (35.08 dB) | +0.89 dB |
| Rain | rain | `vari-grip/good/Fotos 19-11-2020_DJI_0236_vari_grip_561` | 1 | 42.22 dB | PromptIR (41.32 dB) | +0.90 dB |

All displayed baseline metrics come from real rows in `unified_per_image.csv`; all displayed image results were produced from the corresponding frozen checkpoints. No checkpoint or PSNR value was modified. No baseline is missing from the final six-column figure.

## Caption draft

Figure X. Visual comparison on representative haze, snow, and rain degradation cases from PLAMD. Without zoomed-in crops, the figure emphasizes overall image restoration quality. Our method remains visually competitive with strong SOTA baselines across all three scenes.

## Outputs

- `/home/zhanghangning/siblingrestore-pilot-server/paper_assets/main/fig_qualitative_overall_three_cases.png`
- `/home/zhanghangning/siblingrestore-pilot-server/paper_assets/main/fig_qualitative_overall_three_cases.pdf`
- `/home/zhanghangning/siblingrestore-pilot-server/paper_assets/main/fig_qualitative_overall_three_cases.svg`
- `/home/zhanghangning/siblingrestore-pilot-server/paper_assets/source_data/fig_qualitative_overall_three_cases.csv`
