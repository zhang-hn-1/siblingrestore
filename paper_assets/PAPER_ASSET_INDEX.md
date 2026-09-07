# PAPER_ASSET_INDEX

| ID | Section | File | Reused from | Data source | Main/Supp | Ready |
|---|---|---|---|---|---|---|
| Figure 1 | Main | `paper_assets/main/figures/fig1_motivation.png` | offline composition / source evidence | data/PLAMD_SFR_v1_method_b paths from unified_per_image.csv | Main | READY |
| Figure 2 | Method | `paper_assets/main/figures/fig2_method_overview.png` | offline composition / source evidence | Prompt-defined A5 architecture logic | Main | READY |
| Figure 3 | Qualitative | `paper_assets/main/figures/fig3_visual_comparison_MISSING_DATA.md` | offline composition / source evidence | No complete 7-degradation baseline restored-image grid | Main | MISSING_DATA |
| Figure 4 | Identity | `paper_assets/main/figures/fig4_identity_analysis.png` | artifacts/identity_evaluation/visualization + paper_figures | artifacts/identity_evaluation/visualization + paper_figures | Main | READY |
| Figure 5 | Trade-off | `paper_assets/main/figures/fig5_restoration_identity_tradeoff.png` | outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.* | outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.* | Main | READY |
| Figure 6 | Efficiency | `paper_assets/main/figures/fig6_quality_efficiency_identity_pareto.png` | offline composition / source evidence | results/TEST_summary_table.csv + experiment_data/metrics/model_metrics.json | Main | READY |
| Figure 7 | Ablation | `paper_assets/main/figures/fig7_ablation_interaction.png` | offline composition / source evidence | artifacts/adaptive_anchor_ablation + unified_per_image.csv | Main | READY |
| Table 1 | Main Results | `paper_assets/main/tables/table1_main_comparison.tex` | offline composition / source evidence | unified_per_image.csv + official aggregate JSON | Main | READY |
| Table 2 | Per-Degradation | `paper_assets/main/tables/table2_per_degradation.tex` | unified_per_image.csv | unified_per_image.csv | Main | READY |
| Table 3 | Identity | `paper_assets/main/tables/table3_identity_preservation.tex` | offline composition / source evidence | unified_per_image.csv + official aggregate JSON | Main | READY |
| Table 4 | Ablation | `paper_assets/main/tables/table4_ablation.tex` | artifacts/adaptive_anchor_ablation/adaptive_anchor_ablation.csv | artifacts/adaptive_anchor_ablation/adaptive_anchor_ablation.csv | Main | READY |
| Table 5 | Efficiency | `paper_assets/main/tables/table5_efficiency.tex` | offline composition / source evidence | experiment_data/metrics/model_metrics.json + unified_per_image.csv | Main | READY |

Supplementary assets are listed under `paper_assets/supplementary/` and include cross-verifier, source-bootstrap, adaptive-anchor, historical-sibling, identity-recovery, and embedding evidence.
