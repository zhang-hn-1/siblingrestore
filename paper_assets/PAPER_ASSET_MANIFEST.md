# PAPER_ASSET_MANIFEST

Frozen git HEAD: `81c15a1fa6261aef685b16a4e72767242d05e34a`

| Slot | Section | Status | Output | Data/source | Ready |
|---|---|---|---|---|---|
| Figure 1 | Main | GENERATE | paper_assets/main/figures/fig1_motivation.png | data/PLAMD_SFR_v1_method_b paths from unified_per_image.csv | READY |
| Figure 2 | Method | GENERATE | paper_assets/main/figures/fig2_method_overview.png | Prompt-defined A5 architecture logic | READY |
| Figure 3 | Qualitative | MISSING_DATA | paper_assets/main/figures/fig3_visual_comparison_MISSING_DATA.md | No complete 7-degradation baseline restored-image grid | MISSING_DATA |
| Figure 4 | Identity | REFINE | paper_assets/main/figures/fig4_identity_analysis.png | artifacts/identity_evaluation/visualization + paper_figures | READY |
| Figure 5 | Trade-off | REFINE | paper_assets/main/figures/fig5_restoration_identity_tradeoff.png | outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.* | READY |
| Figure 6 | Efficiency | GENERATE | paper_assets/main/figures/fig6_quality_efficiency_identity_pareto.png | results/TEST_summary_table.csv + experiment_data/metrics/model_metrics.json | READY |
| Figure 7 | Ablation | GENERATE | paper_assets/main/figures/fig7_ablation_interaction.png | artifacts/adaptive_anchor_ablation + unified_per_image.csv | READY |
| Table 1 | Main Results | GENERATE | paper_assets/main/tables/table1_main_comparison.tex | unified_per_image.csv + official aggregate JSON | READY |
| Table 2 | Per-Degradation | REFINE | paper_assets/main/tables/table2_per_degradation.tex | unified_per_image.csv | READY |
| Table 3 | Identity | GENERATE | paper_assets/main/tables/table3_identity_preservation.tex | unified_per_image.csv + official aggregate JSON | READY |
| Table 4 | Ablation | REFINE | paper_assets/main/tables/table4_ablation.tex | artifacts/adaptive_anchor_ablation/adaptive_anchor_ablation.csv | READY |
| Table 5 | Efficiency | GENERATE | paper_assets/main/tables/table5_efficiency.tex | experiment_data/metrics/model_metrics.json + unified_per_image.csv | READY |

## Audit notes

- Raw data, split, checkpoints, and true experimental values were not modified.
- No training or model evaluation was run by this asset assembly.
- Bootstrap claims retain source_id as the statistical unit.
- Figure 3 is explicitly MISSING_DATA because the complete requested baseline restored-image grid is not stored.
- The pre-existing detailed source-retrieval cases are preserved as supplementary Figure S1.

## Worktree snapshot

```text
?? SiblingRestore_Final_Paper_Assets_Codex_Prompt.md
?? analysis/paper_assets/
?? analysis/paper_figures_v1/11_tradeoff_rich.py
?? docs/
?? evaluation/export_sfid_restored.py
?? outputs/paper_figures_v1/bootstrap/degradation_joint_improvement.csv
?? outputs/paper_figures_v1/bootstrap/tradeoff_bootstrap_ci.csv
?? outputs/paper_figures_v1/bootstrap/tradeoff_summary.csv
?? outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.pdf
?? outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.png
?? outputs/paper_figures_v1/figures/restoration_identity_tradeoff_rich.svg
?? outputs/paper_table2/
?? paper_assets/
```
