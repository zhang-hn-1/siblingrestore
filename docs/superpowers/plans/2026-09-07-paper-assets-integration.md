# SiblingRestore Final Paper Assets Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the existing SiblingRestore experiment outputs into the Prompt-defined `paper_assets/` package, reusing valid assets, generating only offline compositions/summary tables/diagrams, and documenting any missing data without retraining or changing experimental values.

**Architecture:** Keep all source data and existing outputs immutable. Add one reproducible offline builder under `analysis/paper_assets/` that reads the frozen CSV/JSON/PNG/PDF assets, writes the requested `paper_assets/` directory, and emits a manifest, index, captions, style manifest, and final audit. Existing `paper_figures_v1` outputs are copied as `REUSE` or composited as `REFINE`; new diagrams and summary plots are explicitly marked `GENERATE`.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib, Pillow, SciPy; existing repository CSV/JSON/PNG/PDF/SVG assets; LaTeX source generation without requiring a LaTeX compiler.

**Spec:** `SiblingRestore_Final_Paper_Assets_Codex_Prompt.md`

## Global Constraints

- Do not train or retrain any model and do not modify any `best.pt`.
- Do not modify split, raw data, real experiment values, or PLAMD degradation images.
- Reuse an equivalent existing asset before generating a new one.
- Bootstrap units are `source_id`, never individual views.
- Mark unavailable content `MISSING_DATA`; never fabricate baseline outputs or claims.
- Keep English paper assets on a white background with restrained colors, vector PDF/SVG where applicable, and 300 dpi PNG exports.
- Never claim A5 is best on all metrics; preserve the LPIPS and cross-verifier caveats.

### Task 1: Freeze the audit inventory

**Files:**
- Read: `SiblingRestore_Final_Paper_Assets_Codex_Prompt.md`
- Read: `outputs/paper_figures_v1/asset_manifest.csv`
- Read: `outputs/paper_figures_v1/tables/*.csv`
- Read: `artifacts/identity_evaluation/**/*.csv`
- Create: `paper_assets/PAPER_ASSET_MANIFEST.md`

- [x] Record the current git SHA, dirty-worktree state, all candidate assets, their source paths, and their reuse status.
- [x] Classify every main-paper slot as `REUSE`, `REFINE`, `GENERATE`, or `MISSING_DATA` using the exact definitions in the Prompt.
- [x] Record known data gaps: no complete Restormer/PromptIR/DehazeFormer seven-row restored-image grid is currently stored in `unified_per_image.csv` or `sample_restored/`.

### Task 2: Add the reproducible paper-assets builder

**Files:**
- Create: `analysis/paper_assets/build_paper_assets.py`
- Create: `analysis/paper_assets/README.md`

- [x] Create `paper_assets/main/{figures,tables}`, `paper_assets/supplementary/{figures,tables}`, and `paper_assets/source_data`.
- [x] Copy existing assets without changing source files; write `style_manifest.json` with fixed method/degradation encodings.
- [x] Generate or refine only offline compositions: Figure 1 motivation montage, Figure 2 architecture diagram, Figure 4 identity composite, Figure 6 Pareto, Figure 7 ablation interaction.
- [x] Copy the existing rich trade-off figure into the requested Figure 5 names and copy existing valid tables into source-data and target locations.
- [x] Write missing-data records instead of fabricating Figure 3 baseline outputs.

### Task 3: Generate the five main tables from frozen evidence

**Files:**
- Create through builder: `paper_assets/main/tables/table1_main_comparison.{csv,tex,md}`
- Create through builder: `paper_assets/main/tables/table2_per_degradation.{csv,tex,md}`
- Create through builder: `paper_assets/main/tables/table3_identity_preservation.{csv,tex,md}`
- Create through builder: `paper_assets/main/tables/table4_ablation.{csv,tex,md}`
- Create through builder: `paper_assets/main/tables/table5_efficiency.{csv,tex,md}`

- [x] Use `unified_per_image.csv` and official aggregate JSON for Tables 1/3/4/5.
- [x] Use existing per-degradation quality results for Table 2.
- [x] Apply bold/underline only by actual metric ranking and preserve LPIPS/cross-verifier caveats.
- [x] Include exact source paths and denominators in table captions or adjacent metadata.

### Task 4: Assemble supplementary assets and narrative metadata

**Files:**
- Create: `paper_assets/PAPER_ASSET_INDEX.md`
- Create: `paper_assets/CAPTIONS.md`
- Create: `paper_assets/FINAL_PAPER_ASSET_AUDIT.md`
- Create through builder: `paper_assets/supplementary/tables/table_s1_cross_verifier.csv`
- Create through builder: `paper_assets/supplementary/tables/table_s2_source_bootstrap.csv`
- Reuse: `artifacts/identity_evaluation/paper_figures/*`

- [x] Add cross-verifier results with only the true best EER emphasis.
- [x] Add source-level bootstrap and adaptive-anchor/historical-sibling tables as supplementary evidence.
- [x] Add one three-sentence caption for every main figure/table and a source path for every quantitative artifact.
- [x] Map every main slot to `READY` or `MISSING_DATA`, with a concrete reason and non-fabricating next step.

### Task 5: Validate and audit the package

**Files:**
- Validate: `paper_assets/`
- Validate: `analysis/paper_assets/build_paper_assets.py`

- [x] Run the builder from a clean command and verify all output files are non-empty.
- [x] Recompute key counts: 1792 views, 256 sources, 7 degradations; verify all percentages and bootstrap CIs.
- [x] Verify figures have correct panel titles, no forbidden debug/PCA fallback language, 300 dpi PNG, and vector PDF/SVG where required.
- [x] Verify no training command was run, no checkpoint/raw data file changed, and the audit answers all nine Prompt questions.
