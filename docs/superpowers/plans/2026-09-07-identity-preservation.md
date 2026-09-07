# Identity Preservation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible Table 3 identity-preservation analysis from the existing official frozen-verifier test outputs.

**Architecture:** A standalone offline analyzer reads official `.test.json` metadata and `.test.json.sources.jsonl` score vectors, validates the shared 256-source test gallery and verifier fingerprint, computes cosine/margin/recovery statistics with source-level bootstrap, and writes paper-ready tables, figure data/plots, protocol metadata, and an evidence-backed report. Existing training, model, Table2, and raw result files remain unchanged.

**Tech Stack:** Python 3, NumPy, pandas, Matplotlib, standard-library JSON/CSV/hashlib, pytest.

**Spec:** `/home/zhanghangning/.codex/attachments/263bf317-8444-4b6b-bcef-1f12b76fc066/pasted-text.txt`

## Global Constraints

- Use the Table2 official frozen verifier: `runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt` with SHA-256 recorded from disk.
- Use test split, one shared clean gallery, seven degradations, and source IDs as the bootstrap unit.
- Main methods are Degraded, Restormer, DehazeFormer, PromptIR, and Ours=A5; optional methods enter only when complete.
- Do not retrain or modify restoration/verifier models, data split, Table2, or existing outputs.
- Never silently skip missing samples; write `missing_identity_samples.csv` and block paper tables on incomplete required data.
- Block paper tables when recomputed A5 aggregate Top1/AUC/EER differs from its official aggregate by more than 0.005.

### Task 1: Test metric primitives

**Files:**
- Create: `tests/test_identity_preservation.py`
- Create: `analysis/identity_preservation/__init__.py`
- Create: `analysis/identity_preservation/core.py`

- [x] Write failing tests for own-source cosine, hardest-negative margin excluding self, recovery gain/ratio, and source-block paired bootstrap.
- [x] Run `pytest tests/test_identity_preservation.py -q` and confirm failure because the new module is absent.
- [x] Implement the smallest NumPy functions needed by the tests.
- [x] Run the focused tests and confirm they pass.

### Task 2: Official-output loader and integrity checks

**Files:**
- Create: `analysis/identity_preservation/analyze_identity_preservation.py`
- Modify: `analysis/identity_preservation/core.py`

- [x] Add explicit method catalog and result paths for the five required methods, with optional DFPIR/FFANet/Uformer discovery.
- [x] Load test groups, validate exactly seven degradation keys, unique source IDs, common source/gallery ordering, required JSON/CSV/score files, and verifier fingerprints.
- [x] Emit every missing required source/degradation/path as `missing_identity_samples.csv`; do not drop incomplete methods silently.
- [x] Re-run focused tests and verify missing-output detection/source alignment against the real official outputs.

### Task 3: Analysis outputs and sanity gate

**Files:**
- Modify: `analysis/identity_preservation/analyze_identity_preservation.py`
- Create through script: `results/identity_preservation/per_degradation_identity.csv`
- Create through script: `results/identity_preservation/per_degradation_identity.json`
- Create through script: `results/identity_preservation/figure6d_identity_recovery.csv`
- Create through script: `results/identity_preservation/identity_bootstrap_ci.csv`
- Create through script: `results/identity_preservation/verifier_protocol.json`

- [x] Aggregate mean/std/count/95% CI for Cosine and Margin per method/degradation.
- [x] Compute A5 and baseline recovery gain/ratio and paired source-level bootstrap comparisons.
- [x] Recompute A5 aggregate Top1/AUC/EER from official scores and write mismatch diagnostics when tolerance fails.
- [x] Serialize stable, explicit JSON schemas and machine-readable CSVs.

### Task 4: Paper artifacts

**Files:**
- Modify: `analysis/identity_preservation/analyze_identity_preservation.py`
- Create through script: `artifacts/identity_preservation/table3_identity_recovery.tex`
- Create through script: `artifacts/identity_preservation/table3_identity_recovery.md`
- Create through script: `artifacts/identity_preservation/table3_identity_margin_supp.tex`
- Create through script: `artifacts/identity_preservation/figure6d_identity_recovery.pdf`
- Create through script: `artifacts/identity_preservation/figure6d_identity_recovery.png`
- Create through script: `artifacts/identity_preservation/IDENTITY_PRESERVATION_REPORT.md`

- [x] Rank only restoration methods per row/metric, bold best and underline second, and keep black/white table styling.
- [x] Plot Figure6(d) from generated CSV, not hard-coded values.
- [x] Write the report sections and 3–5 paper-ready conclusions strictly from computed rankings and intervals.
- [x] Run full tests plus the real-data analyzer and inspect artifact counts, protocol fields, table rows, and figure files.
