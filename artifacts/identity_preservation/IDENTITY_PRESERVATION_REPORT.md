# Identity Preservation Report

## 1. Protocol

- Test split: `test`, 256 sources × 7 degradations = 1792 source-degradation queries.
- Metrics are computed from the official frozen-verifier score vectors; no verifier or restoration model was trained in this analysis.
- Cosine uses the paired clean source; margin uses the hardest wrong clean source from the full gallery.

## 2. Verifier information

- Checkpoint: `/home/zhanghangning/siblingrestore-pilot-server/runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt`
- SHA-256: `d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6`
- Embedding dimension: `128`; normalization: `verifier.embed output is L2-normalized; score is dot product cosine`.
- Preprocessing: `{"input_range": "RGB float32 [0,1]", "verifier_input": "official evaluate_frozen_verifier_v2 verifier_embed_tiled", "tile_size": 512, "tile_overlap": 32, "noise": "clean Lanczos alignment to noise native size per Method B protocol"}`; distance: `cosine similarity / dot product of L2-normalized embeddings`.

## 3. Dataset integrity check

- Required methods: Degraded, Restormer, DehazeFormer, PromptIR, Ours.
- Optional complete methods included in machine-readable results: DFPIR, FFANet, Uformer.
- Missing required samples: 0; protocol issues: 0.
- Gallery: 256 unique test source IDs; train/test overlap: none; every query has one clean-source index.

## 4. Aggregate sanity check

- A5 recomputed vs official: Top1 0.698103 vs 0.698103; AUC 0.974936 vs 0.974936; EER 0.079888 vs 0.079888.
- Protocol sanity result: `PASS` (tolerance 0.005).

## 5. Per-degradation Cosine

See `table3_identity_recovery.md` and `results/identity_preservation/per_degradation_identity.csv`. Ours aggregate mean is 0.977983 (95% CI [0.971939, 0.983501]).

## 6. Per-degradation Margin

Margin is positive when the paired clean source is more similar than every wrong clean source. The supplementary LaTeX table reports mean margins.

## 7. Identity Recovery Gain

- Ours highest gain: Snow (0.321497).
- Ours lowest gain: Blur (0.001438).

## 8. Relative Recovery Ratio

The ratio is `(restored cosine - degraded cosine) / (1 - degraded cosine)` and is summarized per degradation in the JSON/CSV outputs.

## 9. Ours vs baselines

- Ours_vs_Restormer: mean paired ΔCosine difference 0.005243, 95% CI [0.001873, 0.008698], significant=`True`.
- Ours_vs_DehazeFormer: mean paired ΔCosine difference 0.006393, 95% CI [0.002454, 0.010467], significant=`True`.
- Ours_vs_PromptIR: mean paired ΔCosine difference 0.007324, 95% CI [0.003928, 0.010768], significant=`True`.

## 10. Bootstrap confidence intervals

Source-level bootstrap: 2000 resamples, seed 13. Each source contributes a seven-degradation block for the Average row; no individual views are treated as independent in paired comparisons.

## 11. Hardest degradation analysis

Snow has the lowest degraded own-source cosine (0.616457), so it is the hardest by the pre-restoration cosine criterion.

## 12. Easiest degradation analysis

Blur has the highest degraded own-source cosine (0.991395), so it is the easiest by the pre-restoration cosine criterion.

## 13. Paper-ready conclusions

1. Across the seven PLAMD degradations, restoration changes source identity similarity by a measurable amount; the largest Ours gain occurs on Snow (0.3215).
2. The hardest and easiest degradations are data-driven: Snow is lowest before restoration, while Blur is highest.
3. Ours is significantly better than a baseline only where the corresponding source-level paired bootstrap CI is entirely above zero; see `identity_bootstrap_ci.csv` for the exact comparisons.
