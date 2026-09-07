# CAPTIONS

## Figure 1

This figure motivates the joint problem using one fixed source across clean and seven existing degradations. Read the top row as the corruption family and the lower blocks as the three paper objectives. It establishes why restoration fidelity, identity fidelity, and deployment efficiency must be considered together.

## Figure 2

This diagram shows the final A5 information flow with a lightweight restoration backbone, degradation-aware representation, and identity anchor constraint. Solid arrows are the restoration path and dashed arrows are the frozen verifier branch marked Training Only. The verifier is therefore not presented as a deployment-time dependency.

## Figure 3

The requested complete seven-degradation qualitative baseline grid is not available as stored evidence. The package marks this slot MISSING_DATA and preserves the available selected retrieval cases in supplementary Figure S1 rather than fabricating images.

## Figure 4

This figure combines the fixed embedding projections with the existing per-degradation identity-recovery plot. Colors encode source identity and the recovery panel compares degraded and restored cosine evidence across degradations. The companion cross-verifier table remains supplementary.

## Figure 5

This four-panel analysis compares A5 and B1 at view and source grain. Read positive ΔPSNR and positive ΔRank as simultaneous improvement, with source-bootstrap uncertainty used for source-level summaries. The composition and degradation-wise forest plot expose heterogeneity rather than claiming universal improvement.

## Figure 6

This Pareto view plots recorded parameter count against PSNR, uses bubble area for MACs, and color for restored Top-1. It compares the fixed set of available baselines and highlights the A5/dim-48 efficiency point without filtering unfavorable methods.

## Figure 7

This figure links the core Deg × Anchor ablation, the two observed anchor-weight settings, and the historical sibling design. Read it as an interaction and design-history summary, not as a fitted sensitivity curve.

## Table 1

This table reports the main restoration and identity comparison. Rows are grouped by restoration type and columns contain PSNR, SSIM, LPIPS, Top-1, and EER on the frozen test set. LPIPS is shown honestly because A5 is not the best LPIPS row.

## Table 2

This dense table reports PSNR and SSIM for each of seven degradations and the equal-weight average. Each degradation uses the same 256-source test population. It isolates restoration quality from the separate identity table.

## Table 3

This table reports identity preservation for degraded input, Restormer, DehazeFormer, and A5. Read Top-1/AUC/Cosine/Margin upward and EER downward; the degraded row is a common input reference.

## Table 4

This table reports the fixed A0/B1/B2/A5 2×2 ablation. Deg and Anchor columns expose the design factors while the metric columns show the actual recorded trade-offs.

## Table 5

This table reports recorded parameter, MAC, latency, VRAM, throughput, and PSNR values under the project efficiency protocol. Ours uses the A5 quality row paired with the recorded dim-48 efficiency measurement.
