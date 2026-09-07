# Table 3. Per-Degradation Identity Recovery

Values are means over the 256-source test split. Bold/underline indicate the best/second restoration method per row; Degraded is a reference and is not ranked.

| Degradation | Degraded Cosine↑ | Restormer Cosine↑ | DehazeFormer Cosine↑ | PromptIR Cosine↑ | Ours Cosine↑ | Ours ΔCosine↑ |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 0.9914 | **0.9943** | 0.9904 | 0.9927 | _0.9928_ |
| Haze | 0.7958 | 0.9230 | 0.9122 | _0.9364_ | **0.9366** |
| Inpainting | 0.9024 | _0.9981_ | **0.9992** | 0.9979 | 0.9978 |
| Low-light | 0.9217 | 0.9693 | _0.9791_ | 0.9590 | **0.9859** |
| Noise | 0.9689 | 0.9970 | **0.9980** | 0.9949 | _0.9974_ |
| Rain | 0.9904 | _0.9952_ | 0.9884 | 0.9944 | **0.9973** |
| Snow | 0.6165 | 0.9321 | _0.9337_ | 0.9193 | **0.9380** |
| Average | 0.8839 | _0.9727_ | 0.9716 | 0.9707 | **0.9780** |
