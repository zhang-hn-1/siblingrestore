# FINAL IDENTITY EVALUATION REPORT

- Generated: 2026-09-07T14:24:10.359562

## Integrity checks

| Check | Pass | Detail |
|---|---|---|
| restoration_A5_checkpoint | PASS | best.pt: 221250dc333234802cf51a90517e2780311ea26a59d1bd22d37c93ee9e5165d7 |
| restoration_Restormer_checkpoint | PASS | best.pt: cdc94e587131c2afc4028cadd14c9ff25b0a6bb4ce526b6aabcf77ecacb9ec15 |
| restoration_DehazeFormer_checkpoint | PASS | best.pt: 9d775373ef7ef4f9bef66eb483be04e93839b1c875c1dab91df4294983a60a01 |
| evaluation_isolated | PASS | all new outputs under artifacts/identity_evaluation |
| cross_verifier_present | PASS | artifacts/identity_evaluation/cross_verifier/cross_verifier_resnet18.pt |
| cross_verifier_independent | PASS | cross sha256 02dcbf4dfb50d63a06c55a83d49d6ca15ba9bdabb1a0d20b24c5c3457706e023 |
| bootstrap_unit_source | PASS | 12 bootstrap rows |
| cosine_rows_complete | PASS | 21 rows |
| cosine_seven_degradations | PASS | ['blur', 'haze', 'inpainting', 'lowlight', 'noise', 'rain', 'snow'] |
| cross_verifier_results | PASS | 4 methods |
| figure_identity_cosine_per_degradation | PASS | png+pdf |
| figure_identity_recovery_gain | PASS | png+pdf |
| figure_bootstrap_confidence_intervals | PASS | png+pdf |
| test_only_evaluation | PASS | cross-verifier trained on train split; restoration frozen |

**Overall: PASS (14/14 checks passed)**

