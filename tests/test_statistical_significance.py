import numpy as np
import pytest

from analysis.statistical_significance.core import (
    benjamini_hochberg,
    bootstrap_mean_distribution,
    centered_bootstrap_pvalue,
    gallery_bootstrap_distribution,
    gallery_metric,
    improvement,
    source_level_mean,
)


def test_improvement_direction_always_makes_positive_mean_ours_is_better():
    assert improvement(10.0, 9.0, "PSNR") == pytest.approx(1.0)
    assert improvement(0.10, 0.20, "LPIPS") == pytest.approx(0.10)
    assert improvement(0.05, 0.10, "EER") == pytest.approx(0.05)


def test_source_level_mean_aggregates_all_views_before_comparison():
    values = {
        "s1": [1.0, 3.0],
        "s2": [5.0, 7.0],
    }

    result = source_level_mean(values, ["s1", "s2"])

    assert result.tolist() == pytest.approx([2.0, 6.0])


def test_bootstrap_mean_distribution_uses_same_source_indices_for_pair():
    ours = np.array([3.0, 5.0, 7.0])
    baseline = np.array([1.0, 4.0, 6.0])
    indices = np.array([[0, 1, 2], [2, 2, 0]])

    result = bootstrap_mean_distribution(ours, baseline, indices, "PSNR")

    assert result.tolist() == pytest.approx([4.0 / 3.0, 4.0 / 3.0])


def test_gallery_metric_recomputes_auc_and_eer_from_score_blocks():
    # Two sources, one view each, with the own gallery entry as the positive.
    scores = np.array([
        [[0.9, 0.1]],
        [[0.2, 0.8]],
    ])
    assert gallery_metric(scores, [0, 1], "AUC") == pytest.approx(1.0)
    assert gallery_metric(scores, [0, 1], "EER") == pytest.approx(0.0)


def test_gallery_bootstrap_distribution_matches_direct_recomputation():
    scores = np.array([
        [[0.9, 0.1], [0.8, 0.2]],
        [[0.2, 0.8], [0.3, 0.7]],
    ])
    indices = np.array([[0, 1], [1, 1]])

    for metric in ("AUC", "EER"):
        expected = np.array([gallery_metric(scores, sample, metric) for sample in indices])
        assert gallery_bootstrap_distribution(scores, indices, metric).tolist() == pytest.approx(expected)


def test_centered_bootstrap_pvalue_and_bh_fdr_are_two_sided_and_monotone():
    distribution = np.array([0.2, 0.3, 0.4, 0.5])
    assert centered_bootstrap_pvalue(0.35, distribution) == pytest.approx(0.0)
    q_values = benjamini_hochberg(np.array([0.01, 0.04, 0.20]))
    assert q_values.tolist() == pytest.approx([0.03, 0.06, 0.20])
