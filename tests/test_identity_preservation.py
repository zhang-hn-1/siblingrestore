import numpy as np
import pytest

from analysis.identity_preservation.core import (
    bootstrap_mean_ci,
    identity_metrics,
    paired_bootstrap_ci,
    recovery_metrics,
)


def test_identity_metrics_uses_own_source_and_hardest_wrong_gallery_entry():
    query = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=float)
    gallery = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=float)

    result = identity_metrics(query, gallery, own_indices=[0, 1])

    assert result["cosine"].tolist() == pytest.approx([1.0, 1.0])
    assert result["hardest_negative"].tolist() == pytest.approx([0.6, 0.8])
    assert result["margin"].tolist() == pytest.approx([0.4, 0.2])


def test_recovery_metrics_reports_absolute_and_relative_gain():
    result = recovery_metrics(np.array([0.60, 0.80]), np.array([0.90, 0.85]))

    assert result["delta_cosine"] == pytest.approx([0.30, 0.05])
    assert result["recovery_ratio"] == pytest.approx([0.75, 0.25])


def test_bootstrap_mean_ci_resamples_source_level_blocks_deterministically():
    blocks = np.array([-1.0, 0.0, 1.0, 2.0])

    result = bootstrap_mean_ci(blocks, samples=2000, seed=13)

    assert result["mean"] == pytest.approx(0.5)
    assert result["count"] == 4
    assert result["ci_low"] < result["mean"] < result["ci_high"]
    assert result == bootstrap_mean_ci(blocks, samples=2000, seed=13)


def test_paired_bootstrap_ci_compares_source_blocks_not_individual_views():
    ours = np.array([[0.2, 0.4], [0.8, 1.0]], dtype=float)
    baseline = np.array([[0.1, 0.3], [0.7, 0.9]], dtype=float)

    result = paired_bootstrap_ci(ours, baseline, samples=2000, seed=13)

    assert result["mean"] == pytest.approx(0.1)
    assert result["count"] == 2
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)
    assert result["significant"] is True
