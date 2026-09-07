"""Pure numerical helpers for source-level paired significance analysis."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


HIGHER_IS_BETTER = {"PSNR", "SSIM", "Top1", "AUC", "Cosine", "Margin"}
LOWER_IS_BETTER = {"LPIPS", "EER"}


def improvement(ours: float, baseline: float, metric: str) -> float:
    """Return a signed difference where positive always means Ours is better."""
    if metric in HIGHER_IS_BETTER:
        return float(ours - baseline)
    if metric in LOWER_IS_BETTER:
        return float(baseline - ours)
    raise ValueError("unknown metric: %s" % metric)


def source_level_mean(values_by_source: Mapping[str, Sequence[float]], source_order: Sequence[str]) -> np.ndarray:
    """Average all available degradation views within each source."""
    result = []
    for source_id in source_order:
        values = np.asarray(values_by_source[source_id], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("source values must be finite non-empty 1-D arrays")
        result.append(float(values.mean()))
    return np.asarray(result, dtype=np.float64)


def bootstrap_mean_distribution(
    ours: np.ndarray, baseline: np.ndarray, sample_indices: np.ndarray, metric: str
) -> np.ndarray:
    """Compute paired source bootstrap means using shared source indices."""
    ours_array = np.asarray(ours, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if ours_array.shape != baseline_array.shape or ours_array.ndim != 1:
        raise ValueError("ours and baseline must be equal-shaped 1-D arrays")
    if indices.ndim != 2 or indices.shape[1] != ours_array.size:
        raise ValueError("sample_indices must be [repetitions, source_count]")
    if ((indices < 0) | (indices >= ours_array.size)).any():
        raise ValueError("sample_indices contains an invalid source index")
    deltas = np.asarray([improvement(o, b, metric) for o, b in zip(ours_array, baseline_array)])
    return deltas[indices].mean(axis=1)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return float("nan")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _eer(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return float("nan")
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    return float(np.min((fp[ends] / negatives + (positives - tp[ends]) / positives) / 2.0))


def gallery_metric(score_blocks: np.ndarray, source_indices: Sequence[int], metric: str) -> float:
    """Recompute gallery-level AUC/EER from sampled source score blocks."""
    blocks = np.asarray(score_blocks, dtype=np.float64)
    sampled = np.asarray(source_indices, dtype=np.int64)
    if blocks.ndim != 3 or blocks.shape[0] == 0:
        raise ValueError("score_blocks must be [sources, views, gallery]")
    if sampled.ndim != 1 or sampled.size == 0:
        raise ValueError("source_indices must be non-empty 1-D")
    if ((sampled < 0) | (sampled >= blocks.shape[0])).any():
        raise ValueError("source_indices contains an invalid source index")
    scores = blocks[sampled].reshape(-1, blocks.shape[-1])
    labels = np.zeros_like(scores, dtype=np.int8)
    query_sources = np.repeat(sampled, blocks.shape[1])
    for row, source_id in enumerate(query_sources):
        labels[row, source_id] = 1
    flat_scores = scores.reshape(-1)
    flat_labels = labels.reshape(-1)
    if metric == "AUC":
        return _binary_auc(flat_scores, flat_labels)
    if metric == "EER":
        return _eer(flat_scores, flat_labels)
    raise ValueError("gallery_metric only supports AUC and EER")


def _source_weights(sample_indices: np.ndarray, source_count: int) -> np.ndarray:
    weights = np.zeros((sample_indices.shape[0], source_count), dtype=np.float64)
    rows = np.repeat(np.arange(sample_indices.shape[0]), sample_indices.shape[1])
    np.add.at(weights, (rows, sample_indices.reshape(-1)), 1.0)
    return weights


def _auc_source_pair_matrix(score_blocks: np.ndarray) -> np.ndarray:
    source_count, view_count, gallery_count = score_blocks.shape
    flat_scores = score_blocks.reshape(-1)
    query_sources = np.repeat(np.arange(source_count), view_count * gallery_count)
    gallery_indices = np.tile(np.arange(gallery_count), source_count * view_count)
    positive = gallery_indices == query_sources
    order = np.argsort(flat_scores, kind="stable")
    negatives_seen = np.zeros(source_count, dtype=np.float64)
    pairs = np.zeros((source_count, source_count), dtype=np.float64)
    for flat_index in order:
        source = query_sources[flat_index]
        if positive[flat_index]:
            pairs[source] += negatives_seen
        else:
            negatives_seen[source] += 1.0
    return pairs


def _eer_bootstrap_from_weights(score_blocks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    source_count, view_count, gallery_count = score_blocks.shape
    flat_scores = score_blocks.reshape(-1)
    query_sources = np.repeat(np.arange(source_count), view_count * gallery_count)
    gallery_indices = np.tile(np.arange(gallery_count), source_count * view_count)
    positive = gallery_indices == query_sources
    order = np.argsort(-flat_scores, kind="stable")
    sorted_scores = flat_scores[order]
    sorted_sources = query_sources[order]
    sorted_positive = positive[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    group_starts = np.r_[0, group_ends[:-1] + 1]
    positive_per_source = float(view_count)
    negative_per_source = float(view_count * (gallery_count - 1))
    batch_size = 1000
    best = np.full(weights.shape[0], np.inf, dtype=np.float64)
    cumulative_positive = np.zeros(source_count, dtype=np.float64)
    cumulative_negative = np.zeros(source_count, dtype=np.float64)
    for block_start in range(0, len(group_ends), 4096):
        block_end = min(block_start + 4096, len(group_ends))
        coefficients = np.empty((block_end - block_start, source_count), dtype=np.float64)
        for local, group_index in enumerate(range(block_start, block_end)):
            start = int(group_starts[group_index])
            end = int(group_ends[group_index]) + 1
            sources = sorted_sources[start:end]
            labels = sorted_positive[start:end]
            for source in sources[labels]:
                cumulative_positive[source] += 1.0
            for source in sources[~labels]:
                cumulative_negative[source] += 1.0
            coefficients[local] = 0.5 * (
                cumulative_negative / (source_count * negative_per_source)
                - cumulative_positive / (source_count * positive_per_source)
            )
        for batch_start in range(0, weights.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, weights.shape[0])
            values = weights[batch_start:batch_end] @ coefficients.T
            best[batch_start:batch_end] = np.minimum(best[batch_start:batch_end], values.min(axis=1))
    return 0.5 + best


def gallery_bootstrap_distribution(score_blocks: np.ndarray, sample_indices: np.ndarray, metric: str) -> np.ndarray:
    """Recompute gallery AUC/EER for every source bootstrap replicate.

    The implementation uses source multiplicities rather than expanding
    duplicate source blocks. This is mathematically equivalent to concatenating
    all sampled seven-view blocks, while keeping the 10,000-replicate run
    tractable.
    """
    blocks = np.asarray(score_blocks, dtype=np.float64)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if blocks.ndim != 3 or indices.ndim != 2 or indices.shape[1] != blocks.shape[0]:
        raise ValueError("invalid score block or bootstrap index shape")
    weights = _source_weights(indices, blocks.shape[0])
    if metric == "AUC":
        pairs = _auc_source_pair_matrix(blocks)
        positive_total = blocks.shape[0] * blocks.shape[1]
        negative_total = positive_total * (blocks.shape[2] - 1)
        return np.einsum("bi,ij,bj->b", weights, pairs, weights) / (positive_total * negative_total)
    if metric == "EER":
        return _eer_bootstrap_from_weights(blocks, weights)
    raise ValueError("gallery_bootstrap_distribution only supports AUC and EER")


def centered_bootstrap_pvalue(point: float, distribution: np.ndarray) -> float:
    """Two-sided centered-bootstrap tail probability.

    The bootstrap distribution is centered under the null by subtracting the
    observed point estimate. The two tails compare deviations at least as
    extreme as the observed statistic, and are doubled conservatively.
    """
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution must be a finite non-empty 1-D array")
    centered = values - float(point)
    magnitude = abs(float(point))
    left = float(np.mean(centered <= -magnitude))
    right = float(np.mean(centered >= magnitude))
    return float(min(1.0, 2.0 * min(left, right)))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg FDR q-values in input order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be finite probabilities")
    n = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result
