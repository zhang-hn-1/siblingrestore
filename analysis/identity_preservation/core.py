"""Pure numerical primitives for source-level identity-preservation analysis."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def identity_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    own_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Return own-source cosine, hardest wrong-source cosine, and margin.

    Embeddings are expected to be L2-normalized by the official verifier. The
    function deliberately does not renormalize them, so a protocol mistake is
    not hidden by the analysis layer.
    """
    query = _as_float_matrix(query_embeddings, "query_embeddings")
    gallery = _as_float_matrix(gallery_embeddings, "gallery_embeddings")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    if query.shape[0] > gallery.shape[0]:
        raise ValueError("more queries than gallery entries")
    if own_indices is None:
        own = np.arange(query.shape[0], dtype=np.int64)
    else:
        own = np.asarray(list(own_indices), dtype=np.int64)
    if own.shape != (query.shape[0],):
        raise ValueError("own_indices must contain one index per query")
    if ((own < 0) | (own >= gallery.shape[0])).any():
        raise ValueError("own_indices contains an invalid gallery index")

    scores = query @ gallery.T
    cosine = scores[np.arange(query.shape[0]), own]
    wrong_scores = scores.copy()
    wrong_scores[np.arange(query.shape[0]), own] = -np.inf
    hardest_negative = wrong_scores.max(axis=1)
    return {
        "cosine": cosine,
        "hardest_negative": hardest_negative,
        "margin": cosine - hardest_negative,
        "scores": scores,
    }


def recovery_metrics(
    degraded_cosine: np.ndarray, restored_cosine: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute absolute and ideal-cosine-normalized identity recovery."""
    degraded = np.asarray(degraded_cosine, dtype=np.float64)
    restored = np.asarray(restored_cosine, dtype=np.float64)
    if degraded.shape != restored.shape:
        raise ValueError("degraded and restored cosine arrays differ in shape")
    if not np.isfinite(degraded).all() or not np.isfinite(restored).all():
        raise ValueError("cosine arrays contain non-finite values")
    delta = restored - degraded
    denominator = 1.0 - degraded
    ratio = np.divide(
        delta,
        denominator,
        out=np.full_like(delta, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return {"delta_cosine": delta, "recovery_ratio": ratio}


def bootstrap_mean_ci(
    values: np.ndarray, samples: int = 2000, seed: int = 13
) -> dict[str, float | int]:
    """Bootstrap the mean of source-level blocks with a deterministic seed."""
    blocks = np.asarray(values, dtype=np.float64)
    if blocks.ndim != 1 or blocks.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.isfinite(blocks).all():
        raise ValueError("values contains non-finite values")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, blocks.size, size=(samples, blocks.size))
    means = blocks[indices].mean(axis=1)
    return {
        "mean": float(blocks.mean()),
        "std": float(blocks.std(ddof=1)) if blocks.size > 1 else 0.0,
        "count": int(blocks.size),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def paired_bootstrap_ci(
    ours: np.ndarray,
    baseline: np.ndarray,
    samples: int = 2000,
    seed: int = 13,
) -> dict[str, float | int | bool]:
    """Bootstrap a paired mean difference over source-level blocks.

    Inputs may be ``[sources]`` or ``[sources, views]``. In the latter case,
    the view mean is formed inside each source before resampling sources.
    """
    ours_array = np.asarray(ours, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if ours_array.shape != baseline_array.shape or ours_array.ndim not in (1, 2):
        raise ValueError("paired inputs must have identical 1-D or 2-D shapes")
    if not np.isfinite(ours_array).all() or not np.isfinite(baseline_array).all():
        raise ValueError("paired inputs contain non-finite values")
    if samples <= 0:
        raise ValueError("samples must be positive")
    source_differences = (ours_array - baseline_array)
    if source_differences.ndim == 2:
        source_differences = source_differences.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, source_differences.size,
                           size=(samples, source_differences.size))
    means = source_differences[indices].mean(axis=1)
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return {
        "mean": float(source_differences.mean()),
        "count": int(source_differences.size),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(ci_low > 0.0),
    }
