"""Analysis metrics for optimization runs."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


def hamming_distance(sequence_a: str, sequence_b: str) -> int:
    if len(sequence_a) != len(sequence_b):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(a != b for a, b in zip(sequence_a, sequence_b))


def mean_pairwise_hamming(population: Iterable[str]) -> float:
    sequences = list(population)
    if len(sequences) < 2:
        return 0.0
    distances = [
        hamming_distance(a, b)
        for a, b in itertools.combinations(sequences, 2)
    ]
    return float(np.mean(distances))


def first_generation_to_fraction(
    best_fitness: Iterable[float],
    *,
    fraction: float = 0.9,
) -> int:
    values = np.asarray(list(best_fitness), dtype=float)
    if values.size == 0:
        raise ValueError("No fitness values supplied")
    # Use the observed run maximum so t90 is comparable across stochastic runs.
    target = values[0] + fraction * (values.max() - values[0])
    reached = np.flatnonzero(values >= target)
    return int(reached[0]) if reached.size else int(values.size - 1)


def pareto_front(points: np.ndarray, *, maximize: tuple[bool, ...]) -> np.ndarray:
    """Return a boolean mask for non-dominated points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    signs = np.asarray([1.0 if flag else -1.0 for flag in maximize])
    transformed = points * signs
    n_points = transformed.shape[0]
    is_front = np.ones(n_points, dtype=bool)
    for idx in range(n_points):
        if not is_front[idx]:
            continue
        dominates_idx = np.all(transformed >= transformed[idx], axis=1) & np.any(
            transformed > transformed[idx],
            axis=1,
        )
        if np.any(dominates_idx):
            is_front[idx] = False
    return is_front


def hypervolume_monte_carlo(
    points: np.ndarray,
    *,
    reference: tuple[float, float, float] = (0.0, 0.0, 0.0),
    upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
    samples: int = 100_000,
    seed: int = 42,
) -> float:
    """Estimate dominated hypervolume for all-maximized 3D objectives."""
    points = np.asarray(points, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.uniform(reference_array, upper_array, size=(samples, 3))
    dominated = np.zeros(samples, dtype=bool)
    chunk_size = 10_000
    # Chunking avoids building a samples-by-points comparison matrix all at once.
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        chunk = draws[start:end]
        comparisons = points[:, None, :] >= chunk[None, :, :]
        dominated[start:end] = np.any(np.all(comparisons, axis=2), axis=0)
    return float(np.prod(upper_array - reference_array) * dominated.mean())
