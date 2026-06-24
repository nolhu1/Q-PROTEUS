"""Quantum-inspired evolutionary algorithm for peptide search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import AMINO_ACIDS
from .metrics import mean_pairwise_hamming


@dataclass(frozen=True)
class QIEAConfig:
    length: int = 20
    population_size: int = 200
    generations: int = 500
    top_fraction: float = 0.2
    theta: float = 0.1
    entropy_threshold: float = 2.0
    low_entropy_patience: int = 10
    perturbation_fraction: float = 0.1
    seed: int = 42


@dataclass
class QIEARunResult:
    probability_matrix: np.ndarray
    population: list[str]
    fitnesses: list[float]
    history: list[dict[str, float]]


def initialize_probability_matrix(length: int) -> np.ndarray:
    return np.full((length, len(AMINO_ACIDS)), 1.0 / len(AMINO_ACIDS), dtype=float)


def sample_population(probability_matrix: np.ndarray, population_size: int, rng: np.random.Generator) -> list[str]:
    population: list[str] = []
    alphabet = np.asarray(AMINO_ACIDS)
    for _ in range(population_size):
        residues = [
            str(rng.choice(alphabet, p=probability_matrix[position]))
            for position in range(probability_matrix.shape[0])
        ]
        population.append("".join(residues))
    return population


def entropy_bits_per_position(probability_matrix: np.ndarray) -> float:
    safe = np.clip(probability_matrix, 1e-12, 1.0)
    entropy = -np.sum(safe * np.log2(safe), axis=1)
    return float(np.mean(entropy))


def update_probability_matrix(
    probability_matrix: np.ndarray,
    population: list[str],
    fitnesses: list[float],
    *,
    top_fraction: float,
    theta: float,
) -> np.ndarray:
    top_count = max(1, int(round(len(population) * top_fraction)))
    top_indices = np.argsort(np.asarray(fitnesses))[-top_count:]
    frequencies = np.zeros_like(probability_matrix)
    for index in top_indices:
        sequence = population[int(index)]
        for position, residue in enumerate(sequence):
            frequencies[position, AMINO_ACIDS.index(residue)] += 1.0
    frequencies /= float(top_count)
    # Rotation-style update: move residue probabilities partway toward the
    # empirical distribution of the best sampled sequences.
    updated = probability_matrix + theta * (frequencies - probability_matrix)
    updated /= updated.sum(axis=1, keepdims=True)
    return updated


def perturb_probability_matrix(probability_matrix: np.ndarray, fraction: float) -> np.ndarray:
    uniform = np.full_like(probability_matrix, 1.0 / len(AMINO_ACIDS))
    updated = (1.0 - fraction) * probability_matrix + fraction * uniform
    updated /= updated.sum(axis=1, keepdims=True)
    return updated


def run_qiea(config: QIEAConfig, fitness_fn: Callable[[str], float]) -> QIEARunResult:
    rng = np.random.default_rng(config.seed)
    probability_matrix = initialize_probability_matrix(config.length)
    theta = config.theta
    low_entropy_generations = 0
    history: list[dict[str, float]] = []
    population: list[str] = []
    fitnesses: list[float] = []

    for generation in range(config.generations + 1):
        population = sample_population(probability_matrix, config.population_size, rng)
        fitnesses = [float(fitness_fn(sequence)) for sequence in population]
        entropy = entropy_bits_per_position(probability_matrix)
        history.append(
            {
                "generation": float(generation),
                "best_fitness": float(max(fitnesses)),
                "mean_fitness": float(np.mean(fitnesses)),
                "entropy_bits_per_position": float(entropy),
                "mean_pairwise_hamming": float(mean_pairwise_hamming(population)),
                "unique_sequences": float(len(set(population))),
                "theta": float(theta),
            }
        )
        if generation == config.generations:
            break

        if entropy < config.entropy_threshold:
            # Low entropy means positions are fixing too quickly, so slow the
            # update instead of immediately resetting what has been learned.
            theta *= 0.5
            low_entropy_generations += 1
        else:
            low_entropy_generations = 0

        probability_matrix = update_probability_matrix(
            probability_matrix,
            population,
            fitnesses,
            top_fraction=config.top_fraction,
            theta=theta,
        )
        if low_entropy_generations >= config.low_entropy_patience:
            # Add a small uniform component only after sustained collapse.
            probability_matrix = perturb_probability_matrix(
                probability_matrix,
                config.perturbation_fraction,
            )
            low_entropy_generations = 0

    return QIEARunResult(
        probability_matrix=probability_matrix,
        population=population,
        fitnesses=fitnesses,
        history=history,
    )
