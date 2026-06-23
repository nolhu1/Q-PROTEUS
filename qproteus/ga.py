"""Standard genetic algorithm baseline."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from .constants import AMINO_ACIDS


@dataclass(frozen=True)
class GAConfig:
    length: int = 20
    population_size: int = 200
    generations: int = 500
    tournament_size: int = 3
    crossover_probability: float = 0.8
    mutation_probability: float = 0.05
    seed: int = 42


@dataclass
class GARunResult:
    population: list[str]
    fitnesses: list[float]
    history: list[dict[str, float]]


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def tournament_select(
    population: list[str],
    fitnesses: list[float],
    *,
    tournament_size: int,
    rng: random.Random,
) -> str:
    indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    best_index = max(indices, key=lambda index: fitnesses[index])
    return population[best_index]


def crossover(parent_a: str, parent_b: str, rng: random.Random) -> tuple[str, str]:
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have equal length")
    if len(parent_a) < 2:
        return parent_a, parent_b
    point = rng.randrange(1, len(parent_a))
    return parent_a[:point] + parent_b[point:], parent_b[:point] + parent_a[point:]


def mutate(sequence: str, *, mutation_probability: float, rng: random.Random) -> str:
    residues = list(sequence)
    for idx, residue in enumerate(residues):
        if rng.random() < mutation_probability:
            residues[idx] = rng.choice([aa for aa in AMINO_ACIDS if aa != residue])
    return "".join(residues)


def run_ga(config: GAConfig, fitness_fn: Callable[[str], float]) -> GARunResult:
    rng = random.Random(config.seed)
    population = [random_sequence(config.length, rng) for _ in range(config.population_size)]
    history: list[dict[str, float]] = []

    for generation in range(config.generations + 1):
        # Record the current generation before breeding so history includes
        # generation 0 and the final population under the same convention.
        fitnesses = [float(fitness_fn(sequence)) for sequence in population]
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        history.append(
            {
                "generation": float(generation),
                "best_fitness": float(best),
                "mean_fitness": float(mean),
                "unique_sequences": float(len(set(population))),
            }
        )
        if generation == config.generations:
            return GARunResult(population=population, fitnesses=fitnesses, history=history)

        next_population: list[str] = []
        while len(next_population) < config.population_size:
            parent_a = tournament_select(
                population,
                fitnesses,
                tournament_size=config.tournament_size,
                rng=rng,
            )
            parent_b = tournament_select(
                population,
                fitnesses,
                tournament_size=config.tournament_size,
                rng=rng,
            )
            child_a, child_b = parent_a, parent_b
            if rng.random() < config.crossover_probability:
                child_a, child_b = crossover(parent_a, parent_b, rng)
            next_population.append(
                mutate(child_a, mutation_probability=config.mutation_probability, rng=rng)
            )
            if len(next_population) < config.population_size:
                next_population.append(
                    mutate(child_b, mutation_probability=config.mutation_probability, rng=rng)
                )
        population = next_population

    raise RuntimeError("GA loop exited unexpectedly")
