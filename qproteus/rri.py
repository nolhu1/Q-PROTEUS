"""Resistance Resilience Index implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol

import numpy as np
from Bio.Align import substitution_matrices

from .constants import AMINO_ACIDS
from .features import calculate_features, hydrophobic_moment
from .sequence import Mutation, all_single_point_mutants, require_standard_sequence


class MICPredictor(Protocol):
    def predict(self, sequences: list[str]) -> list[float] | np.ndarray:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, sequences: list[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RRIComponents:
    sequence: str
    functional: float
    structural: float
    physicochemical: float
    rri: float
    n_mutants: int


def blosum62_probabilities(
    residue: str,
    *,
    beta: float = 0.5,
    cysteine_weight: float = 0.25,
) -> dict[str, float]:
    """Return BLOSUM62-softmax substitution probabilities for one residue."""
    residue = require_standard_sequence(residue)
    if len(residue) != 1:
        raise ValueError("Expected a single amino acid residue")
    matrix = substitution_matrices.load("BLOSUM62")
    weights: dict[str, float] = {}
    for replacement in AMINO_ACIDS:
        if replacement == residue:
            continue
        score = float(matrix[residue, replacement])
        weight = math.exp(beta * score)
        if residue == "C" or replacement == "C":
            # Cysteine changes are possible but down-weighted because disulfide
            # constraints can make them less interchangeable than the matrix implies.
            weight *= cysteine_weight
        weights[replacement] = weight
    total = sum(weights.values())
    return {replacement: weight / total for replacement, weight in weights.items()}


def weighted_mutation_table(
    sequence: str,
    *,
    beta: float = 0.5,
    cysteine_weight: float = 0.25,
) -> list[tuple[Mutation, float]]:
    """Enumerate all single substitutions with their model probabilities."""
    seq = require_standard_sequence(sequence)
    rows: list[tuple[Mutation, float]] = []
    for idx, residue in enumerate(seq):
        probabilities = blosum62_probabilities(
            residue,
            beta=beta,
            cysteine_weight=cysteine_weight,
        )
        for replacement, probability in probabilities.items():
            rows.append(
                (
                    Mutation(
                        position=idx + 1,
                        original=residue,
                        replacement=replacement,
                        sequence=seq[:idx] + replacement + seq[idx + 1 :],
                    ),
                    probability / len(seq),
                )
            )
    return rows


def functional_robustness(
    sequence: str,
    mutants: Iterable[Mutation],
    mic_predictor: MICPredictor,
    *,
    weights: Iterable[float] | None = None,
    clip_max: float = 1.5,
) -> float:
    """Mean retention using linear MIC ratios from log10 MIC predictions."""
    seq = require_standard_sequence(sequence)
    mutant_list = list(mutants)
    predictions = np.asarray(mic_predictor.predict([seq, *[m.sequence for m in mutant_list]]), dtype=float)
    # The MIC predictor returns log10 values; ratios need the linear MIC scale.
    wildtype_mic = 10.0 ** predictions[0]
    mutant_mics = 10.0 ** predictions[1:]
    ratios = wildtype_mic / np.maximum(mutant_mics, 1e-12)
    clipped = np.clip(ratios, 0.0, clip_max)
    return float(np.average(clipped, weights=_normalized_weights(weights, len(clipped))))


def structural_robustness(
    sequence: str,
    mutants: Iterable[Mutation],
    embedding_provider: EmbeddingProvider,
    *,
    weights: Iterable[float] | None = None,
    lambda_struct: float = 2.0,
) -> float:
    seq = require_standard_sequence(sequence)
    mutant_list = list(mutants)
    embeddings = np.asarray(embedding_provider.embed([seq, *[m.sequence for m in mutant_list]]), dtype=float)
    wildtype = embeddings[0]
    mutant_embeddings = embeddings[1:]
    distances = np.linalg.norm(mutant_embeddings - wildtype, axis=1)
    mean_distance = float(np.average(distances, weights=_normalized_weights(weights, len(distances))))
    return float(math.exp(-lambda_struct * mean_distance))


def physicochemical_robustness(
    sequence: str,
    mutants: Iterable[Mutation],
    *,
    weights: Iterable[float] | None = None,
    charge_weight: float = 0.5,
    hydrophobic_weight: float = 0.5,
    eta: float = 1.5,
) -> float:
    seq = require_standard_sequence(sequence)
    mutant_list = list(mutants)
    wild_features = calculate_features(seq)
    wild_charge = wild_features["net_charge"]
    wild_moment = hydrophobic_moment(seq)

    charge_deltas = []
    moment_deltas = []
    for mutation in mutant_list:
        features = calculate_features(mutation.sequence)
        charge_deltas.append((features["net_charge"] - wild_charge) ** 2)
        moment_deltas.append((hydrophobic_moment(mutation.sequence) - wild_moment) ** 2)

    normalized_weights = _normalized_weights(weights, len(charge_deltas))
    var_charge = float(np.average(charge_deltas, weights=normalized_weights))
    var_moment = float(np.average(moment_deltas, weights=normalized_weights))
    combined = charge_weight * var_charge + hydrophobic_weight * var_moment
    return float(math.exp(-eta * combined))


def calculate_rri(
    sequence: str,
    mic_predictor: MICPredictor,
    embedding_provider: EmbeddingProvider,
    *,
    max_mutants: int | None = None,
    seed: int = 42,
) -> RRIComponents:
    """Calculate multiplicative RRI for a peptide sequence."""
    seq = require_standard_sequence(sequence)
    weighted_rows = weighted_mutation_table(seq)
    mutants = [mutation for mutation, _weight in weighted_rows]
    weights = np.asarray([weight for _mutation, weight in weighted_rows], dtype=float)
    if max_mutants is not None and max_mutants < len(mutants):
        rng = np.random.default_rng(seed)
        probabilities = weights / weights.sum()
        # Optional subsampling preserves the same biologically weighted mutation model.
        indices = rng.choice(len(mutants), size=max_mutants, replace=False, p=probabilities)
        mutants = [mutants[int(index)] for index in indices]
        weights = weights[indices]
    weights = weights / weights.sum()
    functional = functional_robustness(seq, mutants, mic_predictor, weights=weights)
    structural = structural_robustness(seq, mutants, embedding_provider, weights=weights)
    physicochemical = physicochemical_robustness(seq, mutants, weights=weights)
    rri = functional * structural * physicochemical
    return RRIComponents(
        sequence=seq,
        functional=functional,
        structural=structural,
        physicochemical=physicochemical,
        rri=float(rri),
        n_mutants=len(mutants),
    )


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if math.isclose(maximum, minimum):
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def _normalized_weights(weights: Iterable[float] | None, expected_length: int) -> np.ndarray | None:
    if weights is None:
        return None
    array = np.asarray(list(weights), dtype=float)
    if len(array) != expected_length:
        raise ValueError("weights length must match mutants length")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return array / total
