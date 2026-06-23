"""Physicochemical descriptor extraction for peptide sequences."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .constants import (
    DEFAULT_FEATURE_COLUMNS,
    HYDROPHOBIC_RESIDUES,
    KYTE_DOOLITTLE,
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
)
from .sequence import normalize_sequence, require_standard_sequence


def residue_fraction(sequence: str, residues: set[str] | frozenset[str]) -> float:
    seq = require_standard_sequence(sequence)
    return sum(1 for residue in seq if residue in residues) / len(seq)


def hydrophobic_moment(sequence: str, *, angle_degrees: float = 100.0) -> float:
    """Compute a simple alpha-helical hydrophobic moment."""
    seq = require_standard_sequence(sequence)
    angle = math.radians(angle_degrees)
    x_component = 0.0
    y_component = 0.0
    for idx, residue in enumerate(seq):
        # 100 degrees approximates the residue-to-residue turn of an alpha helix.
        hydropathy = KYTE_DOOLITTLE[residue]
        x_component += hydropathy * math.cos(idx * angle)
        y_component += hydropathy * math.sin(idx * angle)
    return math.sqrt(x_component * x_component + y_component * y_component) / len(seq)


def calculate_features(sequence: str) -> dict[str, float]:
    """Return the descriptor set used by the Q-PROTEUS methods."""
    seq = require_standard_sequence(sequence)
    analysis = ProteinAnalysis(seq)
    helix_fraction, _turn_fraction, sheet_fraction = analysis.secondary_structure_fraction()
    net_charge = float(analysis.charge_at_pH(7.4))
    length = len(seq)
    return {
        "length": float(length),
        "net_charge": net_charge,
        "gravy": float(analysis.gravy()),
        "fraction_positive": residue_fraction(seq, POSITIVE_RESIDUES),
        "fraction_negative": residue_fraction(seq, NEGATIVE_RESIDUES),
        "fraction_hydrophobic": residue_fraction(seq, HYDROPHOBIC_RESIDUES),
        "instability_index": float(analysis.instability_index()),
        "molecular_weight": float(analysis.molecular_weight()),
        "helix_fraction": float(helix_fraction),
        "sheet_fraction": float(sheet_fraction),
        "charge_density": net_charge / length,
    }


def feature_vector(sequence: str, columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS) -> np.ndarray:
    features = calculate_features(sequence)
    return np.asarray([features[column] for column in columns], dtype=float)


def feature_table(
    sequences: Iterable[str],
    *,
    columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    rows = []
    for sequence in sequences:
        seq = normalize_sequence(sequence)
        features = calculate_features(seq)
        rows.append({"sequence": seq, **features})
    selected_columns = ["sequence", *columns]
    return pd.DataFrame(rows)[selected_columns]


def flag_extrapolations(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    k: float = 2.0,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag rows outside mean +/- k sd for any descriptor."""
    columns = list(feature_columns)
    stats_source = frame if reference is None else reference
    means = stats_source[columns].mean(axis=0)
    stds = stats_source[columns].std(axis=0, ddof=0).replace(0, np.nan)
    lower = means - k * stds
    upper = means + k * stds
    # Keep flagged rows in the data, but make them easy to exclude from search.
    outlier_mask = ((frame[columns] < lower) | (frame[columns] > upper)).any(axis=1)

    result = frame.copy()
    result["is_extrapolation"] = outlier_mask.fillna(False)
    stats = pd.DataFrame({"mean": means, "std": stds, "lower": lower, "upper": upper})
    return result, stats
