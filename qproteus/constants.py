"""Shared constants for peptide processing."""

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_SET = frozenset(AMINO_ACIDS)

POSITIVE_RESIDUES = frozenset("KRH")
NEGATIVE_RESIDUES = frozenset("DE")
HYDROPHOBIC_RESIDUES = frozenset("AILMFWVP")
POLAR_RESIDUES = frozenset("STNQYC")

AMP_EXCLUSION_KEYWORDS = (
    "defensin",
    "cathelicidin",
    "antimicrobial",
    "anti-microbial",
    "bactericidal",
    "cationic",
)

DEFAULT_FEATURE_COLUMNS = (
    "length",
    "net_charge",
    "gravy",
    "fraction_positive",
    "fraction_negative",
    "fraction_hydrophobic",
    "instability_index",
    "molecular_weight",
    "helix_fraction",
    "sheet_fraction",
    "charge_density",
)

KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
