from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
import numpy as np
import joblib
import pickle

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
import pandas as pd

def features_to_df(features: dict, scaler):
    return pd.DataFrame([features], columns=scaler.feature_names_in_)
# -------------------
# Feature Extraction
# -------------------
def compute_features(sequence: str) -> dict:
    """Compute features consistent with training pipeline."""
    analysed_seq = ProteinAnalysis(sequence)

    length = len(sequence)
    net_charge = analysed_seq.charge_at_pH(7.4)
    gravy = analysed_seq.gravy()
    instability = analysed_seq.instability_index()

    # Fraction counts
    pos_residues = "KRH"
    neg_residues = "DE"
    hydrophobic_residues = "AILMFWV"

    frac_pos = sum(sequence.count(aa) for aa in pos_residues) / length if length > 0 else 0
    frac_neg = sum(sequence.count(aa) for aa in neg_residues) / length if length > 0 else 0
    frac_hydro = sum(sequence.count(aa) for aa in hydrophobic_residues) / length if length > 0 else 0

    return {
        "Length": length,
        "Net_Charge": net_charge,
        "Hydrophobicity_GRAVY": gravy,
        "Instability_Index": instability,
        "Fraction_Positive_AA": frac_pos,
        "Fraction_Negative_AA": frac_neg,
        "Fraction_Hydrophobic_AA": frac_hydro
    }

# -------------------
# Mutant Generator
# -------------------
def generate_mutants(sequence: str, n_mutants: int = 100):
    """Generate N single-point random mutants of the sequence."""
    mutants = []
    for _ in range(n_mutants):
        pos = random.randrange(len(sequence))
        new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != sequence[pos]])
        mutant = sequence[:pos] + new_aa + sequence[pos+1:]
        mutants.append(mutant)
    return mutants

# -------------------
# RRI Calculation
# -------------------
def calculate_rri(sequence: str, model_path="models/efficacy_predictor.pkl",
                  scaler_path="models/feature_scaler.pkl", n_mutants: int = 100,
                  threshold: float = 0.7):
    """Calculate Resistance Resilience Index (RRI) for a peptide."""

    # Load model + scaler
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(scaler_path)

    # Parent prediction
    parent_features = compute_features(sequence)
    X_parent = scaler.transform(features_to_df(parent_features, scaler))
    parent_prob = model.predict_proba(X_parent)[0, 1]

    # Generate mutants + predict
    mutants = generate_mutants(sequence, n_mutants)
    mutant_probs = []
    for m in mutants:
        feats = compute_features(m)
        X = scaler.transform(features_to_df(feats, scaler))
        prob = model.predict_proba(X)[0, 1]
        mutant_probs.append(prob)

    # Binary RRI (fraction of mutants still predicted as active above threshold)
    binary_rri = sum(p >= threshold for p in mutant_probs) / n_mutants

    # Continuous RRI (relative preservation of efficacy)
    continuous_rri = np.mean([p / parent_prob if parent_prob > 0 else 0 for p in mutant_probs])

    return {
        "parent_prob": parent_prob,
        "binary_rri": binary_rri,
        "continuous_rri": continuous_rri,
        "mutant_probs": mutant_probs
    }

