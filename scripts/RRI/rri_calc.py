import numpy as np
import joblib
import random
from tqdm import tqdm
from typing import Dict, Any
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from mutants import sample_mutants
import pandas as pd


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

def predict_efficacy(sequence: str, model, scaler) -> float:
    """Compute scaled features and predict AMP probability using the trained model."""
    feats = compute_features(sequence)

    # Create a DataFrame with matching column names
    feature_df = pd.DataFrame([feats])[scaler.feature_names_in_]

    # Scale the features (no warning now)
    X_scaled = scaler.transform(feature_df)

    # Predict probability of being an AMP
    prob = model.predict_proba(X_scaled)[0, 1]

    return float(prob)


def calculate_rri(
    sequence: str,
    model,
    scaler,
    N: int = 100,
    T: float = 0.5,
    delta: float = 0.3,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute the Resistance Resilience Index (RRI) for a peptide sequence.
    
    RRI is the fraction of mutants that retain antimicrobial activity.
    
    Args:
        sequence: original peptide sequence
        model: trained RandomForestClassifier
        scaler: fitted StandardScaler
        N: number of mutants to sample
        T: efficacy threshold (absolute)
        delta: relative threshold (max allowed drop in efficacy)
        seed: random seed
    Returns:
        dict with RRI_binary, RRI_cont, s_wt, and metadata
    """
    random.seed(seed)
    
    s_wt = predict_efficacy(sequence, model, scaler)
    mutants = sample_mutants(sequence, N=N, seed=seed)
    retained, drops = [], []
    
    if verbose:
        print(f"Calculating RRI for sequence ({len(sequence)} aa)... WT efficacy = {s_wt:.3f}")

    for _, _, _, mut_seq in tqdm(mutants, disable=not verbose):
        s_mut = predict_efficacy(mut_seq, model, scaler)
        retained.append(s_mut >= max(T, s_wt - delta))
        drops.append(max(0, s_wt - s_mut))

    rri_binary = np.mean(retained)
    rri_cont = 1.0 - np.mean(drops)

    return {
        "sequence": sequence,
        "s_wt": s_wt,
        "RRI_binary": rri_binary,
        "RRI_cont": rri_cont,
        "N": N,
        "T": T,
        "delta": delta,
        "seed": seed,
    }


def bootstrap_rri(
    sequence: str,
    model,
    scaler,
    N: int = 100,
    reps: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Compute RRI with bootstrapped confidence intervals."""
    np.random.seed(seed)
    rris = []
    for rep_seed in np.random.randint(0, 1_000_000, reps):
        res = calculate_rri(sequence, model, scaler, N=N, seed=int(rep_seed), verbose=False)
        rris.append(res["RRI_binary"])
    
    mean_rri = np.mean(rris)
    ci_low, ci_high = np.percentile(rris, [2.5, 97.5])
    
    return {
        "sequence": sequence,
        "RRI_mean": mean_rri,
        "RRI_CI_low": ci_low,
        "RRI_CI_high": ci_high,
        "N": N,
        "reps": reps
    }


if __name__ == "__main__":
    # Example usage for quick validation
    sequences = {
    # --- Known AMPs ---
    "Colistin": "CLCRRWIRVCR",
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Pexiganan": "GIGKFLKKAKKFGKAFVKILKK",
    "Temporin A": "FLPLIGRVLSGIL",
    
    # --- Neutral / acidic peptides (non-AMP-like) ---
    "Poly-Gly": "GGGGGGGGGGGG",
    "Poly-Ala": "AAAAAAAAAAAA",
    "Acidic_1": "DEDEDEDEDEDE",
    "Neutral_Random": "GAVLPSTNQYIRKHF",
    "Hydrophobic_Random": "FWILVLAIIVMLLV",
    }
