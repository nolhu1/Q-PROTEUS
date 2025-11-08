import random
import numpy as np
import pickle
import joblib
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Load model + scaler
with open("models/efficacy_predictor.pkl", "rb") as f:
    model = pickle.load(f)
scaler = joblib.load("models/feature_scaler.pkl")

# Feature extraction (same as training)
def compute_features(seq):
    analysed_seq = ProteinAnalysis(seq)
    length = len(seq)
    net_charge = analysed_seq.charge_at_pH(7.4)
    gravy = analysed_seq.gravy()
    instability = analysed_seq.instability_index()
    pos_residues = "KRH"
    neg_residues = "DE"
    hydrophobic_residues = "AILMFWV"
    frac_pos = sum(seq.count(aa) for aa in pos_residues)/length
    frac_neg = sum(seq.count(aa) for aa in neg_residues)/length
    frac_hydro = sum(seq.count(aa) for aa in hydrophobic_residues)/length
    return [length, net_charge, gravy, instability, frac_pos, frac_neg, frac_hydro]

# Generate challenging negatives
negatives = [
    "A"*10, "G"*12, "L"*15,                 # Poly-AA
] + [
    "".join(random.choices(AMINO_ACIDS, k=12)) for _ in range(20)   # Random sequences
] + [
    "MAGAININ2"[::-1], "LL-37"[::-1]        # Scrambled known AMPs
]

# Evaluate
pred_probs = []
for seq in negatives:
    feats = compute_features(seq)
    X = scaler.transform([feats])
    prob = model.predict_proba(X)[0,1]
    pred_probs.append(prob)
    print(f"{seq}: predicted probability = {prob:.3f}")

# Summary
pred_probs = np.array(pred_probs)
print("\n--- Stress Test Summary ---")
print(f"Mean predicted probability for negatives: {pred_probs.mean():.3f}")
print(f"Fraction above 0.7 threshold: {(pred_probs>=0.7).mean():.3f}")
