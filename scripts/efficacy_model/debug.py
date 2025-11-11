import pickle
import joblib
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Load model and scaler ---
with open("models/efficacy_predictor.pkl", "rb") as f:
    model = pickle.load(f)
scaler = joblib.load("models/feature_scaler.pkl")

# --- Load training data to compute stats ---
training_data = pd.read_csv("data/amp_vs_non_amp_training_dataset.csv")
feature_columns = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 
                   'Instability_Index', 'Fraction_Positive_AA', 
                   'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA']
X_train_stats = training_data[feature_columns].describe().T  # mean, std, min, max

# --- Feature extraction function ---
def compute_features(sequence: str) -> pd.DataFrame:
    analysed_seq = ProteinAnalysis(sequence)
    length = len(sequence)
    net_charge = analysed_seq.charge_at_pH(7.4)
    gravy = analysed_seq.gravy()
    instability = analysed_seq.instability_index()
    
    pos_residues = "KRH"
    neg_residues = "DE"
    hydrophobic_residues = "AILMFWV"
    
    frac_pos = sum(sequence.count(aa) for aa in pos_residues) / length if length > 0 else 0
    frac_neg = sum(sequence.count(aa) for aa in neg_residues) / length if length > 0 else 0
    frac_hydro = sum(sequence.count(aa) for aa in hydrophobic_residues) / length if length > 0 else 0

    return pd.DataFrame([{
        "Length": length,
        "Net_Charge": net_charge,
        "Hydrophobicity_GRAVY": gravy,
        "Instability_Index": instability,
        "Fraction_Positive_AA": frac_pos,
        "Fraction_Negative_AA": frac_neg,
        "Fraction_Hydrophobic_AA": frac_hydro
    }])

# --- Function to debug predictions with outlier check ---
def debug_prediction(name, sequence):
    features = compute_features(sequence)
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0, 1]
    prediction = model.predict(scaled)[0]

    print(f"\n=== DEBUG: {name} ===")
    print(f"Sequence: {sequence}")
    
    # Print raw features and z-scores
    print("Computed features (raw, z-score vs training data):")
    for col in features.columns:
        val = features.iloc[0][col]
        mean = X_train_stats.loc[col, 'mean']
        std = X_train_stats.loc[col, 'std']
        z = (val - mean) / std if std > 0 else 0
        print(f"  {col}: {val:.3f} (mean={mean:.3f}, std={std:.3f}, z={z:.2f})")
    
    # Print scaled features
    print(f"Scaled features: {scaled[0]}")
    print(f"Predicted class: {prediction}")
    print(f"Predicted AMP probability: {prob:.3f}")

# --- Example peptides ---
peptides = {
    "Colistin": "KWKKKKKKKKK",
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Pexiganan": "GIGKFLKKAKKFGKAFVKILKK",
    "Temporin A": "FLPLIGRVLSGIL"
}

for name, seq in peptides.items():
    debug_prediction(name, seq)
