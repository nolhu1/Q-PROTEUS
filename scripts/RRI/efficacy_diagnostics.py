import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

# ============ CONFIG ============
DATA_PATH = "data/amp_vs_non_amp_training_dataset.csv"
MODEL_PATH = "models/efficacy_predictor.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "efficacy_diagnostics.txt")

# Known sequences for testing
SEQS = {
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Colistin": "CLCRRLRCG"
}

FEATURE_COLS = [
    'Length','Net_Charge','Hydrophobicity_GRAVY','Instability_Index',
    'Fraction_Positive_AA','Fraction_Negative_AA','Fraction_Hydrophobic_AA'
]

# ============ UTILITIES ============

def compute_features(sequence: str) -> dict:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysed_seq = ProteinAnalysis(sequence)
    length = len(sequence)
    net_charge = analysed_seq.charge_at_pH(7.4)
    gravy = analysed_seq.gravy()
    instability = analysed_seq.instability_index()
    pos, neg, hydro = "KRH", "DE", "AILMFWV"
    frac_pos = sum(sequence.count(a) for a in pos)/length if length>0 else 0
    frac_neg = sum(sequence.count(a) for a in neg)/length if length>0 else 0
    frac_hydro = sum(sequence.count(a) for a in hydro)/length if length>0 else 0
    return {
        "Length": length, "Net_Charge": net_charge, "Hydrophobicity_GRAVY": gravy,
        "Instability_Index": instability, "Fraction_Positive_AA": frac_pos,
        "Fraction_Negative_AA": frac_neg, "Fraction_Hydrophobic_AA": frac_hydro
    }

def predict_efficacy(seq, model, scaler):
    feats = compute_features(seq)
    X = np.array([[feats[c] for c in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0,1]

def mutate(seq, n_mut=1):
    import random
    aas = "ACDEFGHIKLMNPQRSTVWY"
    seq = list(seq)
    for _ in range(n_mut):
        i = random.randint(0, len(seq)-1)
        seq[i] = random.choice(aas)
    return "".join(seq)

def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# ============ MAIN ============

if __name__ == "__main__":
    # reset log
    open(LOG_PATH, "w").close()
    log("=== EFFICACY MODEL DIAGNOSTICS ===")

    # Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["Label"].astype(int)
    X_scaled = scaler.transform(X)

    # ---- Base performance ----
    probs = model.predict_proba(X_scaled)[:,1]
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, y_pred)

    log(f"Accuracy: {acc:.3f}")
    log(f"ROC AUC: {auc:.3f}")
    log(f"Confusion Matrix:\n{cm}")
    log(classification_report(y, y_pred))

    # ---- Probability distribution ----
    plt.hist(probs, bins=40, color="skyblue", edgecolor="black")
    plt.title("Predicted AMP Probabilities (training data)")
    plt.xlabel("Predicted P(AMP)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(LOG_DIR, "probability_distribution.png"), dpi=200)
    plt.close()

    # ---- Feature importance ----
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    log("\nFeature importances:")
    log(importances.to_string())
    plt.barh(importances.index, importances.values)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "feature_importances.png"), dpi=200)
    plt.close()

    # ---- Calibration curve ----
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0,1],[0,1],"--",label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Positive Fraction")
    plt.legend()
    plt.title("Calibration Curve")
    plt.savefig(os.path.join(LOG_DIR, "calibration_curve.png"), dpi=200)
    plt.close()

    # ---- Mutant sensitivity ----
    for name, seq in SEQS.items():
        wt_prob = predict_efficacy(seq, model, scaler)
        mutants = [mutate(seq, n_mut=1) for _ in range(200)]
        mut_probs = [predict_efficacy(m, model, scaler) for m in mutants]
        log(f"\n{name}: WT={wt_prob:.3f}, mutants_mean={np.mean(mut_probs):.3f}, std={np.std(mut_probs):.3f}")
        plt.hist(mut_probs, bins=30, color="lightcoral", edgecolor="black")
        plt.title(f"{name} Mutant Predictions (N=200)")
        plt.xlabel("Predicted P(AMP)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(LOG_DIR, f"{name}_mutant_hist.png"), dpi=200)
        plt.close()

    log("\n✅ Diagnostics complete. See PNGs in logs/ and paste this log below into ChatGPT.")
    print("\n=== CROSS-SPECTRUM VALIDATION ===")

# Reference peptides to test
test_peptides = {
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

# Compute features for each peptide using the same function as the model
from rri_calc import compute_features

for name, seq in test_peptides.items():
    features = compute_features(seq)
    X_test = [list(features.values())]
    prob = model.predict_proba(X_test)[0][1]  # AMP probability
    print(f"{name:<20} | AMP probability: {prob:.3f}")

