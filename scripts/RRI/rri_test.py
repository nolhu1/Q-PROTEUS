import joblib
from rri_calc import bootstrap_rri

# Load model + scaler
model = joblib.load("models/efficacy_predictor.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

# Peptides to test
peptides = {
    "LL-37 variant": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Protegrin-1": "RGGRLCYCRRRFCVCVGR",
    "Temporin-1": "FLPLIGRVLSGIL",
    "Dermaseptin S4": "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV",
    "Cathelicidin mini": "KRLKKLLKKLKK",
    "Synthetic neutral": "AIGILVAGLVALGA",
    "Weak AMP variant": "GAFVLSGAKILK"
}

# Run RRI bootstrap for each peptide
for name, seq in peptides.items():
    res = bootstrap_rri(seq, model, scaler, N=100, reps=50)
    print(f"{name}: RRI_mean={res['RRI_mean']:.3f}, "
          f"CI=({res['RRI_CI_low']:.3f}, {res['RRI_CI_high']:.3f})")
