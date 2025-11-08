import joblib
from rri_calc import bootstrap_rri

model = joblib.load("models/efficacy_predictor.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

# Test on known AMPs
colistin = "CLCRRLRCG"     # known resistance-prone
ll37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"  # durable AMP

for seq, name in [(colistin, "Colistin"), (ll37, "LL-37")]:
    res = bootstrap_rri(seq, model, scaler, N=100, reps=50)
    print(f"{name}: RRI_mean={res['RRI_mean']:.3f}, CI=({res['RRI_CI_low']:.3f}, {res['RRI_CI_high']:.3f})")
