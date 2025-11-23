# scripts/diagnostics/calibrate_efficacy_fixed.py
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# Paths
DATA_CSV = "data/amp_vs_non_amp_training_dataset.csv"
MODEL_PATH = "models/efficacy_predictor.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
OUT_DIR = "logs"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "Length","Net_Charge","Hydrophobicity_GRAVY","Instability_Index",
    "Fraction_Positive_AA","Fraction_Negative_AA","Fraction_Hydrophobic_AA"
]

def load_everything():
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLS].copy()
    y = df["Label"].astype(int).copy()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(SCALER_PATH)
    return X, y, model, scaler

def eval_and_plot(name, estimator, X_test, y_test, scaler):
    Xs = scaler.transform(X_test)
    probs = estimator.predict_proba(Xs)[:,1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    # histogram
    plt.figure(figsize=(6,3))
    plt.hist(probs, bins=40, edgecolor='k')
    plt.title(f"{name} predicted prob histogram\nROC-AUC={roc:.3f}  Brier={brier:.4f}")
    plt.xlabel("Predicted P(AMP)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}_prob_hist.png", dpi=200)
    plt.close()

    # calibration curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(4,4))
    plt.plot(prob_pred, prob_true, "o-", label=name)
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}_calibration_curve.png", dpi=200)
    plt.close()

    return {"roc_auc":roc, "brier":brier, "accuracy":acc, "probs":probs}

def main():
    X, y, model, scaler = load_everything()

    # split into test + rest
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # from rest, take a calibration set
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=42, stratify=y_rest
    )

    # evaluate base model
    base_metrics = eval_and_plot("original", model, X_test, y_test, scaler)
    print("ORIGINAL on test:", base_metrics)

    # fit calibrators
    calibrators = {}
    for method in ["sigmoid", "isotonic"]:
        print(f"\nFitting calibrator: {method}")
        calib = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        Xc = scaler.transform(X_calib)
        calib.fit(Xc, y_calib)
        calibrators[method] = calib
        metrics = eval_and_plot(f"calibrated_{method}", calib, X_test, y_test, scaler)
        print(f"Calibrated ({method}) on test: ROC-AUC={metrics['roc_auc']:.4f}  Brier={metrics['brier']:.4f}  Acc={metrics['accuracy']:.4f}")

    # select best by Brier score
    best_method = min(
        calibrators.keys(),
        key=lambda m: eval_and_plot(f"tmp_{m}", calibrators[m], X_test, y_test, scaler)["brier"]
    )
    best_calib = calibrators[best_method]
    best_metrics = eval_and_plot(f"best_calibrated_{best_method}", best_calib, X_test, y_test, scaler)
    print("\nBest calibrator:", best_method, best_metrics)

    # save calibrated model
    joblib.dump(best_calib, "models/efficacy_predictor_calibrated.pkl")
    print(f"Saved best calibrated model to models/efficacy_predictor_calibrated.pkl")

    # CSV comparing probs
    Xs_test = scaler.transform(X_test)
    df_out = pd.DataFrame({
        "y_true": y_test.values,
        "orig_prob": model.predict_proba(Xs_test)[:,1],
        f"calib_prob_{best_method}": best_calib.predict_proba(Xs_test)[:,1]
    })
    df_out.to_csv(os.path.join(OUT_DIR, "prob_compare_test.csv"), index=False)
    print(f"Wrote probability comparison to {OUT_DIR}/prob_compare_test.csv")

if __name__ == "__main__":
    main()
