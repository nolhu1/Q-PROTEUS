import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, log_evaluation

# ------------------------------------------
# 1. Load dataset
# ------------------------------------------
DATA_PATH = "data/dramp_training_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ------------------------------------------
# 2. Target and feature selection
# ------------------------------------------
TARGET = "log_mean_MIC"

drop_cols = [
    "Sequence",
    "clean_seq",
    "log_mean_MIC"
]

X = df.drop(columns=drop_cols)
y = df[TARGET]

# ------------------------------------------
# 3. Split
# ------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ------------------------------------------
# 4. Scale non-embedding features only
# ------------------------------------------
def find_embedding_columns(df):
    numeric_cols = []
    for c in df.columns:
        try:
            int(c)
            numeric_cols.append(c)
        except:
            pass
    return numeric_cols

embedding_cols = find_embedding_columns(X)
non_embedding_cols = [c for c in X.columns if c not in embedding_cols]

scaler = StandardScaler()
scaler.fit(X_train[non_embedding_cols])

def scale_split(df):
    df_scaled = df.copy()
    df_scaled[non_embedding_cols] = scaler.transform(df[non_embedding_cols])
    return df_scaled

X_train_scaled = scale_split(X_train)
X_val_scaled   = scale_split(X_val)
X_test_scaled  = scale_split(X_test)

print("NaN:", np.any(np.isnan(X_train_scaled.values)))
print("Inf:", np.any(np.isinf(X_train_scaled.values)))
print(X_train_scaled.var().sort_values().head(30))
print(df.filter(regex="pca_").head())
print(y_train.describe())


# ------------------------------------------
# 5. Train LightGBM (mutation-sensitive regression)
# ------------------------------------------
model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.01,
    num_leaves=64,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1.0,
    reg_lambda=1.0,
    objective="regression",
    random_state=42
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric="l2",
    callbacks=[log_evaluation(100)]
)

# ------------------------------------------
# 6. Evaluate
# ------------------------------------------
def evaluate(split_name, Xs, ys):
    pred = model.predict(Xs)
    mae = mean_absolute_error(ys, pred)
    rmse = np.sqrt(mean_squared_error(ys, pred))
    r2 = r2_score(ys, pred)
    print(f"\n--- {split_name} ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

evaluate("TRAIN", X_train_scaled, y_train)
evaluate("VAL",   X_val_scaled, y_val)
evaluate("TEST",  X_test_scaled, y_test)

# ------------------------------------------
# 7. Save artifacts
# ------------------------------------------
joblib.dump(model, "models/mic_regressor_lgbm.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")

with open("models/feature_list.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("\nSaved:")
print("models/mic_regressor_lgbm.pkl")
print("models/feature_scaler.pkl")
print("models/feature_list.txt")

