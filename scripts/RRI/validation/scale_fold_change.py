import pandas as pd
import numpy as np

# -----------------------------
# Input / output paths
# -----------------------------
INPUT_CSV = "data/rri_validation_set.csv"
OUTPUT_CSV = "data/scaled_rri_validation.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# Drop rows without MIC data
df = df.dropna(subset=["MIC_fold_change"])

# Ensure numeric
df["MIC_fold_change"] = pd.to_numeric(df["MIC_fold_change"], errors="coerce")
df = df.dropna(subset=["MIC_fold_change"])

# -----------------------------
# Aggregate per AMP
# -----------------------------
summary = (
    df.groupby(["drug", "AMP_name", "AMP_sequence"])
      .agg(
          n_lines=("MIC_fold_change", "count"),
          median_MIC_fold_change=("MIC_fold_change", "median"),
          mean_MIC_fold_change=("MIC_fold_change", "mean"),
      )
      .reset_index()
)

# -----------------------------
# Log-transform resistance metric
# -----------------------------
summary["log10_median_MIC"] = np.log10(summary["median_MIC_fold_change"])
summary["log10_mean_MIC"] = np.log10(summary["mean_MIC_fold_change"])

# -----------------------------
# Save
# -----------------------------
summary.to_csv(OUTPUT_CSV, index=False)

print("Saved resistance summary to:", OUTPUT_CSV)
