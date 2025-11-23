import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/dramp_mic_expanded.xlsx"
df = pd.read_excel(file_path)

# Print first few rows
print(df.head())

# Count unique sequences
num_unique_sequences = df['Sequence'].nunique()
print(f"Number of unique sequences: {num_unique_sequences}")

# Count occurrences of each organism
organism_counts = df['Organism'].value_counts()

# Get the most common organism
most_common_organism = organism_counts.idxmax()
most_common_count = organism_counts.max()

print(f"Most common organism: {most_common_organism}")
print(f"Number of occurrences: {most_common_count}")

# Filter dataset for the most common organism
common_org_df = df[df['Organism'] == most_common_organism].copy()

# Convert MIC values to numeric, handling ranges and "±" values
def parse_mic(mic_str):
    if pd.isna(mic_str):
        return None
    mic_str = str(mic_str).replace("μg/ml","").replace("µg/ml","").replace("uM","").strip()
    # Handle "x ± y" format
    if "±" in mic_str:
        return float(mic_str.split("±")[0].strip())
    # Handle range "x-y"
    if "-" in mic_str:
        parts = mic_str.split("-")
        return (float(parts[0].strip()) + float(parts[1].strip())) / 2
    # Otherwise just convert to float
    try:
        return float(mic_str)
    except:
        return None

common_org_df['MIC_numeric'] = common_org_df['MIC'].apply(parse_mic)

# Drop rows where MIC_numeric is None
common_org_df = common_org_df.dropna(subset=['MIC_numeric'])

# Compute statistics
mic_mean = common_org_df['MIC_numeric'].mean()
mic_median = common_org_df['MIC_numeric'].median()
mic_min = common_org_df['MIC_numeric'].min()
mic_max = common_org_df['MIC_numeric'].max()
mic_std = common_org_df['MIC_numeric'].std()

print(f"\nMIC statistics for {most_common_organism}:")
print(f"Mean: {mic_mean:.3f}")
print(f"Median: {mic_median:.3f}")
print(f"Min: {mic_min:.3f}")
print(f"Max: {mic_max:.3f}")
print(f"Std: {mic_std:.3f}")

# Optional: histogram
plt.hist(common_org_df['MIC_numeric'], bins=20, edgecolor='black')
plt.title(f"MIC Distribution for {most_common_organism}")
plt.xlabel("MIC")
plt.ylabel("Count")
plt.show()
