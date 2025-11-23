#!/usr/bin/env python3
"""
prepare_dataset_minimal.py

Outputs:
    data/dramp_training_dataset.csv

Features included:
    - clean_seq
    - log_mean_MIC
    - physicochemical features (valid ProteinAnalysis functions only)
    - 1-mer composition (20 dims)
    - PCA-compressed ESM2 embeddings (64 dims)

Everything is checked, consistent, and minimal.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import torch
import esm
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
in_path = "data/dramp_mic_expanded.xlsx"
out_path = "data/dramp_training_dataset.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# -----------------------------------------------------------------------------
# Load DRAMP MIC table
# -----------------------------------------------------------------------------
df = pd.read_excel(in_path)

# -----------------------------------------------------------------------------
# MIC parser
# -----------------------------------------------------------------------------
def parse_mic(x):
    if pd.isna(x):
        return None

    s = str(x).strip()
    # Normalize units
    s = (s.replace("μg/ml", "")
           .replace("µg/ml", "")
           .replace("μg/mL", "")
           .replace("µg/mL", "")
           .replace("uM", "")
           .replace("μM", ""))

    # Case: ± value
    if "±" in s:
        try:
            return float(s.split("±")[0].strip())
        except:
            return None

    # Case: ranges
    for dash in ["-", "–", "—"]:
        if dash in s:
            parts = [p.strip() for p in s.split(dash) if p.strip()]
            try:
                a = float(parts[0])
                b = float(parts[-1])
                return (a + b) / 2
            except:
                return None

    # Case: single numeric
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except:
            return None

    return None

df["MIC_numeric"] = df["MIC"].apply(parse_mic)
df = df.dropna(subset=["MIC_numeric"])
df = df[df["MIC_numeric"] > 0].reset_index(drop=True)
df["log_MIC"] = np.log10(df["MIC_numeric"])

# -----------------------------------------------------------------------------
# Collapse identical sequences → mean MIC
# -----------------------------------------------------------------------------
log_df = df.groupby("Sequence")["log_MIC"].mean().reset_index()
log_df.rename(columns={"log_MIC": "log_mean_MIC"}, inplace=True)

# -----------------------------------------------------------------------------
# Sequence cleaning
# -----------------------------------------------------------------------------
VALID = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(s):
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    s2 = "".join([c for c in s if c in VALID])
    return s2 if s2 else None

log_df["clean_seq"] = log_df["Sequence"].apply(clean_seq)
log_df = log_df.dropna(subset=["clean_seq"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Physicochemical features (ONLY valid functions)
# -----------------------------------------------------------------------------
def physchem(seq):
    try:
        p = ProteinAnalysis(seq)
    except:
        return {k: np.nan for k in [
            "net_charge","gravy","aromaticity",
            "instability_index","isoelectric_point",
            "helix_fraction","sheet_fraction","length"
        ]}

    # Safe wrappers
    def safe(fn):
        try:
            return fn()
        except:
            return np.nan

    feats = {}
    feats["net_charge"] = safe(lambda: p.charge_at_pH(7.0))
    feats["gravy"] = safe(p.gravy)
    feats["aromaticity"] = safe(p.aromaticity)
    feats["instability_index"] = safe(p.instability_index)
    feats["isoelectric_point"] = safe(p.isoelectric_point)

    # Secondary structure
    try:
        helix, turn, sheet = p.secondary_structure_fraction()
        feats["helix_fraction"] = helix
        feats["sheet_fraction"] = sheet
    except:
        feats["helix_fraction"] = np.nan
        feats["sheet_fraction"] = np.nan

    feats["length"] = len(seq)

    return feats

phys_df = pd.DataFrame(log_df["clean_seq"].apply(physchem).tolist())
dataset = pd.concat([log_df, phys_df], axis=1)

# -----------------------------------------------------------------------------
# 1-mer composition (20 AA)
# -----------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"

def k1(seq):
    cnt = Counter(seq)
    L = len(seq)
    return {aa: cnt.get(aa, 0) / L for aa in AA}

k1_df = pd.DataFrame(dataset["clean_seq"].apply(k1).tolist())
k1_df.columns = [f"kmer1_{aa}" for aa in AA]
dataset = pd.concat([dataset, k1_df], axis=1)

# -----------------------------------------------------------------------------
# ESM2 embeddings → PCA(64)
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading ESM2 model on:", device)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

seqs = dataset["clean_seq"].tolist()
pairs = [(str(i), s) for i, s in enumerate(seqs)]

B = 16
embs = []

with torch.no_grad():
    for i in range(0, len(pairs), B):
        batch = pairs[i:i+B]
        labels, seq_strs, toks = batch_converter(batch)
        toks = toks.to(device)

        out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]

        for j, (_, s) in enumerate(batch):
            L = len(s)
            emb = reps[j, 1:L+1].mean(0).cpu().numpy()
            embs.append(emb)

embs = np.array(embs)

print("Running PCA on ESM embeddings…")
pca = PCA(n_components=64, random_state=0)
embs_pca = pca.fit_transform(embs)

emb_df = pd.DataFrame(embs_pca, columns=[f"esm_pca_{i}" for i in range(64)])
dataset = pd.concat([dataset.reset_index(drop=True), emb_df], axis=1)

# -----------------------------------------------------------------------------
# Save final dataset
# -----------------------------------------------------------------------------
dataset.to_csv(out_path, index=False)
print("Saved:", out_path)

df = pd.read_csv("data/dramp_training_dataset.csv")
low = df.var().sort_values().head(20)
print(low)
print(pca.explained_variance_ratio_.sum())
print(df.corr()["log_mean_MIC"].abs().sort_values(ascending=False).head(30))

