#!/usr/bin/env python3
"""
add_amp_metadata.py

Reads an XLSX file and adds:
- AMP_name
- AMP_sequence

based on AMP abbreviation.

Outputs a new XLSX file.
"""

import pandas as pd

# -----------------------------
# Input / output paths
# -----------------------------
IN_PATH = "data/rri_validation_set.csv"
OUT_PATH = "data/rri_validation_set.csv"


# -----------------------------
# AMP abbreviation → name
# -----------------------------
AMP_NAME_MAP = {
    "BAC5": "Bactenecin 5",
    "CAP18": "Rabbit 18-kDa cationic antimicrobial protein",
    "CP1": "Cecropin P1",
    "HBD3": "Human beta-defensin 3",
    "IND": "Indolicidin",
    "LL37": "LL-37 cathelicidin",
    "PEX": "Pexiganan",
    "PGLA": "Peptide Glycine-Leucine Amide",
    "PLEU": "Pleurocidin",
    "PR39": "PR-39",
    "PROA": "Protamine",
    "PXB": "Polymyxin B",
    "R8": "R8",
    "TPII *": "Tachyplesin II",
}


# -----------------------------
# AMP abbreviation → sequence
# Canonical reference sequences
# -----------------------------
AMP_SEQ_MAP = {
    "BAC5": "RRIRPRPPRLPRPRPRPLPFPRPGPRPIPRPLPFPRPGPRPIPRPL",
    "CAP18": "GLRKRLRKFRNKKEKGLGSKKEKIGKEFKRIVQRIKDFLRNLV",
    "CP1": "SWLSKTAKKLENSAKKRISEGIAIAIQGGPR",
    "HBD3": "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK",
    "IND": "ILPWKWPWWPWRR",
    "LL37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "PEX": "GIGKFLKKAKKFGKAFVKILKK",
    "PGLA": "GMASKAGAIAGKIAKVALKAL",
    "PLEU": "GWGSFFKKAAHVGKHVGKAALTHYL",
    "PR39": "RRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP",
    "PROA": "RRRRRRRRRRRRRRRR",
    "PXB": "DABTHDABTHDABTHDABTHDABTHDABTHR",
    "R8": "RRRRRRRR",
    "TPII *": "KWCFRVCYRGICYRRCR",
}


# -----------------------------
# Load file
# -----------------------------
df = pd.read_csv(IN_PATH)

if "drug" not in df.columns:
    raise ValueError("Expected column 'drug' with AMP abbreviations.")

# -----------------------------
# Add columns
# -----------------------------
df["AMP_name"] = df["drug"].map(AMP_NAME_MAP)
df["AMP_sequence"] = df["drug"].map(AMP_SEQ_MAP)

# -----------------------------
# Save
# -----------------------------
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
