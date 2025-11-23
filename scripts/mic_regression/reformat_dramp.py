import pandas as pd
import re
import numpy as np

# ---------- CONFIG ----------
INPUT_FILE = "data/DRAMP_general_amps.xlsx"
OUTPUT_FILE = "data/dramp_mic_expanded.xlsx"
TARGET_COLUMN = "Target_Organism"
SEQUENCE_COLUMN = "Sequence"
ID_COLUMN = "DRAMP_ID"
# ----------------------------

df = pd.read_excel(INPUT_FILE)

expanded_rows = []

# Regex to detect MIC values:
mic_pattern = re.compile(
    r"MIC\s*=\s*([\d\.]+(?:\s*[-–]\s*[\d\.]+)?)"  # value or range
    r"(?:\s*[±+-]\s*[\d\.]+)?"                    # optional ± SD
    r".*?(µg/ml|ug/ml|μg/ml|μM|uM)?"              # units
    , re.IGNORECASE
)

# Regex to split organism entries
entry_split_pattern = re.compile(r"\)\s*,|\s*;\s*")

# Optional: dictionary of peptide MW if known, e.g., {"DRAMP0001": 1200.5, ...}
# If available, you can convert µg/ml → µM using mic_um = (mic_ugml / MW) * 1000
peptide_mw_dict = {}  # fill in if MW known

for idx, row in df.iterrows():
    cell = str(row.get(TARGET_COLUMN, ""))

    if not cell or cell.strip().lower() == "nan":
        continue
    length = row.get("Sequence_Length", None)

    entries = entry_split_pattern.split(cell)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        mic_match = mic_pattern.search(entry)
        if not mic_match:
            continue

        mic_str = mic_match.group(1).replace("..", ".").strip()
        unit = mic_match.group(2).lower() if mic_match.group(2) else "ug/ml"

        # Handle range
        if "-" in mic_str or "–" in mic_str:
            parts = re.split(r"[-–]", mic_str)
            try:
                mic_value = np.mean([float(p) for p in parts])
            except ValueError:
                continue
        else:
            try:
                mic_value = float(mic_str)
            except ValueError:
                continue

        # Convert µg/ml → µM if MW available
        mw = peptide_mw_dict.get(row.get(ID_COLUMN))
        if "ug/ml" in unit and mw is not None:
            mic_value = (mic_value / mw) * 1000  # µM
            unit = "µM"

        # Organism name
        organism = entry.split("MIC")[0].strip()
        organism = organism.rstrip("(). ").strip()

        expanded_rows.append({
            ID_COLUMN: row.get(ID_COLUMN),
            SEQUENCE_COLUMN: row.get(SEQUENCE_COLUMN),
            "Sequence_Length": length,
            "Organism": organism,
            "MIC": mic_value,
            "Unit": unit
        })

expanded_df = pd.DataFrame(expanded_rows)
expanded_df.to_excel(OUTPUT_FILE, index=False)
print(f"Saved expanded MIC dataset with normalized units to {OUTPUT_FILE}")
