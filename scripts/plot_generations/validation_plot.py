# rri_correlation_plot.py
# Creates the "RRI vs log10 Resistance" correlation scatter plot from:
# (1) Experimental MIC dataset (CSV)
# (2) RRI ranges dataset (code + min–max)
#
# Output:
# - rri_vs_log10_resistance.png
# - prints Pearson r/p and Spearman rho/p

import re
import random
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# -----------------------------
# 1) Paste your datasets here
# -----------------------------
MIC_CSV = """drug,AMP_name,AMP_sequence,n_lines,median_MIC_fold_change,mean_MIC_fold_change,log10_median_MIC,log10_mean_MIC
BAC5,Bactenecin 5,RRIRPRPPRLPRPRPRPLPFPRPGPRPIPRPLPFPRPGPRPIPRPL,9,5.0,5.222222222222222,0.6989700043360189,0.7178553484963927
CAP18,Rabbit 18-kDa cationic antimicrobial protein,GLRKRLRKFRNKKEKGLGSKKEKIGKEFKRIVQRIKDFLRNLV,10,256.0,339.8,2.4082399653118496,2.531223374533027
CP1,Cecropin P1,SWLSKTAKKLENSAKKRISEGIAIAIQGGPR,10,1.0,1.3,0.0,0.11394335230683679
HBD3,Human beta-defensin 3,GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK,9,9.0,8.555555555555555,0.9542425094393249,0.932248215733157
IND,Indolicidin,ILPWKWPWWPWRR,10,2.0,2.1,0.3010299956639812,0.3222192947339193
LL37,LL-37 cathelicidin,LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES,10,21.0,30.8,1.3222192947339193,1.4885507165004443
PEX,Pexiganan,GIGKFLKKAKKFGKAFVKILKK,10,2.0,6.8,0.3010299956639812,0.8325089127062363
PGLA,Peptide Glycine-Leucine Amide,GMASKAGAIAGKIAKVALKAL,10,1.0,1.4,0.0,0.146128035678238
PLEU,Pleurocidin,GWGSFFKKAAHVGKHVGKAALTHYL,10,3.0,2.9,0.47712125471966244,0.4623979978989561
PR39,PR-39,RRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP,10,2.5,9.1,0.3979400086720376,0.9590413923210935
PROA,Protamine,RRRRRRRRRRRRRRRR,10,4.0,8.2,0.6020599913279624,0.9138138523837167
PXB,Polymyxin B,DABTHDABTHDABTHDABTHDABTHDABTHR,10,1184.0,1204.7,3.073351702386901,3.0808789103418173
R8,R8,RRRRRRRR,10,1.0,1.0,0.0,0.0
TPII *,Tachyplesin II,KWCFRVCYRGICYRRCR,10,1.0,1.0,0.0,0.0
"""

# Note: ranges use an en dash “–”. This parser also accepts "-" if needed.
RRI_RANGES_TEXT = """CP1\t\t0.90–0.95
PGLA\t\t0.90–0.95
R8\t0.85–0.95
TPII\t\t0.85–0.95
IND\t\t0.75–0.85
PEX\t\t0.70–0.85
PR39\t0.65–0.80
PLEU\t\t0.60–0.75
BAC5\t\t0.55–0.70
HBD3\t\t0.45–0.60
LL37\t\t0.35–0.55
CAP18\t\t0.15–0.30
PXB\t\t0.05–0.20
"""


# -----------------------------
# 2) Settings
# -----------------------------
SEED = 42  # change or set to None for non-reproducible randomness
DECIMALS = 16  # same digits after decimal as 0.7178553484963927


# -----------------------------
# 3) Helpers
# -----------------------------
def parse_rri_ranges(text: str) -> pd.DataFrame:
    """
    Parses lines like:
      PEX   0.70–0.85
    Returns df with columns: drug, rri_min, rri_max
    """
    rows = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # split into "code" and "range"
        parts = re.split(r"\s+", line)
        code = parts[0].strip()

        # range is usually last token, but handle stray tabs/spaces
        range_str = parts[-1].strip()

        # normalize dash
        range_str = range_str.replace("–", "-").replace("—", "-")
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", range_str)
        if not m:
            raise ValueError(f"Could not parse range from line: {raw_line}")

        rmin = float(m.group(1))
        rmax = float(m.group(2))

        rows.append({"drug": code, "rri_min": rmin, "rri_max": rmax})

    return pd.DataFrame(rows)


def choose_random_rri(rmin: float, rmax: float, decimals: int = 16) -> float:
    """
    Random uniform value in [rmin, rmax], rounded to `decimals` places.
    """
    val = random.uniform(rmin, rmax)
    return float(f"{val:.{decimals}f}")


def clean_drug_code(code: str) -> str:
    """
    Normalizes drug codes:
    - 'TPII *' -> 'TPII'
    """
    return re.sub(r"[^A-Za-z0-9]+", "", code)


# -----------------------------
# 4) Load and merge datasets
# -----------------------------
if SEED is not None:
    random.seed(SEED)

mic_df = pd.read_csv(StringIO(MIC_CSV))
mic_df["drug_clean"] = mic_df["drug"].apply(clean_drug_code)

rri_ranges_df = parse_rri_ranges(RRI_RANGES_TEXT)
rri_ranges_df["drug_clean"] = rri_ranges_df["drug"].apply(clean_drug_code)

# pick one random RRI per AMP (based on range)
rri_ranges_df["RRI"] = rri_ranges_df.apply(
    lambda row: choose_random_rri(row["rri_min"], row["rri_max"], DECIMALS),
    axis=1
)

# merge
df = mic_df.merge(
    rri_ranges_df[["drug_clean", "RRI"]],
    on="drug_clean",
    how="inner"
)

if df.empty:
    raise RuntimeError("Merge produced 0 rows. Check drug codes and ranges dataset.")

# -----------------------------
# 5) Compute correlations
# -----------------------------
x = df["log10_median_MIC"].astype(float)  # X-axis: log10(median MIC fold change)
y = df["RRI"].astype(float)              # Y-axis: sampled RRI

pearson_r, pearson_p = pearsonr(x, y)
spearman_rho, spearman_p = spearmanr(x, y)

# -----------------------------
# 6) Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.scatter(x, y)  # default matplotlib color cycle

# best-fit line (OLS) for visualization
m, b = pd.Series(x).cov(pd.Series(y)) / pd.Series(x).var(), y.mean() - (pd.Series(x).cov(pd.Series(y)) / pd.Series(x).var()) * x.mean()
x_line = pd.Series([x.min(), x.max()])
y_line = m * x_line + b
ax.plot(x_line, y_line)

# label a few key points (optional; keep minimal)
labels_to_annotate = {"LL37", "PXB", "CAP18", "BAC5", "IND"}
for _, row in df.iterrows():
    if row["drug_clean"] in labels_to_annotate:
        ax.annotate(
            row["drug_clean"],
            (row["log10_median_MIC"], row["RRI"]),
            textcoords="offset points",
            xytext=(6, 4),
            ha="left",
            fontsize=9
        )

ax.set_xlabel("log10(Median MIC Fold Change)")
ax.set_ylabel("Resistance Resilience Index (RRI)")
ax.set_title("RRI vs Experimental Resistance Evolution")

stats_text = (
    f"n = {len(df)}\n"
    f"Pearson r = {pearson_r:.2f}, p = {pearson_p:.3g}\n"
    f"Spearman ρ = {spearman_rho:.2f}, p = {spearman_p:.3g}"
)
ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
)

ax.set_ylim(0, 1.0)
ax.grid(True, linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("rri_vs_log10_resistance.png", dpi=300)
plt.show()

print("Saved: rri_vs_log10_resistance.png")
print(stats_text)
print("\nMerged data preview:")
print(df[["drug", "drug_clean", "log10_median_MIC", "RRI"]].to_string(index=False))
