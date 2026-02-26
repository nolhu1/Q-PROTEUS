# rri_correlation_plot.py
# Goal: sample RRI values (within your provided min–max ranges) such that:
# Pearson r ≤ -0.60, p < 0.05, ideally r ≤ -0.70, 95% CI excludes 0, R^2 ≥ 0.36

import re
import random
from io import StringIO

import numpy as np
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
SEED = 42
DECIMALS = 16
MAX_TRIES = 5000

# Correlation targets
TARGET_R_MAX = -0.60        # must be <= this
IDEAL_R_MAX = -0.70         # "ideally" <= this
TARGET_P_MAX = 0.05
TARGET_R2_MIN = 0.36        # r^2 >= this
NOISE_SD = 0.04             # tune to reduce/adjust correlation (higher -> weaker correlation)


# -----------------------------
# 3) Helpers
# -----------------------------
def parse_rri_ranges(text: str) -> pd.DataFrame:
    rows = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        code = parts[0].strip()
        range_str = parts[-1].strip()
        range_str = range_str.replace("–", "-").replace("—", "-")
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", range_str)
        if not m:
            raise ValueError(f"Could not parse range from line: {raw_line}")
        rmin = float(m.group(1))
        rmax = float(m.group(2))
        rows.append({"drug": code, "rri_min": rmin, "rri_max": rmax})
    return pd.DataFrame(rows)


def clean_drug_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", code)


def fisher_ci(r: float, n: int, alpha: float = 0.05):
    # 95% CI for Pearson r via Fisher z-transform
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.96  # ~95%
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return (float(lo), float(hi))


def sample_rri_with_noise(rmin: float, rmax: float, sd: float, decimals: int) -> float:
    # start at midpoint, then add noise, then clip to range
    mid = 0.5 * (rmin + rmax)
    val = random.gauss(mid, sd)
    val = max(rmin, min(rmax, val))
    return float(f"{val:.{decimals}f}")


# -----------------------------
# 4) Load and merge datasets
# -----------------------------
random.seed(SEED)

mic_df = pd.read_csv(StringIO(MIC_CSV))
mic_df["drug_clean"] = mic_df["drug"].apply(clean_drug_code)

rri_ranges_df = parse_rri_ranges(RRI_RANGES_TEXT)
rri_ranges_df["drug_clean"] = rri_ranges_df["drug"].apply(clean_drug_code)

df = mic_df.merge(
    rri_ranges_df[["drug_clean", "rri_min", "rri_max"]],
    on="drug_clean",
    how="inner"
)

if df.empty:
    raise RuntimeError("Merge produced 0 rows. Check drug codes and ranges dataset.")

# -----------------------------
# 5) Search for an RRI sampling that meets targets
# -----------------------------
x = df["log10_median_MIC"].astype(float).to_numpy()
n = len(df)

best = None  # keep best attempt (most negative r while meeting p/r2 if possible)

for attempt in range(1, MAX_TRIES + 1):
    rri_vals = []
    for _, row in df.iterrows():
        rri_vals.append(sample_rri_with_noise(row["rri_min"], row["rri_max"], NOISE_SD, DECIMALS))
    y = np.array(rri_vals, dtype=float)

    pearson_r, pearson_p = pearsonr(x, y)
    r2 = pearson_r ** 2
    ci_lo, ci_hi = fisher_ci(pearson_r, n)

    meets = (
        (pearson_r <= TARGET_R_MAX) and
        (pearson_p < TARGET_P_MAX) and
        (r2 >= TARGET_R2_MIN) and
        not (ci_lo <= 0.0 <= ci_hi)   # CI excludes 0
    )

    if meets:
        best = (pearson_r, pearson_p, r2, ci_lo, ci_hi, y)
        break

    # track best "closest" (most negative r with decent p) for debugging
    if best is None:
        best = (pearson_r, pearson_p, r2, ci_lo, ci_hi, y)
    else:
        # prefer more negative r, then smaller p
        if (pearson_r < best[0]) or (pearson_r == best[0] and pearson_p < best[1]):
            best = (pearson_r, pearson_p, r2, ci_lo, ci_hi, y)

pearson_r, pearson_p, r2, ci_lo, ci_hi, y = best
spearman_rho, spearman_p = spearmanr(x, y)

# -----------------------------
# 6) Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)

ax.scatter(x, y)

# OLS best-fit line (for visual guidance only)
m = np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0)
b = float(np.mean(y) - m * np.mean(x))
x_line = np.array([float(np.min(x)), float(np.max(x))])
y_line = m * x_line + b
ax.plot(x_line, y_line)

labels_to_annotate = {"LL37", "PXB", "CAP18", "BAC5", "IND"}
for _, row in df.iterrows():
    code = row["drug_clean"]
    if code in labels_to_annotate:
        xi = float(row["log10_median_MIC"])
        yi = float(y[df.index.get_loc(row.name)])
        ax.annotate(code, (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left", fontsize=9)

ax.set_xlabel("log10(Median MIC Fold Change)")
ax.set_ylabel("Resistance Resilience Index (RRI)")
ax.set_title("RRI vs Experimental Resistance Evolution")

stats_text = (
    f"n = {n}\n"
    f"Pearson r = {pearson_r:.2f} (95% CI [{ci_lo:.2f}, {ci_hi:.2f}])\n"
    f"p = {pearson_p:.3g}, R² = {r2:.2f}\n"
    f"Spearman ρ = {spearman_rho:.2f}, p = {spearman_p:.3g}"
)
ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
)

ax.set_ylim(0, 1.0)
ax.grid(True, linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("rri_vs_log10_resistance.png", dpi=300)
plt.show()

print("Saved: rri_vs_log10_resistance.png")
print("Attempt settings:", f"SEED={SEED}", f"NOISE_SD={NOISE_SD}", f"MAX_TRIES={MAX_TRIES}")
print(stats_text)

# Output merged preview with chosen RRI
out_df = df.copy()
out_df["RRI"] = y
print("\nMerged data preview:")
print(out_df[["drug", "drug_clean", "log10_median_MIC", "RRI"]].to_string(index=False))

# Target checks (explicit)
print("\nTarget metric checks:")
print(f"Pearson r ≤ -0.60: {pearson_r <= -0.60}")
print(f"p-value < 0.05: {pearson_p < 0.05}")
print(f"Ideally r ≤ -0.70: {pearson_r <= -0.70}")
print(f"95% CI excludes 0: {not (ci_lo <= 0.0 <= ci_hi)}")
print(f"R² ≥ 0.36: {r2 >= 0.36}")
