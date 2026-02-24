import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -----------------------------
# Synthetic Figure 8 generator
# -----------------------------
np.random.seed(42)

GENS = 500
SEEDS = 5
g = np.arange(GENS + 1)

H_MAX = np.log2(20)  # ~4.321928
assert 4.31 < H_MAX < 4.33

def make_curve(algo: str, seed: int) -> np.ndarray:
    """
    Create a synthetic entropy-vs-generation curve that meets the target constraints:
    - Initial entropy ≈ 4.32
    - GA drops below 1.5 by gen 200
    - QIEA maintains entropy ≥ 2.5 until gen 400
    - QIEA+RRI stays ≥ 2.0 (throughout)
    """
    rng = np.random.default_rng(seed)

    # Small run-to-run variations
    start = H_MAX + rng.normal(0, 0.03)

    if algo == "GA":
        # Fast collapse to low entropy
        tau = 75 + rng.normal(0, 5)           # faster decay
        floor = 0.95 + rng.normal(0, 0.06)    # final entropy around ~1.0
        curve = floor + (start - floor) * np.exp(-g / tau)

        # Add mild noise that shrinks over time
        noise = rng.normal(0, 0.08, size=g.size) * np.exp(-g / 250)
        curve = curve + noise

    elif algo == "QIEA":
        # Slow decay that stays high for longer
        tau = 350 + rng.normal(0, 15)         # slower decay
        floor = 2.10 + rng.normal(0, 0.08)    # final around ~2.1
        curve = floor + (start - floor) * np.exp(-g / tau)

        # Mild noise
        noise = rng.normal(0, 0.06, size=g.size) * np.exp(-g / 350)
        curve = curve + noise

        # Enforce constraint: ≥ 2.5 until generation 400
        curve[:401] = np.maximum(curve[:401], 2.52 + rng.normal(0, 0.02))

    elif algo == "QIEA+RRI":
        # Slightly lower than QIEA due to added selective pressure
        tau = 260 + rng.normal(0, 12)
        floor = 1.95 + rng.normal(0, 0.07)
        curve = floor + (start - floor) * np.exp(-g / tau)

        noise = rng.normal(0, 0.06, size=g.size) * np.exp(-g / 320)
        curve = curve + noise

        # Enforce constraint: ≥ 2.0 throughout
        curve = np.maximum(curve, 2.02 + rng.normal(0, 0.015, size=g.size))

    else:
        raise ValueError("Unknown algorithm")

    # Entropy cannot exceed max theoretical (numerical safety)
    curve = np.clip(curve, 0.0, H_MAX)

    # Ensure the first point is ~H_MAX
    curve[0] = start
    return curve

def build_runs(algo: str) -> np.ndarray:
    runs = []
    for s in range(SEEDS):
        runs.append(make_curve(algo, seed=1000 * (hash(algo) % 1000) + s))
    return np.vstack(runs)  # shape: (SEEDS, GENS+1)

runs_ga = build_runs("GA")
runs_qiea = build_runs("QIEA")
runs_qiea_rri = build_runs("QIEA+RRI")

# -----------------------------
# Verify target constraints
# -----------------------------
gen200 = 200
gen400 = 400

ga_mean_200 = runs_ga.mean(axis=0)[gen200]
qiea_mean_400 = runs_qiea.mean(axis=0)[gen400]
qiea_rri_min = runs_qiea_rri.min()

print(f"Initial entropy (mean across algos at gen0): "
      f"{np.mean([runs_ga[:,0].mean(), runs_qiea[:,0].mean(), runs_qiea_rri[:,0].mean()]):.3f} "
      f"(target ~{H_MAX:.3f})")

print(f"GA mean entropy at gen 200: {ga_mean_200:.3f} (target < 1.5)")
print(f"QIEA mean entropy at gen 400: {qiea_mean_400:.3f} (target ≥ 2.5)")
print(f"QIEA+RRI min entropy overall: {qiea_rri_min:.3f} (target ≥ 2.0)")

# -----------------------------
# Statistical difference (p < 0.05)
# Use generation 400 as the comparison point (clear separation).
# -----------------------------
ga_at_400 = runs_ga[:, gen400]
qiea_at_400 = runs_qiea[:, gen400]
qiea_rri_at_400 = runs_qiea_rri[:, gen400]

t_ga_vs_qiea = ttest_ind(ga_at_400, qiea_at_400, equal_var=False)
t_ga_vs_qiea_rri = ttest_ind(ga_at_400, qiea_rri_at_400, equal_var=False)

print("\nStat tests at generation 400 (Welch t-test):")
print(f"GA vs QIEA:      p = {t_ga_vs_qiea.pvalue:.6f} (target < 0.05)")
print(f"GA vs QIEA+RRI:  p = {t_ga_vs_qiea_rri.pvalue:.6f} (target < 0.05)")

# -----------------------------
# Plot Figure 8
# -----------------------------
def plot_with_band(runs: np.ndarray, label: str):
    mean = runs.mean(axis=0)
    sd = runs.std(axis=0, ddof=1)
    plt.plot(g, mean, label=label)
    plt.fill_between(g, mean - sd, mean + sd, alpha=0.2)

plt.figure(figsize=(10, 6))
plot_with_band(runs_ga, "GA")
plot_with_band(runs_qiea, "QIEA")
plot_with_band(runs_qiea_rri, "QIEA+RRI")

plt.axhline(H_MAX, linestyle="--", linewidth=1, alpha=0.7)
plt.text(5, H_MAX - 0.08, f"max entropy = log2(20) = {H_MAX:.2f}", fontsize=9)

plt.axvline(gen200, linestyle=":", linewidth=1, alpha=0.6)
plt.axvline(gen400, linestyle=":", linewidth=1, alpha=0.6)
plt.text(gen200 + 5, 0.4, "gen 200", fontsize=9, alpha=0.8)
plt.text(gen400 + 5, 0.4, "gen 400", fontsize=9, alpha=0.8)

plt.title("Population Diversity (Shannon Entropy) Over Generations")
plt.xlabel("Generation")
plt.ylabel("Mean Shannon Entropy (bits)")
plt.ylim(0.0, H_MAX + 0.2)
plt.xlim(0, GENS)
plt.legend()
plt.tight_layout()
plt.show()
