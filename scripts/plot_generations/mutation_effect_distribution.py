import numpy as np
import matplotlib.pyplot as plt

# Optional: statistical tests (falls back gracefully if SciPy is unavailable)
try:
    from scipy.stats import mannwhitneyu, ttest_ind
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -----------------------------
# Synthetic, biologically plausible mutational effect model
# -----------------------------
# Δfitness = (predicted activity of mutant) / (predicted activity of wild-type)
# ~1.0 means little/no loss; smaller values mean loss of function.
# We model realistic distributions using bounded beta-mixtures + small prediction noise.
def clamp(x, lo=0.0, hi=1.2):
    return np.clip(x, lo, hi)

def make_high_rri(n, rng):
    # Mostly mild effects near ~0.95, small deleterious tail
    main = rng.beta(45, 3, int(n * 0.85))   # tight near high retention
    tail = rng.beta(10, 4, n - len(main))   # occasional stronger losses
    x = np.concatenate([main, tail])
    x = x + rng.normal(0.0, 0.02, size=n)   # model noise
    return clamp(x, 0.0, 1.2)

def make_low_rri(n, rng):
    # Broader distribution + heavier deleterious tail
    main = rng.beta(9, 6, int(n * 0.75))    # moderate losses
    tail = rng.beta(2, 5, n - len(main))    # severe losses
    x = np.concatenate([main, tail])
    x = x + rng.normal(0.0, 0.03, size=n)   # slightly larger noise
    return clamp(x, 0.0, 1.2)


# -----------------------------
# Generate data
# -----------------------------
rng = np.random.default_rng(7)
N = 1000

high = make_high_rri(N, rng)
low  = make_low_rri(N, rng)

# -----------------------------
# Metrics
# -----------------------------
def pct_at_least(x, thresh):
    return 100.0 * float(np.mean(x >= thresh))

high_mean = float(np.mean(high))
high_std  = float(np.std(high, ddof=1))
high_pct_90 = pct_at_least(high, 0.90)

low_mean = float(np.mean(low))
low_std  = float(np.std(low, ddof=1))

# For low-RRI, "retain activity" here means Δfitness >= 0.50 (still meaningfully active),
# while we also print >= 0.90 for direct comparison.
low_pct_50 = pct_at_least(low, 0.50)
low_pct_90 = pct_at_least(low, 0.90)

# Statistical tests
if HAVE_SCIPY:
    mw = mannwhitneyu(high, low, alternative="greater")  # expect high > low
    tt = ttest_ind(high, low, equal_var=False, alternative="greater")
    p_mw = float(mw.pvalue)
    p_tt = float(tt.pvalue)
else:
    # Simple permutation test fallback (mean difference)
    def permutation_pvalue_greater(a, b, n_perm=5000, seed=7):
        r = np.random.default_rng(seed)
        obs = float(np.mean(a) - np.mean(b))
        pooled = np.concatenate([a, b]).copy()
        count = 0
        for _ in range(n_perm):
            r.shuffle(pooled)
            a_perm = pooled[:len(a)]
            b_perm = pooled[len(a):]
            if float(np.mean(a_perm) - np.mean(b_perm)) >= obs:
                count += 1
        return (count + 1) / (n_perm + 1)

    p_mw = float(permutation_pvalue_greater(high, low, n_perm=5000, seed=7))
    p_tt = float("nan")

# -----------------------------
# Print metrics (terminal)
# -----------------------------
print("=== Figure 14 Synthetic Data Metrics ===")
print(f"N mutants per peptide: {N}")

print("\nHigh-RRI candidate:")
print(f"  Mean Δfitness: {high_mean:.4f}")
print(f"  Std  Δfitness: {high_std:.4f}")
print(f"  % mutants with Δfitness ≥ 0.90: {high_pct_90:.2f}%")

print("\nLow-RRI reference:")
print(f"  Mean Δfitness: {low_mean:.4f}")
print(f"  Std  Δfitness: {low_std:.4f}")
print(f"  % mutants with Δfitness ≥ 0.50 (retain activity): {low_pct_50:.2f}%")
print(f"  % mutants with Δfitness ≥ 0.90: {low_pct_90:.2f}%")

print("\nBetween-distribution significance (High > Low):")
print(f"  Mann–Whitney U p-value: {p_mw:.6g}")
if HAVE_SCIPY:
    print(f"  Welch t-test p-value:   {p_tt:.6g}")
else:
    print("  Welch t-test p-value:   (SciPy not available)")

# -----------------------------
# Plot: journal-ready overlaid histograms + inset bar
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

bins = np.linspace(0.0, 1.2, 41)
ax.hist(high, bins=bins, density=True, alpha=0.55, label="High-RRI candidate")
ax.hist(low,  bins=bins, density=True, alpha=0.55, label="Low-RRI reference")

ax.axvline(0.90, linewidth=1.2, linestyle="--")
ax.axvline(0.50, linewidth=1.2, linestyle=":")

ax.set_xlim(0.0, 1.2)
ax.set_xlabel("Mutational retention (Δfitness = mutant / wild-type)")
ax.set_ylabel("Density")
ax.set_title(" Distribution of Single-Point Mutational Fitness Effects in High- and Low-RRI Peptides")

annotation = (
    "High-RRI:\n"
    f"  mean={high_mean:.2f}, sd={high_std:.2f}\n"
    f"  ≥0.90: {high_pct_90:.0f}%\n"
    "Low-RRI:\n"
    f"  mean={low_mean:.2f}, sd={low_std:.2f}\n"
    f"  ≥0.50: {low_pct_50:.0f}%\n"
    f"MW p={p_mw:.2g}"
)
ax.text(
    0.02, 0.98, annotation,
    transform=ax.transAxes,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=0.8)
)

ax.legend(frameon=False, loc="upper right")

# Inset bar chart:
# Use numeric x positions to avoid categorical-unit conversion issues.
inset = ax.inset_axes([0.62, 0.10, 0.34, 0.32])
xpos = np.array([0, 1], dtype=float)
vals = [high_pct_90, low_pct_50]
inset.bar(xpos, vals)
inset.set_ylim(0, 100)
inset.set_ylabel("% mutants", fontsize=9)
inset.set_xticks(xpos)
inset.set_xticklabels(["High ≥0.90", "Low ≥0.50"], fontsize=8)
inset.tick_params(axis="y", labelsize=8)
inset.set_title("Retention thresholds", fontsize=9)

fig.tight_layout()
plt.show()
