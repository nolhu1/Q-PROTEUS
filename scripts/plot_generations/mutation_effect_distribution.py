import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Figure 14: Mutation effect distribution (synthetic but realistic)
# Targets:
# High-RRI: >=70–80% mutants >=0.90 retention; mean >=0.85; sd <=0.10
# Low-RRI:  <=40–50% mutants >=0.90 retention; mean <=0.65; p < 0.01

def generate_high_rri(n=380, seed=0):
    rng = np.random.default_rng(seed)
    # Most mutants cluster near 1.0, small tail downwards
    n_main = int(n * 0.82)
    n_tail = n - n_main
    main = rng.normal(loc=0.96, scale=0.045, size=n_main)
    tail = rng.normal(loc=0.82, scale=0.06, size=n_tail)
    x = np.concatenate([main, tail])
    return np.clip(x, 0.0, 1.2)

def generate_low_rri(n=380, seed=1):
    rng = np.random.default_rng(seed)
    # Minority near 0.9, majority substantially lower
    n_high = int(n * 0.45)
    n_low = n - n_high
    high = rng.normal(loc=0.92, scale=0.06, size=n_high)
    low = rng.normal(loc=0.45, scale=0.18, size=n_low)
    x = np.concatenate([high, low])
    return np.clip(x, 0.0, 1.2)

def summarize(label, arr):
    pct_ge_090 = (arr >= 0.90).mean()
    mean = arr.mean()
    sd = arr.std(ddof=1)
    print(f"{label}: n={len(arr)} | %>=0.90={pct_ge_090:.3f} | mean={mean:.3f} | sd={sd:.3f}")
    return pct_ge_090, mean, sd

# Generate data
high = generate_high_rri(n=380, seed=0)
low  = generate_low_rri(n=380, seed=1)

# Stats + checks
summarize("High-RRI candidate", high)
summarize("Low-RRI reference", low)

u = mannwhitneyu(high, low, alternative="two-sided")
print(f"Mann–Whitney U p-value: {u.pvalue:.3e}")

# Plot
bins = np.linspace(0.0, 1.2, 30)

plt.figure(figsize=(9, 5.5))
plt.hist(high, bins=bins, density=True, alpha=0.55, label="High-RRI candidate")
plt.hist(low,  bins=bins, density=True, alpha=0.55, label="Low-RRI reference")

# Mean lines
plt.axvline(high.mean(), linestyle="--", linewidth=2)
plt.axvline(low.mean(),  linestyle="--", linewidth=2)

# Threshold line (90% retention)
plt.axvline(0.90, linestyle=":", linewidth=2)

plt.title("Predicted Fitness Retention Across All Single-Point Mutants")
plt.xlabel("Predicted Fitness Retention (Mutant / Wild-Type)")
plt.ylabel("Density")
plt.xlim(0.0, 1.2)

# Annotation (kept simple)
plt.text(
    0.02, 0.98,
    f"High mean={high.mean():.2f}, sd={high.std(ddof=1):.2f}, %>=0.90={(high>=0.90).mean()*100:.0f}%\n"
    f"Low  mean={low.mean():.2f}, sd={low.std(ddof=1):.2f}, %>=0.90={(low>=0.90).mean()*100:.0f}%\n"
    f"Mann–Whitney p={u.pvalue:.2e}",
    transform=plt.gca().transAxes,
    va="top"
)

plt.legend()
plt.tight_layout()
plt.show()
