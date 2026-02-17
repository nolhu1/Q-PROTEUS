"""
Figure 14: Mutation Effect Distribution Histogram (synthetic demo)

What this script does:
- Creates realistic synthetic mutation-effect data (Δ log10(MIC)) for:
  1) High-RRI candidate peptide (tight distribution near 0)
  2) Low-RRI reference peptide (wider distribution with a right tail)
- Plots overlaid histograms + a vertical line at Δ=0
- Computes summary stats + Welch’s t-test and prints them
- Saves the figure as a PNG

When you replace with real data:
- Replace `delta_high` and `delta_low` with your computed Δ arrays:
  Δ = predicted_log10MIC_mutant - predicted_log10MIC_wildtype
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# -----------------------------
# Synthetic data generation
# -----------------------------
def truncated_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float, n: int) -> np.ndarray:
    """Simple rejection-sampling truncated normal."""
    out: list[float] = []
    # Oversample in chunks to keep it fast and simple
    while len(out) < n:
        chunk = rng.normal(mean, sd, size=max(1000, n))
        chunk = chunk[(chunk >= low) & (chunk <= high)]
        out.extend(chunk.tolist())
    return np.array(out[:n], dtype=float)


def make_realistic_delta_distributions(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Choose a "realistic" mutant count for single-point mutants:
    # total mutants = L * 19; for L ~ 25 → 475 mutants
    n_mutants = 475

    # High-RRI candidate:
    # Most mutations cause near-zero change; occasional moderate losses; rare slight improvements (negative Δ).
    high_core = truncated_normal(rng, mean=0.06, sd=0.10, low=-0.25, high=0.45, n=int(n_mutants * 0.88))
    high_moderate_losses = truncated_normal(rng, mean=0.22, sd=0.08, low=0.05, high=0.60, n=int(n_mutants * 0.10))
    high_rare_bad = truncated_normal(rng, mean=0.55, sd=0.10, low=0.35, high=0.90, n=n_mutants - len(high_core) - len(high_moderate_losses))
    delta_high = np.concatenate([high_core, high_moderate_losses, high_rare_bad])

    # Low-RRI reference:
    # Wider distribution with a heavier right tail (more mutations degrade activity strongly).
    low_core = truncated_normal(rng, mean=0.28, sd=0.22, low=-0.25, high=0.90, n=int(n_mutants * 0.68))
    low_tail = truncated_normal(rng, mean=0.75, sd=0.25, low=0.25, high=1.60, n=int(n_mutants * 0.30))
    low_rare_extreme = truncated_normal(rng, mean=1.30, sd=0.18, low=0.90, high=1.85, n=n_mutants - len(low_core) - len(low_tail))
    delta_low = np.concatenate([low_core, low_tail, low_rare_extreme])

    # Shuffle to remove any ordering artifacts
    rng.shuffle(delta_high)
    rng.shuffle(delta_low)

    return delta_high, delta_low


# -----------------------------
# Replace this section with your real Δ arrays later
# -----------------------------
delta_high, delta_low = make_realistic_delta_distributions(seed=11)

# Example of what replacing looks like:
# delta_high = np.loadtxt("delta_high.csv", delimiter=",")
# delta_low  = np.loadtxt("delta_low.csv", delimiter=",")


# -----------------------------
# Stats
# -----------------------------
def summarize(x: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "frac_near0_abs_le_0.10": float(np.mean(np.abs(x) <= 0.10)),
        "frac_large_loss_ge_0.50": float(np.mean(x >= 0.50)),
    }


s_high = summarize(delta_high)
s_low = summarize(delta_low)

if SCIPY_AVAILABLE:
    # Welch's t-test (does not assume equal variance)
    t_res = ttest_ind(delta_high, delta_low, equal_var=False)
    p_value = float(t_res.pvalue)
else:
    p_value = float("nan")


print("=== Synthetic Figure 14 Summary (Δ log10(MIC)) ===")
print("High-RRI candidate:", s_high)
print("Low-RRI reference:", s_low)
print(f"Welch t-test p-value: {p_value:.3g}" if SCIPY_AVAILABLE else "SciPy not available; skipping t-test.")


# -----------------------------
# Plot (Figure 14)
# -----------------------------
# Choose bins that cover the plausible Δ range.
# You can lock these for consistent comparison across runs.
x_min = -0.30
x_max = 1.90
bins = 36

fig = plt.figure(figsize=(9, 5.2))
ax = plt.gca()

ax.hist(delta_high, bins=bins, range=(x_min, x_max), alpha=0.60, label="High-RRI candidate")
ax.hist(delta_low,  bins=bins, range=(x_min, x_max), alpha=0.60, label="Low-RRI reference")

ax.axvline(0.0, linestyle="--", linewidth=1.2)

ax.set_xlabel("Δ log10(MIC) = log10(MIC_mutant) − log10(MIC_wildtype)")
ax.set_ylabel("Frequency (single-point mutants)")
ax.set_title("Distribution of Predicted Mutation Effects on Antimicrobial Activity")

# Small stats box (keep short and readable)
stats_text = (
    f"High-RRI: mean={s_high['mean']:.2f}, sd={s_high['sd']:.2f}, n={int(s_high['n'])}\n"
    f"Low-RRI:  mean={s_low['mean']:.2f}, sd={s_low['sd']:.2f}, n={int(s_low['n'])}\n"
)
if SCIPY_AVAILABLE:
    stats_text += f"Welch t-test: p={p_value:.2g}"

ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85)
)

ax.legend()
ax.set_xlim(x_min, x_max)

plt.tight_layout()
out_path = "Figure14_mutation_effect_histogram.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved: {out_path}")
