import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Synthetic data (realistic-ish)
# -----------------------------
rng = np.random.default_rng(7)

def clipped_normal(mean, sd, low, high, n):
    x = rng.normal(mean, sd, n)
    return np.clip(x, low, high)

N = 200  # population size per algorithm

# GA: decent efficacy, higher toxicity, lower RRI
ga_eff = clipped_normal(0.90, 0.03, 0.80, 0.98, N)
ga_tox = clipped_normal(0.24, 0.05, 0.05, 0.40, N)
ga_rri = clipped_normal(0.55, 0.10, 0.10, 0.80, N)

# QIEA: better exploration -> slightly higher efficacy, lower tox, moderate RRI
q_eff  = clipped_normal(0.93, 0.025, 0.82, 0.99, N)
q_tox  = clipped_normal(0.20, 0.045, 0.03, 0.35, N)
q_rri  = clipped_normal(0.65, 0.10, 0.20, 0.85, N)

# QIEA+RRI: include a strong "top cluster" that hits your targets
qr_eff = clipped_normal(0.945, 0.02, 0.85, 0.995, N)
qr_tox = clipped_normal(0.17, 0.04, 0.02, 0.30, N)
qr_rri = clipped_normal(0.74, 0.08, 0.25, 0.95, N)

# Force 12 "poster-worthy" points into the superior region:
# Efficacy ≥ 0.95, Toxicity ≤ 0.20, RRI ≥ 0.80
k = 12
idx = rng.choice(N, size=k, replace=False)
qr_eff[idx] = rng.uniform(0.955, 0.990, size=k)
qr_tox[idx] = rng.uniform(0.060, 0.195, size=k)
qr_rri[idx] = rng.uniform(0.805, 0.930, size=k)

# -----------------------------
# Hypervolume (simple + robust)
# Monte Carlo HV in 3D
# Objectives: maximize efficacy, maximize (1 - toxicity), maximize RRI
# -----------------------------
def hypervolume_mc(eff, tox, rri, n_samples=250_000):
    x = eff
    y = 1.0 - tox
    z = rri

    ref = np.array([0.80, 0.60, 0.00])  # eff=0.80, (1-tox)=0.60 (tox=0.40), rri=0
    upper = np.array([1.00, 1.00, 1.00])

    P = np.column_stack([x, y, z])
    P = np.clip(P, ref, upper)

    samples = rng.uniform(ref, upper, size=(n_samples, 3))

    dominated = np.zeros(n_samples, dtype=bool)
    chunk = 25_000

    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        S = samples[start:end]

        comp = (P[:, None, :] >= S[None, :, :])
        dom = np.all(comp, axis=2)
        dominated[start:end] = np.any(dom, axis=0)

    box_volume = np.prod(upper - ref)
    return box_volume * dominated.mean()

hv_ga = hypervolume_mc(ga_eff, ga_tox, ga_rri)
hv_q  = hypervolume_mc(q_eff,  q_tox,  q_rri)
hv_qr = hypervolume_mc(qr_eff, qr_tox, qr_rri)

# -----------------------------
# Target-region counts + stats
# -----------------------------
def superior_region_count(eff, tox, rri):
    mask = (eff >= 0.95) & (tox <= 0.20) & (rri >= 0.80)
    return int(mask.sum()), mask

count_qr, mask_qr = superior_region_count(qr_eff, qr_tox, qr_rri)
count_ga, _ = superior_region_count(ga_eff, ga_tox, ga_rri)
count_q, _  = superior_region_count(q_eff,  q_tox,  q_rri)

# -----------------------------
# Print metrics (terminal output)
# -----------------------------
def pct(a, b):
    return 100.0 * (a - b) / b

print("\n=== Figure 9 Metrics (Synthetic Data) ===")
print(f"Hypervolume (MC) — GA:       {hv_ga:.6f}")
print(f"Hypervolume (MC) — QIEA:     {hv_q:.6f}")
print(f"Hypervolume (MC) — QIEA+RRI: {hv_qr:.6f}")
print(f"\nHV improvement QIEA+RRI vs GA:   {pct(hv_qr, hv_ga):.2f}%")
print(f"HV improvement QIEA+RRI vs QIEA: {pct(hv_qr, hv_q):.2f}%")

print("\nSuperior-region counts (eff≥0.95, tox≤0.20, rri≥0.80):")
print(f"GA:       {count_ga}")
print(f"QIEA:     {count_q}")
print(f"QIEA+RRI: {count_qr}")

# -----------------------------
# 3D Figure: easier to see
# Changes:
# - plot (1 - toxicity) so "higher is better" aligns across axes
# - bigger markers + edge outlines for depth cues
# - lighter panes + subtle grid
# - slightly stronger perspective + better camera angle
# - optional projection views (commented) for paper/poster clarity
# -----------------------------
# Transform toxicity to "safety" for visual consistency
ga_safe = 1.0 - ga_tox
q_safe  = 1.0 - q_tox
qr_safe = 1.0 - qr_tox

# Compute limits in the transformed space
eff_all  = np.concatenate([ga_eff, q_eff, qr_eff])
safe_all = np.concatenate([ga_safe, q_safe, qr_safe])
rri_all  = np.concatenate([ga_rri, q_rri, qr_rri])

def padded_limits(x, low_cap=None, high_cap=None, pad_frac=0.06):
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = max(hi - lo, 1e-9)
    pad = span * pad_frac
    lo2 = lo - pad
    hi2 = hi + pad
    if low_cap is not None:
        lo2 = max(lo2, low_cap)
    if high_cap is not None:
        hi2 = min(hi2, high_cap)
    return lo2, hi2

eff_lim  = padded_limits(eff_all,  low_cap=0.80, high_cap=1.00, pad_frac=0.06)
safe_lim = padded_limits(safe_all, low_cap=0.60, high_cap=1.00, pad_frac=0.08)  # safe=1-tox
rri_lim  = padded_limits(rri_all,  low_cap=0.00, high_cap=1.00, pad_frac=0.06)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

fig = plt.figure(figsize=(9.2, 6.8), dpi=220)
ax = fig.add_subplot(111, projection="3d")

# Perspective helps depth perception (smaller = stronger perspective)
try:
    ax.set_proj_type("persp")
except Exception:
    pass

# Make panes light (improves contrast without picking colors explicitly)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    try:
        axis.pane.set_alpha(0.05)
    except Exception:
        pass

# Subtle grid (3D can get noisy fast)
ax.grid(True, linewidth=0.5, alpha=0.25)

# Scatter with edges for depth cues (edgecolor defaults are fine)
scatter_kw = dict(alpha=0.65, linewidths=0.35, edgecolors="k")

ax.scatter(ga_eff, ga_safe, ga_rri, s=28, marker="o", label="GA", **scatter_kw)
ax.scatter(q_eff,  q_safe,  q_rri,  s=28, marker="^", label="QIEA", **scatter_kw)
ax.scatter(qr_eff, qr_safe, qr_rri, s=30, marker="s", label="QIEA + RRI", **scatter_kw)

# Emphasize the superior region points (use transformed y)
if count_qr > 0:
    ax.scatter(qr_eff[mask_qr], (1.0 - qr_tox[mask_qr]), qr_rri[mask_qr],
               s=90, alpha=0.95, marker="s", edgecolors="k", linewidths=0.6,
               label="QIEA+RRI (superior region)")

ax.set_title("Pareto-Optimal Solution Distributions for GA, QIEA, and QIEA+RRI\nAcross Efficacy, Toxicity, and Resistance Resilience Objectives", pad=10)

ax.set_xlabel("Predicted Efficacy (higher)", labelpad=10)
ax.set_ylabel("Predicted Safety = 1 − Toxicity (higher)", labelpad=10)
ax.set_zlabel("RRI (higher)", labelpad=10)

ax.set_xlim(*eff_lim)
ax.set_ylim(*safe_lim)
ax.set_zlim(*rri_lim)

# Keep the cube from looking stretched
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass

# Camera tuned for separation + depth
ax.view_init(elev=24, azim=-42)

# Fewer ticks to reduce clutter
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.zaxis.set_major_locator(plt.MaxNLocator(5))

ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True)

# 3D + titles behave better with manual margins
fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.88)

plt.show()

# OPTIONAL (use if you want 2D projections for maximum readability on a poster):
# 1) Efficacy vs Toxicity vs RRI is hard to parse; 3 small 2D plots can be clearer.
#    If you want, I can rewrite this to generate:
#    - Efficacy vs Toxicity (colored by RRI)
#    - Efficacy vs RRI (colored by Toxicity)
#    - Toxicity vs RRI (colored by Efficacy)