import numpy as np
import matplotlib.pyplot as plt

# Optional: use SciPy if available for t-test; otherwise fall back to a simple Welch t-test.
try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def generate_runs(
    n_gens: int,
    n_runs: int,
    y0: float,
    asymptote: float,
    t90_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic 'best fitness' curves with realistic stochasticity.
    Curves rise toward an asymptote with run-to-run variation and mild noise,
    then are forced to be non-decreasing (best-so-far).
    """
    g = np.arange(n_gens)

    # Choose k so the expected curve reaches 90% of (asymptote - y0) by ~t90_target
    k_base = np.log(10.0) / max(1, t90_target)

    runs = []
    for _ in range(n_runs):
        # Run-to-run variation
        a = asymptote + rng.normal(0.0, 0.008)          # asymptote variability
        k = k_base * (1.0 + rng.normal(0.0, 0.10))      # speed variability
        k = max(k, 1e-6)

        # Base exponential approach curve
        y = a - (a - y0) * np.exp(-k * g)

        # Add mild, realistic noise that decreases over time (early training noisier)
        noise_scale = 0.010 * np.exp(-g / 250.0) + 0.002
        y = y + rng.normal(0.0, noise_scale, size=n_gens)

        # Best-so-far: enforce non-decreasing
        y = np.maximum.accumulate(y)

        # Clamp to plausible range
        y = np.clip(y, 0.0, 1.0)

        runs.append(y)

    return np.vstack(runs)


def gen_to_reach_fraction(mean_curve: np.ndarray, y0: float, frac: float = 0.90) -> int:
    """
    Return the first generation where mean_curve reaches y0 + frac*(max - y0),
    where max is the final mean (as an estimate of the plateau).
    """
    max_est = float(mean_curve[-1])
    target = y0 + frac * (max_est - y0)
    idx = np.argmax(mean_curve >= target)
    return int(idx)


def welch_ttest(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sided Welch's t-test p-value (fallback if SciPy is unavailable).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)

    t_num = mx - my
    t_den = np.sqrt(vx / nx + vy / ny)
    if t_den == 0:
        return 1.0

    t = t_num / t_den

    # Welch–Satterthwaite df
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    df = df_num / df_den if df_den > 0 else 1.0

    # Compute p-value via survival function of Student-t using SciPy if possible
    try:
        from scipy.stats import t as student_t
        p = 2.0 * student_t.sf(np.abs(t), df)
        return float(p)
    except Exception:
        # If SciPy isn't available, return a conservative placeholder.
        return 1.0


def main():
    rng = np.random.default_rng(7)

    n_gens = 501  # 0..500 inclusive
    n_runs = 5
    y0 = 0.45

    # Target plateaus consistent with your requirements
    GA_asym = 0.74
    QIEA_asym = 0.80
    QIEA_RRI_asym = 0.86

    # Target t90 behavior (on average)
    GA_t90 = 420          # >= 400
    QIEA_t90 = 280        # intermediate
    QIEA_RRI_t90 = 220    # <= 250

    ga = generate_runs(n_gens, n_runs, y0, GA_asym, GA_t90, rng)
    qiea = generate_runs(n_gens, n_runs, y0, QIEA_asym, QIEA_t90, rng)
    qiea_rri = generate_runs(n_gens, n_runs, y0, QIEA_RRI_asym, QIEA_RRI_t90, rng)

    g = np.arange(n_gens)

    # Mean and std across runs
    ga_mean, ga_std = ga.mean(axis=0), ga.std(axis=0, ddof=1)
    q_mean, q_std = qiea.mean(axis=0), qiea.std(axis=0, ddof=1)
    qr_mean, qr_std = qiea_rri.mean(axis=0), qiea_rri.std(axis=0, ddof=1)

    # Metrics: generations to reach 90% of (final_mean - y0)
    ga_t90_meas = gen_to_reach_fraction(ga_mean, y0, 0.90)
    q_t90_meas = gen_to_reach_fraction(q_mean, y0, 0.90)
    qr_t90_meas = gen_to_reach_fraction(qr_mean, y0, 0.90)

    # Final generation statistics (across runs)
    ga_final = ga[:, -1]
    q_final = qiea[:, -1]
    qr_final = qiea_rri[:, -1]

    # t-tests on final fitness
    if SCIPY_AVAILABLE:
        p_qr_vs_ga = float(ttest_ind(qr_final, ga_final, equal_var=False).pvalue)
        p_qr_vs_q = float(ttest_ind(qr_final, q_final, equal_var=False).pvalue)
    else:
        p_qr_vs_ga = welch_ttest(qr_final, ga_final)
        p_qr_vs_q = welch_ttest(qr_final, q_final)

    # Print metrics (terminal)
    print("=== Figure 7 Metrics (Synthetic Data) ===")
    print(f"Runs per algorithm: {n_runs}")
    print("")
    print("Final mean best fitness (mean ± SD across runs):")
    print(f"  GA:        {ga_final.mean():.4f} ± {ga_final.std(ddof=1):.4f}")
    print(f"  QIEA:      {q_final.mean():.4f} ± {q_final.std(ddof=1):.4f}")
    print(f"  QIEA+RRI:  {qr_final.mean():.4f} ± {qr_final.std(ddof=1):.4f}")
    print("")
    print("Generations to reach 90% of max (using final mean as max estimate):")
    print(f"  GA:        {ga_t90_meas} generations")
    print(f"  QIEA:      {q_t90_meas} generations")
    print(f"  QIEA+RRI:  {qr_t90_meas} generations")
    print("")
    print("Welch t-test on final generation best fitness:")
    print(f"  QIEA+RRI vs GA:   p = {p_qr_vs_ga:.6g}")
    print(f"  QIEA+RRI vs QIEA: p = {p_qr_vs_q:.6g}")
    print("")
    print("\n=== TARGET METRIC VALUES ===")

    print(f"QIEA+RRI t90 (target ≤ 250): {qr_t90_meas}")
    print(f"GA t90 (target ≥ 400): {ga_t90_meas}")

    print(f"Final QIEA+RRI mean fitness (target ≥ 0.85): {qr_final.mean():.4f}")
    print(f"Final QIEA mean fitness (target ≤ 0.80): {q_final.mean():.4f}")
    print(f"Final GA mean fitness (target ≤ 0.75): {ga_final.mean():.4f}")

    print(f"p-value (QIEA+RRI vs GA) (target < 0.01): {p_qr_vs_ga:.6e}")


    # Plot (poster-ready styling without specifying colors)
    plt.figure(figsize=(7.2, 4.8), dpi=200)

    # Mean ± SD shading
    plt.fill_between(g, ga_mean - ga_std, ga_mean + ga_std, alpha=0.15)
    plt.plot(g, ga_mean, linewidth=2.2, label="GA")

    plt.fill_between(g, q_mean - q_std, q_mean + q_std, alpha=0.15)
    plt.plot(g, q_mean, linewidth=2.2, label="QIEA")

    plt.fill_between(g, qr_mean - qr_std, qr_mean + qr_std, alpha=0.15)
    plt.plot(g, qr_mean, linewidth=2.2, label="QIEA + RRI")

    plt.title(
    "Comparative Convergence of GA, QIEA, and QIEA+RRI\n"
    "Mean Best Fitness Across Generations", pad=10)    
    plt.xlabel("Generation")
    plt.ylabel("Mean Best Fitness (Efficacy – Toxicity + RRI)")
    plt.xlim(0, 500)

    y_min = min((ga_mean - ga_std).min(), (q_mean - q_std).min(), (qr_mean - qr_std).min())
    y_max = max((ga_mean + ga_std).max(), (q_mean + q_std).max(), (qr_mean + qr_std).max())
    plt.ylim(max(0.0, y_min - 0.02), min(1.0, y_max + 0.02))

    plt.grid(True, linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
