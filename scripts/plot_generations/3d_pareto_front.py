"""
Figure 9: 3D Pareto Front Comparison (REALISTIC SYNTHETIC DATA)

This script generates synthetic populations for:
  1) GA
  2) QIEA
  3) QIEA+RRI
and highlights:
  4) QIEA+RRI Pareto Front

It then computes hypervolume (Monte Carlo) on transformed objectives:
  - maximize efficacy
  - maximize (1 - toxicity)
  - maximize RRI

The generator is calibrated (and will auto-tune if needed) to meet targets:
  - HV(QIEA+RRI) / HV(GA) in [1.20, 1.30]
  - HV(QIEA+RRI) / HV(QIEA) in [1.10, 1.15]
  - Superior region points (eff>=0.95, tox<=0.20, rri>=0.80) in [5, 10]

No assertions are used; instead, the script auto-adjusts until targets are met,
then plots and prints the achieved metrics.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Pareto + Hypervolume helpers
# -----------------------------

def nondominated_mask(points_max: np.ndarray) -> np.ndarray:
    """
    points_max: (N, 3) array where ALL objectives are to be maximized.
    Returns boolean mask of nondominated points.
    """
    n = points_max.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        p = points_max[i]
        dominated_by_any = np.any(
            (np.all(points_max >= p, axis=1) & np.any(points_max > p, axis=1))
        )
        if dominated_by_any:
            keep[i] = False
    return keep


def hypervolume_monte_carlo(points_max: np.ndarray, n_samples: int, seed: int) -> float:
    """
    Monte Carlo estimate of hypervolume in [0,1]^3 w.r.t. reference point (0,0,0).
    HV = volume of union of axis-aligned boxes from origin to each point.
    """
    rng = np.random.default_rng(seed)
    samples = rng.random((n_samples, 3))
    covered = np.zeros(n_samples, dtype=bool)

    # Vectorized incremental OR over points
    for p in points_max:
        covered |= np.all(samples <= p, axis=1)

    return float(covered.mean())


def to_max_objectives(eff: np.ndarray, tox: np.ndarray, rri: np.ndarray) -> np.ndarray:
    """
    Convert to all-maximization objectives.
    Toxicity is minimized, so use tox_good = 1 - tox.
    """
    tox_good = 1.0 - tox
    return np.column_stack([eff, tox_good, rri])


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# -----------------------------
# Synthetic data generation
# -----------------------------

def generate_population(rng: np.random.Generator, n: int, kind: str, cfg: dict):
    """
    Returns (eff, tox, rri) arrays of length n in [0,1].
    cfg contains tunable means/stds and the size/quality of the "superior cluster".
    """
    if kind == "GA":
        eff = clip01(rng.normal(cfg["ga_eff_mu"], cfg["ga_eff_sd"], n))
        tox = clip01(rng.normal(cfg["ga_tox_mu"], cfg["ga_tox_sd"], n))
        rri = clip01(rng.normal(cfg["ga_rri_mu"], cfg["ga_rri_sd"], n))
        return eff, tox, rri

    if kind == "QIEA":
        eff = clip01(rng.normal(cfg["q_eff_mu"], cfg["q_eff_sd"], n))
        tox = clip01(rng.normal(cfg["q_tox_mu"], cfg["q_tox_sd"], n))
        rri = clip01(rng.normal(cfg["q_rri_mu"], cfg["q_rri_sd"], n))
        return eff, tox, rri

    if kind == "QIEA+RRI":
        top_k = int(cfg["qr_top_k"])
        base_n = n - top_k

        eff_base = clip01(rng.normal(cfg["qr_eff_mu"], cfg["qr_eff_sd"], base_n))
        tox_base = clip01(rng.normal(cfg["qr_tox_mu"], cfg["qr_tox_sd"], base_n))
        rri_base = clip01(rng.normal(cfg["qr_rri_mu"], cfg["qr_rri_sd"], base_n))

        # Superior cluster (must satisfy targets)
        eff_top = clip01(rng.normal(cfg["top_eff_mu"], cfg["top_eff_sd"], top_k))
        tox_top = clip01(rng.normal(cfg["top_tox_mu"], cfg["top_tox_sd"], top_k))
        rri_top = clip01(rng.normal(cfg["top_rri_mu"], cfg["top_rri_sd"], top_k))

        # Hard-clamp to ensure every "top" point is in the superior region
        eff_top = np.maximum(eff_top, 0.95)
        tox_top = np.minimum(tox_top, 0.20)
        rri_top = np.maximum(rri_top, 0.80)

        eff = np.concatenate([eff_base, eff_top])
        tox = np.concatenate([tox_base, tox_top])
        rri = np.concatenate([rri_base, rri_top])
        return eff, tox, rri

    raise ValueError(f"Unknown kind: {kind}")


# -----------------------------
# Auto-tune to hit targets
# -----------------------------

def compute_metrics(eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr, hv_samples=250_000):
    P_ga = to_max_objectives(eff_ga, tox_ga, rri_ga)
    P_q = to_max_objectives(eff_q, tox_q, rri_q)
    P_qr = to_max_objectives(eff_qr, tox_qr, rri_qr)

    nd_ga = P_ga[nondominated_mask(P_ga)]
    nd_q = P_q[nondominated_mask(P_q)]
    nd_qr = P_qr[nondominated_mask(P_qr)]

    hv_ga = hypervolume_monte_carlo(nd_ga, n_samples=hv_samples, seed=1)
    hv_q = hypervolume_monte_carlo(nd_q, n_samples=hv_samples, seed=2)
    hv_qr = hypervolume_monte_carlo(nd_qr, n_samples=hv_samples, seed=3)

    ratio_qr_ga = hv_qr / hv_ga if hv_ga > 0 else np.nan
    ratio_qr_q = hv_qr / hv_q if hv_q > 0 else np.nan

    superior = (eff_qr >= 0.95) & (tox_qr <= 0.20) & (rri_qr >= 0.80)
    n_superior = int(superior.sum())

    return {
        "hv_ga": hv_ga,
        "hv_q": hv_q,
        "hv_qr": hv_qr,
        "ratio_qr_ga": ratio_qr_ga,
        "ratio_qr_q": ratio_qr_q,
        "n_superior": n_superior,
    }


def meets_targets(m):
    return (
        1.20 <= m["ratio_qr_ga"] <= 1.30 and
        1.10 <= m["ratio_qr_q"] <= 1.15 and
        5 <= m["n_superior"] <= 10
    )


def autotune(seed=42, n_points=120, max_tries=60):
    rng = np.random.default_rng(seed)

    # Baseline config chosen to be realistic and close to targets.
    cfg = {
        # GA distribution (worse)
        "ga_eff_mu": 0.88, "ga_eff_sd": 0.05,
        "ga_tox_mu": 0.34, "ga_tox_sd": 0.10,
        "ga_rri_mu": 0.55, "ga_rri_sd": 0.12,

        # QIEA (better than GA)
        "q_eff_mu": 0.91, "q_eff_sd": 0.04,
        "q_tox_mu": 0.27, "q_tox_sd": 0.08,
        "q_rri_mu": 0.66, "q_rri_sd": 0.10,

        # QIEA+RRI base (better than QIEA)
        "qr_eff_mu": 0.93, "qr_eff_sd": 0.035,
        "qr_tox_mu": 0.23, "qr_tox_sd": 0.07,
        "qr_rri_mu": 0.74, "qr_rri_sd": 0.08,

        # Superior cluster controls (5–10 points)
        "qr_top_k": 8,
        "top_eff_mu": 0.965, "top_eff_sd": 0.010,
        "top_tox_mu": 0.155, "top_tox_sd": 0.020,
        "top_rri_mu": 0.86, "top_rri_sd": 0.030,
    }

    best = None

    for _ in range(max_tries):
        # Generate with a fresh RNG stream each try (deterministic across runs)
        eff_ga, tox_ga, rri_ga = generate_population(rng, n_points, "GA", cfg)
        eff_q, tox_q, rri_q = generate_population(rng, n_points, "QIEA", cfg)
        eff_qr, tox_qr, rri_qr = generate_population(rng, n_points, "QIEA+RRI", cfg)

        m = compute_metrics(eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr)

        # Keep the best attempt (closest to all target intervals) in case we never meet.
        def interval_distance(val, lo, hi):
            if val < lo:
                return lo - val
            if val > hi:
                return val - hi
            return 0.0

        score = (
            interval_distance(m["ratio_qr_ga"], 1.20, 1.30) +
            interval_distance(m["ratio_qr_q"], 1.10, 1.15) +
            interval_distance(m["n_superior"], 5, 10)
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "cfg": cfg.copy(),
                "metrics": m,
                "data": (eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr),
            }

        # If it meets targets, return immediately
        if meets_targets(m):
            return cfg, m, (eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr)

        # --- Auto-adjust rules (gentle and stable) ---
        # If QIEA+RRI vs GA HV is too low -> improve QIEA+RRI base a bit and/or worsen GA a bit.
        if m["ratio_qr_ga"] < 1.20:
            cfg["qr_eff_mu"] = min(cfg["qr_eff_mu"] + 0.005, 0.955)
            cfg["qr_tox_mu"] = max(cfg["qr_tox_mu"] - 0.005, 0.18)
            cfg["qr_rri_mu"] = min(cfg["qr_rri_mu"] + 0.010, 0.85)
            cfg["ga_tox_mu"] = min(cfg["ga_tox_mu"] + 0.005, 0.45)
            cfg["ga_rri_mu"] = max(cfg["ga_rri_mu"] - 0.005, 0.40)

        # If QIEA+RRI vs QIEA HV is too low -> slightly improve QIEA+RRI or slightly relax QIEA
        if m["ratio_qr_q"] < 1.10:
            cfg["qr_eff_mu"] = min(cfg["qr_eff_mu"] + 0.003, 0.955)
            cfg["qr_tox_mu"] = max(cfg["qr_tox_mu"] - 0.003, 0.18)
            cfg["qr_rri_mu"] = min(cfg["qr_rri_mu"] + 0.008, 0.85)
            cfg["q_rri_mu"] = max(cfg["q_rri_mu"] - 0.003, 0.58)

        # If ratios are too high, tone down QIEA+RRI slightly
        if m["ratio_qr_ga"] > 1.30 or m["ratio_qr_q"] > 1.15:
            cfg["qr_eff_mu"] = max(cfg["qr_eff_mu"] - 0.003, 0.90)
            cfg["qr_tox_mu"] = min(cfg["qr_tox_mu"] + 0.003, 0.30)
            cfg["qr_rri_mu"] = max(cfg["qr_rri_mu"] - 0.006, 0.65)

        # Keep superior points in range [5,10]
        if m["n_superior"] < 5:
            cfg["qr_top_k"] = min(cfg["qr_top_k"] + 1, 10)
        elif m["n_superior"] > 10:
            cfg["qr_top_k"] = max(cfg["qr_top_k"] - 1, 5)

    # Fall back: return best attempt found (still plots; no errors)
    return best["cfg"], best["metrics"], best["data"]


# -----------------------------
# Plotting (Figure 9)
# -----------------------------

def plot_figure_9(eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr, metrics):
    P_qr = to_max_objectives(eff_qr, tox_qr, rri_qr)
    mask_pf_qr = nondominated_mask(P_qr)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Four legend entries:
    ax.scatter(eff_ga, tox_ga, rri_ga, alpha=0.25, label="GA", s=18)
    ax.scatter(eff_q, tox_q, rri_q, alpha=0.25, label="QIEA", s=18)
    ax.scatter(eff_qr, tox_qr, rri_qr, alpha=0.35, label="QIEA+RRI", s=22)
    ax.scatter(
        eff_qr[mask_pf_qr], tox_qr[mask_pf_qr], rri_qr[mask_pf_qr],
        alpha=0.95, label="QIEA+RRI Pareto Front", s=60
    )

    ax.set_xlabel("Predicted Efficacy (maximize)")
    ax.set_ylabel("Predicted Toxicity (minimize)")
    ax.set_zlabel("RRI (maximize)")
    ax.set_title("3D Pareto Front Comparison (Synthetic Data)")

    ax.set_xlim(0.75, 1.00)
    ax.set_ylim(0.00, 0.55)
    ax.set_zlim(0.30, 1.00)

    ax.view_init(elev=22, azim=235)
    ax.legend(loc="upper left")

    superior = (eff_qr >= 0.95) & (tox_qr <= 0.20) & (rri_qr >= 0.80)
    n_superior = int(superior.sum())

    text = (
        "Hypervolume (MC) computed on transformed objectives:\n"
        "maximize efficacy, maximize (1 - toxicity), maximize RRI\n"
        f"HV(GA)={metrics['hv_ga']:.3f}  HV(QIEA)={metrics['hv_q']:.3f}  HV(QIEA+RRI)={metrics['hv_qr']:.3f}\n"
        f"HV ratio: QIEA+RRI/GA={metrics['ratio_qr_ga']:.3f} (target 1.20–1.30)\n"
        f"HV ratio: QIEA+RRI/QIEA={metrics['ratio_qr_q']:.3f} (target 1.10–1.15)\n"
        f"Superior points (eff≥0.95, tox≤0.20, rri≥0.80): {n_superior} (target 5–10)"
    )
    fig.text(0.02, 0.02, text, fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    cfg, metrics, data = autotune(seed=42, n_points=120, max_tries=60)
    eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr = data

    # Print final metrics (so you can confirm targets)
    print("=== Achieved Metrics ===")
    print(f"HV(GA)      : {metrics['hv_ga']:.6f}")
    print(f"HV(QIEA)    : {metrics['hv_q']:.6f}")
    print(f"HV(QIEA+RRI): {metrics['hv_qr']:.6f}")
    print(f"HV ratio QIEA+RRI/GA  : {metrics['ratio_qr_ga']:.6f} (target 1.20–1.30)")
    print(f"HV ratio QIEA+RRI/QIEA: {metrics['ratio_qr_q']:.6f} (target 1.10–1.15)")
    print(f"Superior points count : {metrics['n_superior']} (target 5–10)")
    print("\nConfig used (top cluster size etc.):")
    for k in ["qr_top_k", "qr_eff_mu", "qr_tox_mu", "qr_rri_mu", "ga_tox_mu", "ga_rri_mu", "q_rri_mu"]:
        print(f"  {k}: {cfg[k]}")

    plot_figure_9(eff_ga, tox_ga, rri_ga, eff_q, tox_q, rri_q, eff_qr, tox_qr, rri_qr, metrics)


if __name__ == "__main__":
    main()
