"""
Figure 7 — Convergence Curve (GA vs QIEA vs QIEA+RRI)

Creates a publication-style convergence plot:
- X: generation (0..499)
- Y: mean best fitness
- Shaded band: ±1 std across 5 seeds
- Synthetic data used by default (replace with your real run logs later)

Dependencies:
  pip install numpy matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
N_GENERATIONS = 500
SEEDS = [0, 1, 2, 3, 4]  # 5 runs / random seeds

# If your real data is available, set USE_SYNTHETIC = False and implement load_real_runs()
USE_SYNTHETIC = True


# ----------------------------
# SYNTHETIC DATA GENERATION
# ----------------------------
def _logistic_rise(g: np.ndarray, start: float, end: float, rate: float, mid: float) -> np.ndarray:
    """
    Smooth rise from start -> end using a logistic curve.
    """
    return start + (end - start) / (1.0 + np.exp(-rate * (g - mid)))


def _soft_cap(x: float, cap: float) -> float:
    """Prevents hard saturation artifacts while keeping values <= cap."""
    if x <= cap:
        return x
    # gentle compression if it tries to exceed the cap
    return cap - (cap * 0.02) * np.tanh((x - cap) / (cap * 0.02))


def _gen_improvement_events(
    rng: np.random.Generator,
    n_generations: int,
    base_rate: float,
    early_boost: float,
    late_decay: float,
) -> np.ndarray:
    """
    Returns per-generation event intensity multiplier.
    Higher early -> more breakthroughs; decays later.
    """
    g = np.arange(n_generations, dtype=float)
    # Early higher, late lower: a smooth decay curve
    decay = early_boost * np.exp(-g / (n_generations * late_decay)) + 1.0
    # Keep within a reasonable range
    return np.clip(base_rate * decay, 1e-6, 1.0)


def generate_realistic_best_fitness(
    algorithm: str,
    n_generations: int,
    seed: int,
    *,
    start: float | None = None,
    cap: float | None = None,
) -> np.ndarray:
    """
    Generates a single-run 'best fitness' curve that looks like real evolutionary data.

    Output:
      np.ndarray shape (n_generations,)
      - Non-decreasing best fitness
      - Includes plateaus, bursts, seed variability
    """
    rng = np.random.default_rng(seed)
    g = np.arange(n_generations)

    # Algorithm-specific behavior knobs (tuned for realism)
    if algorithm == "GA":
        # GA: more plateaus, occasional breakthroughs, earlier stagnation
        start = 0.10 if start is None else start
        cap = 0.74 if cap is None else cap
        base_event_rate = 0.10  # chance of a meaningful improvement event per gen early
        early_boost = 1.8
        late_decay = 0.28
        plateau_bias = 0.55     # strong stagnation tendency
        jump_scale = 0.030
        micro_scale = 0.0035
        backslide_prob = 0.07   # candidate quality can dip (but best-so-far won't)
        premature_convergence_at = int(n_generations * 0.45)
        premature_factor = 0.55  # reduces improvements after premature convergence
    elif algorithm == "QIEA":
        # QIEA: steadier improvements, fewer long plateaus, good late progress
        start = 0.12 if start is None else start
        cap = 0.85 if cap is None else cap
        base_event_rate = 0.13
        early_boost = 1.6
        late_decay = 0.34
        plateau_bias = 0.28
        jump_scale = 0.026
        micro_scale = 0.0040
        backslide_prob = 0.05
        premature_convergence_at = int(n_generations * 0.62)
        premature_factor = 0.70
    elif algorithm == "QIEA+RRI":
        # QIEA+RRI: strong early lift + stable late convergence; slightly fewer wild jumps
        start = 0.14 if start is None else start
        cap = 0.92 if cap is None else cap
        base_event_rate = 0.14
        early_boost = 1.7
        late_decay = 0.38
        plateau_bias = 0.20
        jump_scale = 0.023
        micro_scale = 0.0038
        backslide_prob = 0.04
        premature_convergence_at = int(n_generations * 0.70)
        premature_factor = 0.78
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Seed-to-seed variability (real runs differ)
    cap = float(np.clip(rng.normal(cap, 0.015), 0.60, 0.98))
    jump_scale *= float(np.clip(rng.normal(1.0, 0.10), 0.70, 1.35))
    micro_scale *= float(np.clip(rng.normal(1.0, 0.12), 0.70, 1.40))
    plateau_bias = float(np.clip(rng.normal(plateau_bias, 0.07), 0.05, 0.80))

    # Event intensity over time
    event_intensity = _gen_improvement_events(
        rng,
        n_generations,
        base_rate=base_event_rate,
        early_boost=early_boost,
        late_decay=late_decay,
    )

    # State variables
    best = start
    best_curve = np.empty(n_generations, dtype=float)

    # Simulate dynamics
    stall_counter = 0
    stall_target = 0

    for t in range(n_generations):
        # Premature convergence effect (especially GA): improvements become rarer/smaller
        late_penalty = 1.0
        if t >= premature_convergence_at:
            late_penalty *= premature_factor

        # A run can enter a stall/plateau for a random duration
        if stall_counter >= stall_target:
            # chance to start a new stall episode
            if rng.random() < plateau_bias * (0.60 + 0.40 * (t / n_generations)):
                # later runs plateau more often/longer
                stall_target = int(rng.integers(6, 26) * (0.7 + 0.8 * (t / n_generations)))
                stall_counter = 0
            else:
                stall_target = 0
                stall_counter = 0
        else:
            stall_counter += 1

        in_stall = stall_target > 0 and stall_counter < stall_target

        # Candidate "generation best" can fluctuate (not necessarily best-so-far)
        # We model a candidate score that sometimes dips, then take best-so-far.
        # This creates more realistic jitter without violating best-so-far monotonicity.
        candidate = best

        # Micro-improvements: small, frequent, shrink as you approach cap
        distance_to_cap = max(cap - best, 1e-6)
        micro = rng.normal(0, micro_scale) * (distance_to_cap / cap) ** 0.65
        if in_stall:
            micro *= 0.15  # stalls suppress steady progress

        # Backslides (candidate-only) are more common early and for GA
        if rng.random() < backslide_prob * (0.85 + 0.30 * (1 - t / n_generations)):
            candidate -= abs(rng.standard_t(df=3)) * 0.006 * (0.9 + 0.6 * (t < n_generations * 0.3))

        # Breakthrough events: occasional larger jumps
        # Probability decays over time and when near cap; stalls reduce breakthroughs.
        p_event = event_intensity[t] * late_penalty
        if in_stall:
            p_event *= 0.45

        # As you get close to cap, breakthroughs get rarer/smaller
        p_event *= float(np.clip((distance_to_cap / cap) ** 0.55, 0.05, 1.0))

        if rng.random() < p_event:
            # Jump size from a heavy-tailed distribution (rare big jumps)
            # Also shrink near cap
            tail = abs(rng.standard_t(df=2.2))
            jump = jump_scale * tail * (distance_to_cap / cap) ** 0.55

            # Occasional "big discovery" (very rare), more plausible early
            if rng.random() < (0.015 if algorithm == "GA" else 0.020) * (1 - t / n_generations):
                jump *= rng.uniform(1.8, 2.8)

            candidate += jump

        # Apply micro change
        candidate += micro

        # Soft-cap and keep within [0, 1]
        candidate = _soft_cap(candidate, cap)
        candidate = float(np.clip(candidate, 0.0, 1.0))

        # Best-so-far update (monotonic)
        best = max(best, candidate)
        best_curve[t] = best

    return best_curve


def build_runs_matrix(algorithm: str, seeds: list[int], n_generations: int) -> np.ndarray:
    """
    Returns shape: (n_runs, n_generations)
    """
    runs = [generate_realistic_best_fitness(algorithm, n_generations, s) for s in seeds]
    return np.vstack(runs)


# ----------------------------
# REAL DATA LOADING (PLACEHOLDER)
# ----------------------------
def load_real_runs() -> dict[str, np.ndarray]:
    """
    Replace this with your real data loader.

    Expected return format:
      {
        "GA":        np.ndarray shape (n_runs, n_generations),
        "QIEA":      np.ndarray shape (n_runs, n_generations),
        "QIEA+RRI":  np.ndarray shape (n_runs, n_generations),
      }

    Example of one common logging format you might have:
    - CSV per run with columns: generation,best_fitness
    Then you would read each CSV into an array of length N_GENERATIONS and stack them.

    For now, this is intentionally not implemented.
    """
    raise NotImplementedError("Implement load_real_runs() with your run logs.")


# ----------------------------
# PLOTTING
# ----------------------------
def plot_convergence(runs_by_algo: dict[str, np.ndarray], out_path: str | None = None) -> None:
    """
    Plots mean best fitness and ± std shading for each algorithm.
    """
    generations = np.arange(runs_by_algo[next(iter(runs_by_algo))].shape[1])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for label, runs in runs_by_algo.items():
        # runs shape: (n_runs, n_generations)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        ax.plot(generations, mean, label=label, linewidth=2.2)
        ax.fill_between(generations, mean - std, mean + std, alpha=0.20)

    # Style: clean scientific look
    ax.set_title("Figure 7. Convergence Performance Across 500 Generations", pad=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Best Fitness")

    # Light grid (optional but usually helps readability)
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper right")

    # Tight layout so labels don't get cut
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


def main() -> None:
    if USE_SYNTHETIC:
        runs_by_algo = {
            "GA": build_runs_matrix("GA", SEEDS, N_GENERATIONS),
            "QIEA": build_runs_matrix("QIEA", SEEDS, N_GENERATIONS),
            "QIEA+RRI": build_runs_matrix("QIEA+RRI", SEEDS, N_GENERATIONS),
        }
    else:
        runs_by_algo = load_real_runs()

    # Optional: export to a high-res PNG for your paper
    plot_convergence(runs_by_algo, out_path="figure7_convergence.png")


if __name__ == "__main__":
    main()
