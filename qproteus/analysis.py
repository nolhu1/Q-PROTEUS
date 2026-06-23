"""Run analysis and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

from .metrics import first_generation_to_fraction


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    if "best_fitness" not in history.columns:
        raise ValueError("history must contain best_fitness")
    best = history["best_fitness"].to_numpy(dtype=float)
    return {
        "final_best_fitness": float(best[-1]),
        "max_best_fitness": float(best.max()),
        "t90": float(first_generation_to_fraction(best, fraction=0.9)),
        "fitness_auc": float(np.trapz(best, dx=1.0)),
    }


def compare_final_fitness(group_a: Iterable[float], group_b: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    test = ttest_ind(a, b, equal_var=False)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "sd_a": float(a.std(ddof=1)),
        "sd_b": float(b.std(ddof=1)),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def resistance_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    fold_change_column: str = "MIC_fold_change",
) -> pd.DataFrame:
    working = frame.copy()
    working[fold_change_column] = pd.to_numeric(working[fold_change_column], errors="coerce")
    working = working.dropna(subset=[fold_change_column])
    summary = (
        working.groupby(group_columns, dropna=False)
        .agg(
            n_lines=(fold_change_column, "count"),
            median_mic_fold_change=(fold_change_column, "median"),
            mean_mic_fold_change=(fold_change_column, "mean"),
        )
        .reset_index()
    )
    summary["log10_median_mic_fold_change"] = np.log10(summary["median_mic_fold_change"])
    summary["log10_mean_mic_fold_change"] = np.log10(summary["mean_mic_fold_change"])
    return summary


def rri_correlation(
    frame: pd.DataFrame,
    *,
    rri_column: str = "rri",
    resistance_column: str = "log10_median_mic_fold_change",
) -> dict[str, float]:
    clean = frame[[rri_column, resistance_column]].dropna()
    pearson = pearsonr(clean[rri_column], clean[resistance_column])
    spearman = spearmanr(clean[rri_column], clean[resistance_column])
    return {
        "n": int(len(clean)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def read_histories(run_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(run_dir).rglob("*history.csv"):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No history CSV files found under {run_dir}")
    return pd.concat(frames, ignore_index=True)
