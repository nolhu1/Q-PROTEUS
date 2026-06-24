#!/usr/bin/env python3
"""Run GA, QIEA, or QIEA+RRI peptide optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.evaluators import SequenceEvaluator
from qproteus.ga import GAConfig, run_ga
from qproteus.qiea import QIEAConfig, run_qiea


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ga", "qiea", "qiea-rri"), required=True)
    parser.add_argument("--efficacy-model")
    parser.add_argument("--toxicity-model", required=True)
    parser.add_argument("--mic-predictor")
    parser.add_argument("--embedding-model")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rri-max-mutants", type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def result_frames(result, algorithm: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.DataFrame(result.history)
    history["algorithm"] = algorithm
    history["seed"] = seed
    population = pd.DataFrame(
        {
            "sequence": result.population,
            "fitness": result.fitnesses,
            "algorithm": algorithm,
            "seed": seed,
        }
    )
    return history, population


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = SequenceEvaluator.from_specs(
        efficacy_model=args.efficacy_model,
        toxicity_model=args.toxicity_model,
        mic_predictor=args.mic_predictor,
        embedding_model=args.embedding_model,
        rri_max_mutants=args.rri_max_mutants,
    )

    for seed in args.seeds:
        if args.algorithm == "ga":
            config = GAConfig(
                length=args.length,
                population_size=args.population_size,
                generations=args.generations,
                seed=seed,
            )
            result = run_ga(config, evaluator.ga_baseline_fitness)
        else:
            config = QIEAConfig(
                length=args.length,
                population_size=args.population_size,
                generations=args.generations,
                seed=seed,
            )
            include_rri = args.algorithm == "qiea-rri"
            result = run_qiea(config, lambda seq: evaluator.fitness(seq, include_rri=include_rri))

        history, population = result_frames(result, args.algorithm, seed)
        history.to_csv(output_dir / f"{args.algorithm}_seed{seed}_history.csv", index=False)
        population.to_csv(output_dir / f"{args.algorithm}_seed{seed}_population.csv", index=False)
        print(f"Finished {args.algorithm} seed {seed}")


if __name__ == "__main__":
    main()
