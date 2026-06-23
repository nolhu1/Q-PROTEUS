#!/usr/bin/env python3
"""Run random single-point mutational scans for candidate peptides."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.evaluators import SequenceEvaluator
from qproteus.sequence import random_single_point_mutants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--toxicity-model", required=True)
    parser.add_argument("--efficacy-model")
    parser.add_argument("--mic-predictor")
    parser.add_argument("--embedding-model")
    parser.add_argument("--mutants-per-sequence", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_csv)
    evaluator = SequenceEvaluator.from_specs(
        efficacy_model=args.efficacy_model,
        toxicity_model=args.toxicity_model,
        mic_predictor=args.mic_predictor,
        embedding_model=args.embedding_model,
    )
    rows = []
    for candidate_index, row in candidates.iterrows():
        sequence = row[args.sequence_column]
        wild_fitness = evaluator.fitness(sequence, include_rri=False)
        mutants = random_single_point_mutants(
            sequence,
            args.mutants_per_sequence,
            seed=args.seed + int(candidate_index),
        )
        for mutation in mutants:
            mutant_fitness = evaluator.fitness(mutation.sequence, include_rri=False)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "wildtype_sequence": sequence,
                    "mutant_sequence": mutation.sequence,
                    "position": mutation.position,
                    "original": mutation.original,
                    "replacement": mutation.replacement,
                    "wildtype_fitness": wild_fitness,
                    "mutant_fitness": mutant_fitness,
                    "fitness_retention": mutant_fitness / max(wild_fitness, 1e-12),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote mutational scan to {output}")


if __name__ == "__main__":
    main()
