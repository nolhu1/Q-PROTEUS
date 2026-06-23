#!/usr/bin/env python3
"""Calculate RRI values for resistance-evolution validation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.analysis import resistance_summary, rri_correlation
from qproteus.models import DescriptorEmbeddingProvider, load_predictor
from qproteus.rri import calculate_rri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--mic-predictor", required=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--fold-change-column", default="MIC_fold_change")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-out")
    parser.add_argument("--max-mutants", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.validation_csv)
    mic_predictor = load_predictor(args.mic_predictor)
    embedding_provider = (
        load_predictor(args.embedding_model) if args.embedding_model else DescriptorEmbeddingProvider()
    )

    group_columns = [args.name_column, args.sequence_column]
    summary = resistance_summary(
        frame,
        group_columns=group_columns,
        fold_change_column=args.fold_change_column,
    )
    rri_rows = []
    for _, row in summary.iterrows():
        components = calculate_rri(
            row[args.sequence_column],
            mic_predictor,
            embedding_provider,
            max_mutants=args.max_mutants,
        )
        rri_rows.append(
            {
                args.name_column: row[args.name_column],
                args.sequence_column: row[args.sequence_column],
                "rri": components.rri,
                "rri_functional": components.functional,
                "rri_structural": components.structural,
                "rri_physicochemical": components.physicochemical,
                "n_mutants": components.n_mutants,
            }
        )
    scored = summary.merge(pd.DataFrame(rri_rows), on=group_columns, how="left")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    metrics = rri_correlation(scored)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote RRI validation table to {output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
