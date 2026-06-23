#!/usr/bin/env python3
"""Build the AMP vs non-AMP efficacy training table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.data import build_training_dataset, curate_uniprot_controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-csv", required=True)
    parser.add_argument("--uniprot-fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=55)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amp_frame = pd.read_csv(args.amp_csv)
    non_amp_frame = curate_uniprot_controls(
        args.uniprot_fasta,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    training = build_training_dataset(
        amp_frame,
        non_amp_frame,
        balance=not args.no_balance,
        random_state=args.random_state,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(output, index=False)
    print(f"Wrote {len(training)} training rows to {output}")


if __name__ == "__main__":
    main()
