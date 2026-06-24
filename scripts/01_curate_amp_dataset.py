#!/usr/bin/env python3
"""Curate APD3 and DRAMP Gram-negative AMP sequences."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.data import curate_amp_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apd3-fasta")
    parser.add_argument("--dramp-natural-fasta")
    parser.add_argument("--dramp-gram-negative-fasta")
    parser.add_argument("--dramp-csv")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = curate_amp_records(
        apd3_fasta=args.apd3_fasta,
        dramp_natural_fasta=args.dramp_natural_fasta,
        dramp_gram_negative_fasta=args.dramp_gram_negative_fasta,
        dramp_csv=args.dramp_csv,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} curated AMP sequences to {output}")


if __name__ == "__main__":
    main()
