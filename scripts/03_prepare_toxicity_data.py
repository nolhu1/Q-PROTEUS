#!/usr/bin/env python3
"""Prepare DBAASP hemolysis data for toxicity classification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.data import prepare_toxicity_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--activity-column")
    parser.add_argument("--label-column")
    parser.add_argument("--toxic-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = prepare_toxicity_frame(
        args.input,
        sequence_column=args.sequence_column,
        activity_column=args.activity_column,
        label_column=args.label_column,
        toxic_threshold=args.toxic_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} toxicity rows to {output}")


if __name__ == "__main__":
    main()
