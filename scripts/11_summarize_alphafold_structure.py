#!/usr/bin/env python3
"""Summarize AlphaFold PDB confidence and sequence-level structural features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.features import calculate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean_plddt_from_pdb(path: str | Path) -> float:
    values = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        raise ValueError(f"No ATOM pLDDT/B-factor values found in {path}")
    return float(pd.Series(values).mean())


def main() -> None:
    args = parse_args()
    features = calculate_features(args.sequence)
    summary = pd.DataFrame(
        [
            {
                "sequence": args.sequence,
                "mean_plddt": mean_plddt_from_pdb(args.pdb),
                "helix_fraction": features["helix_fraction"],
                "sheet_fraction": features["sheet_fraction"],
                "net_charge": features["net_charge"],
                "fraction_hydrophobic": features["fraction_hydrophobic"],
            }
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Wrote structural summary to {output}")


if __name__ == "__main__":
    main()
