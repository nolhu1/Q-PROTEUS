#!/usr/bin/env python3
"""Run BLASTp novelty assessment for candidate peptides."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--blast-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blastp", default="blastp")
    parser.add_argument("--evalue", default="1e-3")
    parser.add_argument("--max-target-seqs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_csv)
    with tempfile.TemporaryDirectory() as tmpdir:
        query = Path(tmpdir) / "candidates.fasta"
        with query.open("w", encoding="utf-8") as handle:
            for idx, sequence in enumerate(candidates[args.sequence_column]):
                handle.write(f">candidate_{idx}\n{sequence}\n")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.blastp,
            "-query",
            str(query),
            "-db",
            args.blast_db,
            "-out",
            str(output),
            "-outfmt",
            "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-evalue",
            args.evalue,
            "-max_target_seqs",
            str(args.max_target_seqs),
        ]
        subprocess.run(command, check=True)
    print(f"Wrote BLASTp tabular output to {args.output}")


if __name__ == "__main__":
    main()
