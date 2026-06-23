#!/usr/bin/env python3
"""Summarize optimization run histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.analysis import summarize_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in Path(args.run_dir).rglob("*history.csv"):
        history = pd.read_csv(path)
        summary = summarize_history(history)
        summary["source_file"] = str(path)
        if "algorithm" in history.columns:
            summary["algorithm"] = str(history["algorithm"].iloc[0])
        if "seed" in history.columns:
            summary["seed"] = int(history["seed"].iloc[0])
        summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No history files found under {args.run_dir}")
    summary_frame = pd.DataFrame(summaries)
    grouped = (
        summary_frame.groupby("algorithm")
        .agg(["mean", "std"])
        .drop(columns=[("seed", "mean"), ("seed", "std")], errors="ignore")
    )
    output = {
        "runs": summaries,
        "by_algorithm": json.loads(grouped.to_json()),
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {summary_path}")


if __name__ == "__main__":
    main()
