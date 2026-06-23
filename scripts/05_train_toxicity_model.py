#!/usr/bin/env python3
"""Train the XGBoost hemolytic toxicity classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qproteus.models import save_predictor, train_xgboost_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    predictor, report = train_xgboost_classifier(frame, random_state=args.random_state)
    save_predictor(predictor, args.model_out)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": report.metrics,
                "classification_report": report.classification_report_text,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote toxicity model to {args.model_out}")
    print(f"Wrote metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
