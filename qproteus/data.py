"""Data curation functions for public-source peptide datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import AMP_EXCLUSION_KEYWORDS, DEFAULT_FEATURE_COLUMNS
from .features import calculate_features, flag_extrapolations
from .sequence import (
    SequenceRecord,
    deduplicate_records,
    filter_records,
    normalize_sequence,
    read_fasta,
)


def records_to_frame(records: Iterable[SequenceRecord], *, label: int | None = None) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "record_id": record.record_id,
            "description": record.description,
            "sequence": normalize_sequence(record.sequence),
            "source": record.source,
        }
        if label is not None:
            row["label"] = int(label)
        row.update(calculate_features(row["sequence"]))
        rows.append(row)
    return pd.DataFrame(rows)


def read_sequence_csv(
    path: str | Path,
    *,
    source: str,
    sequence_column: str | None = None,
    id_column: str | None = None,
    description_column: str | None = None,
) -> list[SequenceRecord]:
    frame = pd.read_csv(path)
    lower_map = {column.lower(): column for column in frame.columns}
    seq_col = sequence_column or lower_map.get("sequence") or lower_map.get("seq")
    if seq_col is None:
        raise ValueError(f"Could not identify a sequence column in {path}")
    records: list[SequenceRecord] = []
    for idx, row in frame.iterrows():
        sequence = normalize_sequence(row[seq_col])
        if not sequence:
            continue
        record_id = str(row[id_column]) if id_column else str(idx)
        description = str(row[description_column]) if description_column else ""
        records.append(SequenceRecord(record_id, sequence, description, source))
    return records


def curate_amp_records(
    *,
    apd3_fasta: str | Path | None = None,
    dramp_natural_fasta: str | Path | None = None,
    dramp_gram_negative_fasta: str | Path | None = None,
    dramp_csv: str | Path | None = None,
    min_length: int = 10,
    max_length: int = 80,
) -> pd.DataFrame:
    """Curate APD3 and DRAMP Gram-negative AMP sequences."""
    records: list[SequenceRecord] = []
    if apd3_fasta:
        records.extend(read_fasta(apd3_fasta, source="APD3"))

    if dramp_natural_fasta and dramp_gram_negative_fasta:
        natural = read_fasta(dramp_natural_fasta, source="DRAMP")
        gram_negative = read_fasta(dramp_gram_negative_fasta, source="DRAMP")
        gram_negative_sequences = {normalize_sequence(record.sequence) for record in gram_negative}
        # DRAMP exports activity and natural-origin labels separately, so the
        # curated set is the sequence-level intersection of those two files.
        records.extend(
            record
            for record in natural
            if normalize_sequence(record.sequence) in gram_negative_sequences
        )

    if dramp_csv:
        records.extend(read_sequence_csv(dramp_csv, source="DRAMP"))

    clean = filter_records(
        deduplicate_records(records),
        min_length=min_length,
        max_length=max_length,
        standard_only=True,
    )
    frame = records_to_frame(clean, label=1)
    if not frame.empty:
        frame = frame.sort_values(["source", "record_id", "sequence"]).reset_index(drop=True)
    return frame


def curate_uniprot_controls(
    fasta_path: str | Path,
    *,
    min_length: int = 10,
    max_length: int = 55,
    exclusion_keywords: Iterable[str] = AMP_EXCLUSION_KEYWORDS,
) -> pd.DataFrame:
    """Curate reviewed non-AMP controls from a UniProt FASTA export."""
    keywords = tuple(keyword.lower() for keyword in exclusion_keywords)
    records = read_fasta(fasta_path, source="UniProt")
    filtered: list[SequenceRecord] = []
    for record in records:
        haystack = f"{record.record_id} {record.description}".lower()
        if any(keyword in haystack for keyword in keywords):
            continue
        filtered.append(record)
    clean = filter_records(
        deduplicate_records(filtered),
        min_length=min_length,
        max_length=max_length,
        standard_only=True,
    )
    frame = records_to_frame(clean, label=0)
    if not frame.empty:
        frame = frame.sort_values(["record_id", "sequence"]).reset_index(drop=True)
    return frame


def build_training_dataset(
    amp_frame: pd.DataFrame,
    non_amp_frame: pd.DataFrame,
    *,
    balance: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Combine AMP and non-AMP frames, optionally matching class counts."""
    amp = amp_frame.copy()
    non_amp = non_amp_frame.copy()
    if balance and len(non_amp) > len(amp):
        non_amp = non_amp.sample(n=len(amp), random_state=random_state)
    combined = pd.concat([amp, non_amp], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"], keep="first")
    combined, _stats = flag_extrapolations(combined, feature_columns=DEFAULT_FEATURE_COLUMNS)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def prepare_toxicity_frame(
    input_csv: str | Path,
    *,
    sequence_column: str = "sequence",
    activity_column: str | None = None,
    label_column: str | None = None,
    toxic_threshold: float | None = None,
) -> pd.DataFrame:
    """Prepare DBAASP hemolysis records for toxicity classification."""
    raw = pd.read_csv(input_csv)
    if sequence_column not in raw.columns:
        lower_map = {column.lower(): column for column in raw.columns}
        sequence_column = lower_map.get(sequence_column.lower(), sequence_column)
    if sequence_column not in raw.columns:
        raise ValueError(f"Missing sequence column: {sequence_column}")

    rows = []
    for _, row in raw.iterrows():
        sequence = normalize_sequence(row[sequence_column])
        try:
            features = calculate_features(sequence)
        except ValueError:
            continue

        output = {
            "sequence": sequence,
            **features,
        }
        if label_column and label_column in raw.columns:
            output["label"] = int(row[label_column])
        elif activity_column and activity_column in raw.columns:
            # Some DBAASP exports provide continuous hemolysis measurements
            # instead of labels; binarize them below after all rows are read.
            activity = pd.to_numeric(row[activity_column], errors="coerce")
            if pd.isna(activity):
                continue
            output["hemolytic_activity"] = float(activity)
        else:
            raise ValueError("Provide either label_column or activity_column")
        rows.append(output)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    if "label" not in frame.columns:
        threshold = toxic_threshold
        if threshold is None:
            threshold = float(frame["hemolytic_activity"].median())
        frame["label"] = (frame["hemolytic_activity"] > threshold).astype(int)
        frame["toxic_threshold"] = threshold
    return frame
