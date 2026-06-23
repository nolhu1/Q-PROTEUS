"""Sequence parsing, validation, and mutation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable, Iterator

from .constants import AMINO_ACIDS, AMINO_ACID_SET


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    sequence: str
    description: str = ""
    source: str = ""


@dataclass(frozen=True)
class Mutation:
    position: int
    original: str
    replacement: str
    sequence: str


def normalize_sequence(sequence: str) -> str:
    """Normalize sequence strings without guessing missing residues."""
    return re.sub(r"\s+", "", str(sequence).upper()).replace("-", "")


def is_standard_sequence(sequence: str) -> bool:
    seq = normalize_sequence(sequence)
    return bool(seq) and all(residue in AMINO_ACID_SET for residue in seq)


def require_standard_sequence(sequence: str) -> str:
    seq = normalize_sequence(sequence)
    if not is_standard_sequence(seq):
        raise ValueError(f"Sequence contains non-standard amino acids: {sequence!r}")
    return seq


def read_fasta(path: str | Path, source: str = "") -> list[SequenceRecord]:
    """Read FASTA records using a small tolerant parser."""
    records: list[SequenceRecord] = []
    current_header: str | None = None
    current_sequence: list[str] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    def flush() -> None:
        if current_header is None:
            return
        sequence = normalize_sequence("".join(current_sequence))
        record_id = current_header.split()[0] if current_header else ""
        records.append(
            SequenceRecord(
                record_id=record_id,
                sequence=sequence,
                description=current_header,
                source=source,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_header = line[1:].strip()
            current_sequence = []
        else:
            current_sequence.append(line)
    flush()
    return records


def deduplicate_records(records: Iterable[SequenceRecord]) -> list[SequenceRecord]:
    """Keep the first record for each exact normalized sequence."""
    seen: set[str] = set()
    unique: list[SequenceRecord] = []
    for record in records:
        sequence = normalize_sequence(record.sequence)
        if sequence in seen:
            continue
        seen.add(sequence)
        unique.append(
            SequenceRecord(
                record_id=record.record_id,
                sequence=sequence,
                description=record.description,
                source=record.source,
            )
        )
    return unique


def filter_records(
    records: Iterable[SequenceRecord],
    *,
    min_length: int = 1,
    max_length: int | None = None,
    standard_only: bool = True,
) -> list[SequenceRecord]:
    filtered: list[SequenceRecord] = []
    for record in records:
        sequence = normalize_sequence(record.sequence)
        if standard_only and not is_standard_sequence(sequence):
            continue
        if len(sequence) < min_length:
            continue
        if max_length is not None and len(sequence) > max_length:
            continue
        filtered.append(
            SequenceRecord(
                record_id=record.record_id,
                sequence=sequence,
                description=record.description,
                source=record.source,
            )
        )
    return filtered


def all_single_point_mutants(sequence: str) -> Iterator[Mutation]:
    seq = require_standard_sequence(sequence)
    for idx, original in enumerate(seq):
        for replacement in AMINO_ACIDS:
            if replacement == original:
                continue
            mutated = seq[:idx] + replacement + seq[idx + 1 :]
            yield Mutation(
                position=idx + 1,
                original=original,
                replacement=replacement,
                sequence=mutated,
            )


def random_single_point_mutants(
    sequence: str,
    n: int,
    *,
    seed: int | None = None,
) -> list[Mutation]:
    seq = require_standard_sequence(sequence)
    rng = random.Random(seed)
    mutants: list[Mutation] = []
    for _ in range(n):
        idx = rng.randrange(len(seq))
        original = seq[idx]
        replacement = rng.choice([aa for aa in AMINO_ACIDS if aa != original])
        mutants.append(
            Mutation(
                position=idx + 1,
                original=original,
                replacement=replacement,
                sequence=seq[:idx] + replacement + seq[idx + 1 :],
            )
        )
    return mutants
