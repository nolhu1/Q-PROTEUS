import random
import string
import os, sys
import random
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from AMPGen.AMP_discriminator.Discriminator_model.discriminator import classify_sequences_list
from scripts.mic_regression.MIC_scorer import MIC_scorer


def calculate_rri_binary(
    sequence: str,
    N: int = 100
) -> float:

    # Generate mutants
    mutants = generate_point_mutant_list(sequence, N)

    # Classify mutants (0 = non-AMP, 1 = AMP)
    preds = classify_sequences_list(mutants, None, "models/discriminator.json")

    # Safety check
    if len(preds) != N:
        raise ValueError("Classifier output length mismatch")

    # RRI = fraction of active mutants
    rri = sum(preds) / N
    return rri

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Standard genetic code
GENETIC_CODE = {
    "ATA":"I", "ATC":"I", "ATT":"I", "ATG":"M",
    "ACA":"T", "ACC":"T", "ACG":"T", "ACT":"T",
    "AAC":"N", "AAT":"N", "AAA":"K", "AAG":"K",
    "AGC":"S", "AGT":"S", "AGA":"R", "AGG":"R",
    "CTA":"L", "CTC":"L", "CTG":"L", "CTT":"L",
    "CCA":"P", "CCC":"P", "CCG":"P", "CCT":"P",
    "CAC":"H", "CAT":"H", "CAA":"Q", "CAG":"Q",
    "CGA":"R", "CGC":"R", "CGG":"R", "CGT":"R",
    "GTA":"V", "GTC":"V", "GTG":"V", "GTT":"V",
    "GCA":"A", "GCC":"A", "GCG":"A", "GCT":"A",
    "GAC":"D", "GAT":"D", "GAA":"E", "GAG":"E",
    "GGA":"G", "GGC":"G", "GGG":"G", "GGT":"G",
    "TCA":"S", "TCC":"S", "TCG":"S", "TCT":"S",
    "TTC":"F", "TTT":"F", "TTA":"L", "TTG":"L",
    "TAC":"Y", "TAT":"Y", "TAA":"*", "TAG":"*",
    "TGC":"C", "TGT":"C", "TGA":"*", "TGG":"W"
}

# Precompute AA → codons map
CODONS_FOR_AA = {}
for codon, aa in GENETIC_CODE.items():
    CODONS_FOR_AA.setdefault(aa, []).append(codon)

NUCLEOTIDES = ["A", "C", "G", "T"]

import random
import math

def generate_point_mutant_list(sequence: str, N: int) -> list:
    """
    Generates N biologically realistic single-AA mutants.
    Returns a list of mutated sequences.
    """
    mutants = []
    for _ in range(N):
        if len(sequence) == 0:
            mutants.append(sequence)
            continue

        idx = random.randint(0, len(sequence) - 1)
        aa = sequence[idx]

        codon = random.choice(CODONS_FOR_AA[aa])

        # mutate codon until valid non-synonymous AA
        while True:
            pos_nt = random.randint(0, 2)
            orig_nt = codon[pos_nt]
            new_nt = random.choice([n for n in NUCLEOTIDES if n != orig_nt])

            mutated_codon = codon[:pos_nt] + new_nt + codon[pos_nt+1:]
            new_aa = GENETIC_CODE.get(mutated_codon, None)

            if new_aa is None:
                continue
            if new_aa == aa:
                continue
            if new_aa == "*":
                continue

            break

        mutant = sequence[:idx] + new_aa + sequence[idx+1:]
        mutants.append(mutant)

    return mutants



def calculate_rri_mic(
    sequence: str,
    model,
    N: int = 100,
    T: float = 1.0  # activity threshold (e.g., log10 MIC ≤ 1 means ≤ 10 µM)
):
    """
    Resistance Resilience Index (RRI):
    Quantifies the ability of an AMP to retain biological activity across realistic mutants.

    - sequence: wild-type peptide
    - model.predict(): must take a list[str] and return list[float] of log10(MIC µM)
    - N: number of mutants to sample
    - T: log10(MIC µM) threshold for acceptable antimicrobial activity
    """

    # Generate mutants
    mutants = generate_point_mutant_list(sequence, N)

    # Predict their MICs
    mutant_mics = model.predict(mutants)
    print(mutant_mics)
    penalties = []
    for m_mic in mutant_mics:
        # Penalize mutants only if they exceed activity threshold
        delta = m_mic - T

        # If delta <= 0 → fully acceptable (penalty = 1)
        # If delta > 0  → exponential penalty
        p = math.exp(-max(delta, 0))

        penalties.append(p)

    # RRI = mean robustness across mutants
    return sum(penalties) / N



# mic_predictor = MIC_scorer()
# print(calculate_rri_mic("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", mic_predictor))

# ---------------------------------------------------------
sequences = {
    "Colistin":      "FAKKLAVYRAPRKAFK",
    "LL-37":         "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin-2":    "GIGKFLHSAKKFGKAFVGEIMNS",
    "Random_low":    "AAAAAAAVVVVVV",
    "Random_high":   "GKKRRKGGRKKR"
}

results = [calculate_rri_binary(seq) for seq in sequences.values()]

# Print results
for name, res in zip(sequences.keys(), results):
    print(f"=== {name} ===")
    print(f" Pred rri: {res}")
