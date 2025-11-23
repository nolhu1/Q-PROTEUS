import random
import string
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AMPlify.src.utils import predict_amplify, load_amplify_models

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

import random

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

def generate_single_point_mutant(sequence: str) -> str:
    """Biologically realistic single-AA mutant via one nucleotide substitution."""
    if len(sequence) == 0:
        return sequence
    
    # choose residue position
    idx = random.randint(0, len(sequence) - 1)
    aa = sequence[idx]

    # choose one of its codons
    codon = random.choice(CODONS_FOR_AA[aa])

    # mutate codon until it produces a different valid AA
    while True:
        pos_nt = random.randint(0, 2)
        orig_nt = codon[pos_nt]
        new_nt = random.choice([n for n in NUCLEOTIDES if n != orig_nt])

        mutated_codon = codon[:pos_nt] + new_nt + codon[pos_nt+1:]
        new_aa = GENETIC_CODE.get(mutated_codon, None)

        # skip invalid, synonymous, or stop mutations
        if new_aa is None:
            continue
        if new_aa == aa:
            continue
        if new_aa == "*":
            continue
        
        break

    # return mutated sequence (same return type as original function)
    return sequence[:idx] + new_aa + sequence[idx+1:]



# def generate_single_point_mutant(sequence: str) -> str:
#     """Generate a single-point mutant by changing one residue."""
#     if len(sequence) == 0:
#         return sequence
    
#     seq_list = list(sequence)
#     idx = random.randint(0, len(seq_list) - 1)
    
#     # choose a different amino acid
#     original_aa = seq_list[idx]
#     choices = [aa for aa in AMINO_ACIDS if aa != original_aa]
#     seq_list[idx] = random.choice(choices)
    
#     return "".join(seq_list)


def calculate_rri_mut(sequence: str, models, N: int = 100) -> float:
    """
    Compute continuous Phase 1 RRI using mutational robustness.

    Continuous definition:
        RRI = mean( predicted_activity_probability(mutant_i) )
    
    Steps:
      - generate N single-point mutants
      - run AMPlify on all mutants
      - return the mean predicted probability
    """
    # generate mutants
    mutants = [generate_single_point_mutant(sequence) for _ in range(100)]
    print(mutants)
    # predict probabilities with AMPlify
    predictions = predict_amplify(mutants, models)  # should return list/array of floats in [0, 1]
    print(predictions)
    # continuous RRI = mean probability
    rri = float(sum(predictions) / len(predictions))

    return rri



def calculate_rri(sequence: str, N: int = 100, threshold: float = 0.5) -> float:
    """Wrapper for Phase 1 RRI (mutational only)."""
    return calculate_rri_mut(sequence, models, N=N)
models = load_amplify_models()
print(calculate_rri("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", models))
