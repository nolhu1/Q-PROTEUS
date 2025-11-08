import random

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def sample_mutants(sequence: str, N: int = 100, seed: int = 42) -> list:
    """
    Generate N single-point mutants for a given peptide sequence.
    
    Returns a list of tuples: (position, original_aa, new_aa, mutant_sequence)
    """
    random.seed(seed)
    mutants = []
    seq_list = list(sequence)
    length = len(seq_list)
    
    for _ in range(N):
        pos = random.randint(0, length - 1)
        orig_aa = seq_list[pos]
        new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != orig_aa])
        mutant_seq = seq_list.copy()
        mutant_seq[pos] = new_aa
        mutants.append((pos, orig_aa, new_aa, ''.join(mutant_seq)))
    
    return mutants
