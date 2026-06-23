import math

from qproteus.qiea import entropy_bits_per_position, initialize_probability_matrix


def test_uniform_probability_matrix_entropy_matches_amino_acid_count():
    matrix = initialize_probability_matrix(5)
    entropy = entropy_bits_per_position(matrix)
    assert math.isclose(entropy, math.log2(20), rel_tol=1e-6)
