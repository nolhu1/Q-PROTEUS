import math

from qproteus.rri import blosum62_probabilities, minmax_normalize, weighted_mutation_table


def test_blosum62_probabilities_normalize():
    probabilities = blosum62_probabilities("A")
    assert len(probabilities) == 19
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert "A" not in probabilities


def test_minmax_normalize_handles_constant_values():
    values = minmax_normalize([0.5, 0.5, 0.5])
    assert values.tolist() == [0.0, 0.0, 0.0]


def test_weighted_mutation_table_has_one_weight_per_mutant():
    rows = weighted_mutation_table("ACD")
    assert len(rows) == 57
    assert math.isclose(sum(weight for _mutation, weight in rows), 1.0)
