from qproteus.sequence import all_single_point_mutants, is_standard_sequence, normalize_sequence


def test_normalize_sequence_removes_spacing_and_hyphens():
    assert normalize_sequence(" acd-ef \n") == "ACDEF"


def test_standard_sequence_rejects_unknown_residue():
    assert is_standard_sequence("ACDE")
    assert not is_standard_sequence("ACDX")


def test_all_single_point_mutants_has_19_per_position():
    mutants = list(all_single_point_mutants("ACD"))
    assert len(mutants) == 57
    assert all(mutant.sequence != "ACD" for mutant in mutants)
