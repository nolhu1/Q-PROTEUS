import numpy as np
from calculate_RRI import calculate_rri

# -------------------
# Test Runner
# -------------------
def test_rri():
    peptides = {
        "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "Magainin2": "GIGKFLHSAKKFGKAFVGEIMNS",
        "CecropinA": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "ColistinFrag": "CLCRRWQWRMKKLG",
        "ControlAAAA": "AAAAAAAAAAAAAA"
    }

    expected = {
        "LL-37": {"binary": 0.6, "continuous": 0.7},
        "Magainin2": {"binary": 0.55, "continuous": 0.65},
        "CecropinA": {"binary": 0.7, "continuous": 0.75},
        "ColistinFrag": {"binary": 0.8, "continuous": 0.85},
        "ControlAAAA": {"binary": 0.0, "continuous": 0.1}
    }

    results = {}
    analysis = []

    for name, seq in peptides.items():
        res = calculate_rri(seq, n_mutants=50)  # reduce for speed
        results[name] = res

        # Calculate errors
        b_err = abs(res["binary_rri"] - expected[name]["binary"])
        c_err = abs(res["continuous_rri"] - expected[name]["continuous"])

        analysis.append({
            "peptide": name,
            "expected_binary": expected[name]["binary"],
            "calc_binary": res["binary_rri"],
            "binary_error": b_err,
            "expected_cont": expected[name]["continuous"],
            "calc_cont": res["continuous_rri"],
            "cont_error": c_err
        })

        # Print per peptide
        print(f"\n{name}:")
        print(f"  Parent prob       = {res['parent_prob']:.3f}")
        print(f"  Binary RRI        = {res['binary_rri']:.3f} (expected ~{expected[name]['binary']}) | error = {b_err:.3f}")
        print(f"  Continuous RRI    = {res['continuous_rri']:.3f} (expected ~{expected[name]['continuous']}) | error = {c_err:.3f}")

    # -------------------
    # Final Accuracy Analysis
    # -------------------
    all_b_err = [a["binary_error"] for a in analysis]
    all_c_err = [a["cont_error"] for a in analysis]

    mean_b_err = np.mean(all_b_err)
    mean_c_err = np.mean(all_c_err)
    overall_mean_err = np.mean(all_b_err + all_c_err)

    print("\n--- Accuracy Analysis ---")
    print(f"Mean Binary RRI error:      {mean_b_err:.3f}")
    print(f"Mean Continuous RRI error:  {mean_c_err:.3f}")
    print(f"Overall mean absolute error: {overall_mean_err:.3f}")

if __name__ == "__main__":
    test_rri()
