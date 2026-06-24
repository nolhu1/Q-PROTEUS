# Q-PROTEUS

**Durability by Design: A Quantum-Inspired Computational Framework for Engineering Antibiotic Resistance Resilience in Novel Antimicrobial Peptides**

Q-PROTEUS is a computational framework for designing antimicrobial peptides (AMPs) with explicit pressure for resistance resilience that includes a novel metric Resistance Resilience Index (RRI), paired with a Quantum-Inspired Evolutionary Algorithm (QIEA) for evolutionary optimization.

The repository is organized around the methods of this project and includes source code and used datasets.

## Overview

Antimicrobial peptide design workflows often optimize predicted potency and toxicity, but do not directly measure how robust a peptide remains under mutation. Q-PROTEUS adds a durability objective through the Resistance Resilience Index, which estimates how well a peptide preserves activity and physicochemical structure across its single-point mutational neighborhood.

Procedure followed:

1. Curate Gram-negative AMP sequences from APD3 and DRAMP.
2. Build matched non-AMP controls from reviewed UniProt peptide sequences.
3. Prepare hemolysis data for toxicity prediction.
4. Train efficacy and toxicity prediction models.
5. Use a mutation-sensitive MIC predictor for graded activity estimates.
6. Compute RRI from functional, structural, and physicochemical robustness.
7. Compare GA, QIEA, and QIEA+RRI optimization.
8. Analyze convergence, diversity, Pareto fronts, novelty, structure, and mutational robustness.

## Repository Structure

```text
Q-PROTEUS/
|-- data/                       # Source, training, and validation datasets
|-- qproteus/                   # Reusable Python package
|   |-- analysis.py             # Run summaries and validation statistics
|   |-- data.py                 # Dataset curation helpers
|   |-- evaluators.py           # Sequence objective evaluation
|   |-- features.py             # Peptide descriptor extraction
|   |-- ga.py                   # Genetic algorithm baseline
|   |-- metrics.py              # Pareto, diversity, and convergence metrics
|   |-- models.py               # Model training and predictor adapters
|   |-- qiea.py                 # Quantum-inspired evolutionary algorithm
|   |-- rri.py                  # Resistance Resilience Index implementation
|   `-- sequence.py             # FASTA parsing and mutation utilities
|-- scripts/                    # Numbered reproduction scripts
|-- tests/                      # Unit tests for core methods
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Data

The `data/` directory contains the project-relevant datasets needed to reproduce the computational workflow:

| File | Purpose |
| --- | --- |
| `amp_vs_non_amp_training_dataset.csv` | Training data for the AMP efficacy classifier |
| `complete_curated_combined_gramnegative_amps.csv` | Curated APD3/DRAMP Gram-negative AMP dataset |
| `curated_dramp_natural_gramnegative_amps.csv` | Curated DRAMP natural Gram-negative subset |
| `apd3.fasta` | APD3 source FASTA export |
| `natural_amps.fasta` | DRAMP natural AMP source FASTA |
| `Anti-Gram-_amps.fasta` | DRAMP Gram-negative activity source FASTA |
| `uniprotkb_length_10_TO_55_AND_reviewed_2025_09_03.fasta` | Reviewed UniProt non-AMP control source export |
| `toxicity_data.csv` | DBAASP-derived hemolysis/toxicity records |
| `hemolytic_data.csv` | Cleaned hemolytic peptide labels |
| `hemolytic.fasta`, `non_hemolytic.fasta` | Toxicity sequence FASTA files |
| `dramp_training_dataset.csv` | MIC regression training resource |
| `dramp_mic_expanded.xlsx` | DRAMP MIC source table |
| `DRAMP_general_amps.xlsx` | DRAMP general AMP source table |
| `rri_validation_set.csv` | Experimental resistance-evolution validation records |
| `scaled_rri_validation.csv` | Aggregated/log-scaled RRI validation summary |
| `peptides.csv` | DBAASP peptide source table |

Trained models and generated outputs are not committed. They should be regenerated locally.

## Installation

Use Python 3.10 or newer.

```bash
git clone <repo-url>
cd Q-PROTEUS

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## External MIC Predictor

The RRI calculation requires a mutation-sensitive MIC predictor that returns continuous `log10(MIC uM)` values for peptide sequences. The original project used AMPGen for this step: https://github.com/xiyanxiongnico/AMPGen

AMPGen model weights are not included in this repository. Q-PROTEUS provides a generic predictor interface, so any compatible object can be used if it exposes:

```python
predictor.predict(["PEPTIDESEQ", "MUTANTSEQ"])
```

and returns one numeric MIC prediction per input sequence.

Pass the predictor to scripts with:

```bash
--mic-predictor path/to/mic_predictor.joblib
```

or with a Python import spec:

```bash
--mic-predictor module_name:object_or_class_name
```

## Quick Start
```

Train the efficacy model from the included training dataset:

```bash
python scripts/04_train_efficacy_model.py \
  --input data/amp_vs_non_amp_training_dataset.csv \
  --model-out models/efficacy_predictor.joblib \
  --metrics-out outputs/efficacy_metrics.json
```

Prepare toxicity labels and train the toxicity model:

```bash
python scripts/03_prepare_toxicity_data.py \
  --input data/hemolytic_data.csv \
  --sequence-column Sequence \
  --label-column Hemolytic_Label \
  --output outputs/reproduced/toxicity_training.csv

python scripts/05_train_toxicity_model.py \
  --input outputs/reproduced/toxicity_training.csv \
  --model-out models/toxicity_predictor.joblib \
  --metrics-out outputs/toxicity_metrics.json
```
## Method Summary

### Feature Extraction

Peptide sequences are represented using physicochemical descriptors:

- sequence length
- net charge at pH 7.4
- GRAVY hydrophobicity
- positive, negative, and hydrophobic residue fractions
- instability index
- molecular weight
- predicted helix and sheet fractions
- charge density

Feature extrapolation is flagged with a `mean +/- 2.0 * sd` rule computed on the training feature distribution.

### Predictive Models

The efficacy model is a Random Forest classifier trained to distinguish Gram-negative AMPs from matched non-AMP peptide controls. The toxicity model is an XGBoost classifier trained from hemolysis data.

For RRI, a MIC predictor is used as a continuous activity model so that point mutations produce graded potency changes rather than binary active/inactive labels.

### Resistance Resilience Index

For a peptide of length `L`, Q-PROTEUS enumerates all `19L` single-point substitutions. Mutation probabilities are derived from BLOSUM62 using a temperature-scaled softmax with `beta = 0.5`. Cysteine-involving substitutions are down-weighted by `0.25` and probabilities are renormalized.

RRI combines three terms:

- functional robustness: retention of predicted MIC activity across mutants
- structural robustness: embedding-space stability across mutants
- physicochemical robustness: stability of charge and hydrophobic moment

The final score is multiplicative:

```text
RRI = R_f * R_s * R_pc
```

### Optimization

Q-PROTEUS compares three search strategies:

- GA: standard Genetic Algorithm baseline
- QIEA: Quantum-Inspired Evolutionary Algorithm without RRI
- QIEA+RRI: QIEA with resistance resilience in the objective

Default optimization settings:

- population size: `200`
- generations: `500`
- independent seeds: `5`
- QIEA probability matrix: `length x 20`
- QIEA top-update fraction: `20%`
- QIEA learning rate theta: `0.1`
- entropy threshold: `2.0` bits per position
- perturbation: `10%` uniform mixing after 10 low-entropy generations

The QIEA+RRI fitness objective is:

```text
F(x) = E(x)^0.4 * (1 - T(x))^0.3 * RRI(x)^0.3
```

where `E(x)` is predicted efficacy and `T(x)` is predicted toxicity.

## Outputs

Generated outputs are written under ignored directories:

```text
models/       # trained predictors
outputs/      # metrics, run histories, validation tables
figures/      # generated figures
visuals/      # optional visual summaries
```

## Citation

If you use this repository, cite the project as:

```text
Huang, N. Q-PROTEUS: A Quantum-Inspired Computational Framework for Engineering Antibiotic Resistance Resilience in Novel Antimicrobial Peptides.
```
