# OcuSelect

Computational profiling of compounds using two models:

1. TGFBR1 (ALK5) activity, predicted pIC50 (ChEMBL CHEMBL4439)
2. Corneal permeability, predicted logPapp (QsarDB 2010PR1398)

Compounds that are not similar to the training data, or fall outside the
corneal training range, are flagged as "Outside model range" and no
misleading number is shown.

## Run

    pip install -r requirements.txt
    python webapp/app.py

Then open http://127.0.0.1:5000

## Folders

- data/       cleaned datasets (raw downloads in data/raw)
- models/     trained models
- notebooks/  cleaning and retraining code
- webapp/     Flask app
- results/    sample output
- tmp_results/ temporary batch outputs (auto-deleted after 1 hour)

## Note

For academic research only. Not for clinical decisions.
