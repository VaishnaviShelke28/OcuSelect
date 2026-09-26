# OcuSelect

**Live demo:** https://ocuselect.onrender.com

Computational profiling of compounds using two models:

1. TGFBR1 (ALK5) activity — predicted pIC50 (ChEMBL CHEMBL4439)
2. Corneal permeability — predicted logPapp (QsarDB 2010PR1398)

Compounds that are not similar to the training data, or fall outside the
corneal training range, are flagged as "Low confidence."

## Run locally

    pip install -r requirements.txt
    python webapp/app.py

Open http://127.0.0.1:5000

## Folders
- data/ — cleaned datasets
- models/ — trained models
- notebooks/ — cleaning and retraining code
- webapp/ — Flask app

## Note
For academic research only. Not for clinical decisions.
