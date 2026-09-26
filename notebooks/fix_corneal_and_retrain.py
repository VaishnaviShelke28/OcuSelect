"""
Fixes the corneal training data (same standardization the app uses),
recalculates descriptors, re-checks accuracy and saves the model.

Run from the main project folder (PROJECT_SM):
    python notebooks/fix_corneal_and_retrain.py

Your old files are copied to *.bak before anything is overwritten.
"""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURES = ["MW", "LogP", "TPSA", "HBD", "HBA",
            "RotatableBonds", "RingCount", "FractionCSP3"]

FILES_TO_BACKUP = [
    "data/corneal_permeability_clean.csv",
    "data/corneal_ml_features.csv",
    "models/corneal_permeability_extratrees.pkl",
    "models/corneal_features.pkl",
    "plots/corneal_actual_vs_predicted.png",
]
for f in FILES_TO_BACKUP:
    try:
        shutil.copy(f, f + ".bak")
    except FileNotFoundError:
        pass

largest = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()
tautomer = rdMolStandardize.TautomerEnumerator()
tautomer.SetMaxTautomers(200)


def prep(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = largest.choose(mol)
    mol = uncharger.uncharge(mol)
    try:
        mol = tautomer.Canonicalize(mol)
    except Exception:
        pass
    return mol


df = pd.read_csv("data/corneal_permeability_clean.csv")

new_smiles, rows = [], []
for smi in df["SMILES"]:
    mol = prep(smi)
    new_smiles.append(Chem.MolToSmiles(mol))
    rows.append({
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
    })

changed = sum(a != b for a, b in zip(df["SMILES"], new_smiles))
print(f"SMILES rewritten by standardization: {changed} of {len(df)}")

df["SMILES"] = new_smiles
df.to_csv("data/corneal_permeability_clean.csv", index=False)

feat = pd.concat([df, pd.DataFrame(rows)], axis=1)
feat.to_csv("data/corneal_ml_features.csv", index=False)

X = feat[FEATURES]
y = feat["logPapp"]


def make_model():
    return ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42
    )


# Honest score: 5-fold cross-validation repeated with 10 different splits
r2s, maes, rmses = [], [], []
for seed in range(10):
    pred = cross_val_predict(
        make_model(), X, y,
        cv=KFold(5, shuffle=True, random_state=seed)
    )
    r2s.append(r2_score(y, pred))
    maes.append(mean_absolute_error(y, pred))
    rmses.append(float(np.sqrt(mean_squared_error(y, pred))))

print(f"R2   mean {np.mean(r2s):.3f}  (min {min(r2s):.3f}, max {max(r2s):.3f})")
print(f"MAE  mean {np.mean(maes):.3f}")
print(f"RMSE mean {np.mean(rmses):.3f}")

with open("models/corneal_metrics.json", "w") as fh:
    json.dump({
        "r2": float(np.mean(r2s)),
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "n": int(len(feat)),
    }, fh, indent=2)

# Plot (one split)
pred = cross_val_predict(
    make_model(), X, y, cv=KFold(5, shuffle=True, random_state=42)
)
plt.figure(figsize=(6, 6))
plt.scatter(y, pred, alpha=0.7)
lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Experimental logPapp")
plt.ylabel("Out-of-fold predicted logPapp")
plt.title("Corneal permeability model")
plt.tight_layout()
plt.savefig("plots/corneal_actual_vs_predicted.png", dpi=300)

# Train on all data and save
final = make_model().fit(X, y)
joblib.dump(final, "models/corneal_permeability_extratrees.pkl")
joblib.dump(FEATURES, "models/corneal_features.pkl")
print("Saved model and metrics.")
