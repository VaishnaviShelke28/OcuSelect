import os
import json
import time
import uuid
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, send_from_directory

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT, "models")
DATA_DIR = os.path.join(PROJECT, "data")

# Results are stored OUTSIDE the static folder so they are not public
RESULT_DIR = os.path.join(PROJECT, "tmp_results")

MAX_BATCH_ROWS = 1000
SIMILARITY_MIN = 0.40

os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


print("\n======================================")
print("          OcuSelect")
print("======================================")
print("Loading trained models...")


activity_model = joblib.load(
    os.path.join(MODEL_DIR, "TGFBR1_HistGradientBoosting.pkl")
)

corneal_model = joblib.load(
    os.path.join(MODEL_DIR, "corneal_permeability_extratrees.pkl")
)

reference_df = pd.read_csv(
    os.path.join(DATA_DIR, "TGFBR1_CHEMBL4439_pIC50_clean.csv")
)

corneal_df = pd.read_csv(
    os.path.join(DATA_DIR, "corneal_permeability_clean.csv")
)

corneal_features_df = pd.read_csv(
    os.path.join(DATA_DIR, "corneal_ml_features.csv")
)

activity_median = float(reference_df["pIC50"].median())
permeability_median = float(corneal_df["logPapp"].median())

# Range of the corneal training compounds (used for reliability check)
CORNEAL_RANGE = {
    name: (
        float(corneal_features_df[name].min()),
        float(corneal_features_df[name].max())
    )
    for name in ["MW", "LogP", "TPSA"]
}

# Corneal model scores. The retraining script writes this file.
# If it does not exist yet, the older values are used.
CORNEAL_METRICS = {"r2": 0.564, "mae": 0.329, "rmse": 0.437, "n": 69}
_metrics_path = os.path.join(MODEL_DIR, "corneal_metrics.json")
if os.path.exists(_metrics_path):
    with open(_metrics_path) as fh:
        CORNEAL_METRICS.update(json.load(fh))


morgan = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

reference_fps = []
reference_indices = []

for i, smi in enumerate(reference_df["Canonical_SMILES"]):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        reference_fps.append(morgan.GetFingerprint(mol))
        reference_indices.append(i)


FEATURES = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "FractionCSP3"
]


largest_fragment = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
tautomer_enumerator.SetMaxTautomers(200)


def standardize_mol(mol):
    """Keep the largest fragment (removes salts) and neutralize charges."""
    mol = largest_fragment.choose(mol)
    mol = uncharger.uncharge(mol)
    return mol


def cleanup_old_results(max_age_seconds=3600):
    """Delete result files older than one hour."""
    now = time.time()
    for name in os.listdir(RESULT_DIR):
        path = os.path.join(RESULT_DIR, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            try:
                os.remove(path)
            except OSError:
                pass


def molecule_svg(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(380, 260)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace(
        "<?xml version='1.0' encoding='iso-8859-1'?>",
        ""
    )
    return svg


def analyze_compound(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        mol = standardize_mol(mol)
    except Exception:
        return None

    # Same tautomer form as used for the corneal training data
    try:
        corneal_mol = tautomer_enumerator.Canonicalize(mol)
    except Exception:
        corneal_mol = mol

    canonical = Chem.MolToSmiles(mol, canonical=True)

    fp = morgan.GetFingerprint(mol)

    fp_array = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    fp_array = fp_array.reshape(1, -1)

    predicted_pic50 = float(activity_model.predict(fp_array)[0])

    descriptor_values = {
        "MW": Descriptors.MolWt(corneal_mol),
        "LogP": Descriptors.MolLogP(corneal_mol),
        "TPSA": Descriptors.TPSA(corneal_mol),
        "HBD": Lipinski.NumHDonors(corneal_mol),
        "HBA": Lipinski.NumHAcceptors(corneal_mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(corneal_mol),
        "RingCount": Lipinski.RingCount(corneal_mol),
        "FractionCSP3": Descriptors.FractionCSP3(corneal_mol)
    }

    descriptor_df = pd.DataFrame([descriptor_values], columns=FEATURES)
    predicted_logpapp = float(corneal_model.predict(descriptor_df)[0])

    similarities = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    local_idx = int(np.argmax(similarities))
    best_idx = reference_indices[local_idx]
    similarity = float(similarities[local_idx])
    nearest = reference_df.iloc[best_idx]

    # ---------- reliability checks ----------
    activity_reliable = similarity >= SIMILARITY_MIN

    out_of_range = [
        name for name in ["MW", "LogP", "TPSA"]
        if not (CORNEAL_RANGE[name][0]
                <= descriptor_values[name]
                <= CORNEAL_RANGE[name][1])
    ]
    permeability_reliable = len(out_of_range) == 0

    warnings = []

    if not activity_reliable:
        warnings.append(
            f"Low similarity to the TGFBR1 training compounds "
            f"(Tanimoto {similarity:.2f}, below {SIMILARITY_MIN:.2f}). "
            f"Treat the activity prediction as low confidence."
        )

    if not permeability_reliable:
        warnings.append(
            "Outside the corneal training range for: "
            + ", ".join(out_of_range)
            + ". The permeability prediction is an extrapolation "
              "and is low confidence."
        )

    # ---------- positions and quadrant ----------
    # Numbers are always shown. Low-confidence results are labelled.
    activity_high = predicted_pic50 >= activity_median
    permeability_high = predicted_logpapp >= permeability_median

    activity_position = (
        "Above dataset median" if activity_high
        else "Below dataset median"
    )
    permeability_position = (
        "Above dataset median" if permeability_high
        else "Below dataset median"
    )

    if not activity_reliable:
        activity_position += " (low confidence)"
    if not permeability_reliable:
        permeability_position += " (low confidence)"

    if activity_high and permeability_high:
        quadrant = "Higher activity / Higher permeability"
    elif activity_high:
        quadrant = "Higher activity / Lower permeability"
    elif permeability_high:
        quadrant = "Lower activity / Higher permeability"
    else:
        quadrant = "Lower activity / Lower permeability"

    if not (activity_reliable and permeability_reliable):
        quadrant += " (low confidence)"

    if similarity >= 0.70:
        similarity_context = "High similarity"
    elif similarity >= 0.40:
        similarity_context = "Moderate similarity"
    else:
        similarity_context = "Low similarity"

    nearest_smiles = str(nearest["Canonical_SMILES"])
    nearest_mol = Chem.MolFromSmiles(nearest_smiles)

    return {
        "canonical_smiles": canonical,

        "pic50": round(predicted_pic50, 3),
        "logpapp": round(predicted_logpapp, 3),

        "activity_position": activity_position,
        "permeability_position": permeability_position,
        "quadrant": quadrant,

        "reliability": (
            "In range" if (activity_reliable and permeability_reliable)
            else "Low confidence"
        ),
        "warnings": warnings,

        "similarity": round(similarity, 3),
        "similarity_context": similarity_context,

        "nearest_chembl": str(nearest["Molecule_ChEMBL_ID"]),
        "nearest_pic50": round(float(nearest["pIC50"]), 3),
        "nearest_smiles": nearest_smiles,

        "mw": round(descriptor_values["MW"], 2),
        "logp": round(descriptor_values["LogP"], 2),
        "tpsa": round(descriptor_values["TPSA"], 2),
        "hbd": descriptor_values["HBD"],
        "hba": descriptor_values["HBA"],
        "rotatable": descriptor_values["RotatableBonds"],
        "rings": descriptor_values["RingCount"],
        "fraction_csp3": round(descriptor_values["FractionCSP3"], 3),

        "query_svg": molecule_svg(mol),
        "nearest_svg": (
            molecule_svg(nearest_mol)
            if nearest_mol else ""
        )
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    result = None
    error = None
    submitted_smiles = ""

    if request.method == "POST":

        submitted_smiles = request.form.get("smiles", "").strip()

        if not submitted_smiles:
            error = "Please enter a SMILES string."
        else:
            try:
                result = analyze_compound(submitted_smiles)
            except Exception:
                result = None

            if result is None:
                error = (
                    "Invalid SMILES. Please check the "
                    "molecular structure."
                )

    return render_template(
        "analyze.html",
        result=result,
        error=error,
        submitted_smiles=submitted_smiles
    )


def run_batch(df, smiles_col, compound_col):
    """Analyze every row. Returns (web_rows, full_rows, number_analyzed)."""

    web_results = []
    full_results = []
    n_ok = 0

    for idx, row in df.iterrows():

        name = (
            str(row[compound_col])
            if compound_col is not None and pd.notna(row[compound_col])
            else f"Compound_{idx + 1}"
        )

        raw = row[smiles_col]
        smiles = "" if pd.isna(raw) else str(raw).strip()

        analysis = None
        if smiles:
            try:
                analysis = analyze_compound(smiles)
            except Exception:
                analysis = None

        if analysis is None:

            web_results.append({
                "compound": name,
                "status": "Invalid / Missing SMILES",
                "reliability": "—",
                "pic50": "—",
                "logpapp": "—",
                "similarity": "—",
                "quadrant": "—"
            })

            full_results.append({
                "Compound": name,
                "Input SMILES": smiles,
                "Status": "Invalid / Missing SMILES"
            })

            continue

        n_ok += 1

        web_results.append({
            "compound": name,
            "status": "Analyzed",
            "reliability": analysis["reliability"],
            "pic50": analysis["pic50"] if analysis["pic50"] is not None else "—",
            "logpapp": analysis["logpapp"] if analysis["logpapp"] is not None else "—",
            "similarity": analysis["similarity"],
            "quadrant": analysis["quadrant"]
        })

        full_results.append({
            "Compound": name,
            "Input SMILES": smiles,
            "Status": "Analyzed",
            "Reliability": analysis["reliability"],
            "Warnings": " | ".join(analysis["warnings"]),
            "Canonical SMILES": analysis["canonical_smiles"],
            "Predicted TGFBR1 pIC50": analysis["pic50"],
            "Activity position": analysis["activity_position"],
            "Predicted corneal logPapp": analysis["logpapp"],
            "Permeability position": analysis["permeability_position"],
            "OcuSelect quadrant": analysis["quadrant"],
            "Tanimoto similarity": analysis["similarity"],
            "Similarity context": analysis["similarity_context"],
            "Nearest ChEMBL compound": analysis["nearest_chembl"],
            "Nearest experimental pIC50": analysis["nearest_pic50"],
            "MW": analysis["mw"],
            "LogP": analysis["logp"],
            "TPSA": analysis["tpsa"],
            "HBD": analysis["hbd"],
            "HBA": analysis["hba"],
            "Rotatable Bonds": analysis["rotatable"],
            "Ring Count": analysis["rings"],
            "Fraction CSP3": analysis["fraction_csp3"],
            "Nearest compound SMILES": analysis["nearest_smiles"]
        })

    return web_results, full_results, n_ok


@app.route("/batch", methods=["GET", "POST"])
def batch():

    results = None
    error = None
    download_file = None

    if request.method == "POST":

        cleanup_old_results()

        uploaded = request.files.get("csv_file")

        if uploaded is None or uploaded.filename == "":
            error = "Please choose a CSV file."

        elif not uploaded.filename.lower().endswith(".csv"):
            error = "Please upload a .csv file."

        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception:
                df = None
                error = "Could not read this file. Please upload a valid CSV."

            if df is not None:

                smiles_col = next(
                    (c for c in df.columns
                     if str(c).strip().lower() == "smiles"),
                    None
                )

                compound_col = next(
                    (c for c in df.columns
                     if str(c).strip().lower() == "compound"),
                    None
                )

                if smiles_col is None:
                    error = "CSV must contain a column named SMILES."

                elif len(df) == 0:
                    error = "The CSV file has no rows."

                elif len(df) > MAX_BATCH_ROWS:
                    error = (
                        f"Too many rows ({len(df)}). "
                        f"Maximum allowed is {MAX_BATCH_ROWS}."
                    )

                else:
                    web_results, full_results, n_ok = run_batch(
                        df, smiles_col, compound_col
                    )

                    if n_ok == 0:
                        error = (
                            "No valid SMILES were found in the SMILES "
                            "column. Nothing was analyzed."
                        )
                    else:
                        filename = (
                            f"OcuSelect_results_"
                            f"{uuid.uuid4().hex[:8]}.csv"
                        )

                        pd.DataFrame(full_results).to_csv(
                            os.path.join(RESULT_DIR, filename),
                            index=False
                        )

                        results = web_results
                        download_file = filename

    return render_template(
        "batch.html",
        results=results,
        error=error,
        download_file=download_file
    )


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        RESULT_DIR,
        filename,
        as_attachment=True
    )


@app.route("/model-info")
def model_info():
    return render_template(
        "model_info.html",
        activity_median=round(activity_median, 3),
        permeability_median=round(permeability_median, 3),
        corneal_r2=round(float(CORNEAL_METRICS["r2"]), 3),
        corneal_mae=round(float(CORNEAL_METRICS["mae"]), 3),
        corneal_rmse=round(float(CORNEAL_METRICS["rmse"]), 3),
        corneal_n=int(CORNEAL_METRICS["n"]),
        similarity_min=SIMILARITY_MIN
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    print("Models loaded successfully.")
    print(f"TGFBR1 reference compounds: {len(reference_df)}")
    print(f"Activity median: {activity_median:.3f}")
    print(f"Permeability median: {permeability_median:.3f}")
    print("Starting OcuSelect...")

    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )