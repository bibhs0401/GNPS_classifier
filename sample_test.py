"""
GNPS Pipeline (NO XGBOOST VERSION)

MGF -> Sample -> NPClassifier -> Balance -> Binned Features -> Multiple Models

Models included:
- RandomForest
- Logistic Regression
- Linear SVM
- RBF SVM
- KNN
- Gradient Boosting
"""

import time
import random
import requests
import numpy as np
import pandas as pd

from matchms.importing import load_from_mgf
from rdkit import Chem, RDLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


# config
MGF_FILE = "C:/Users/bibhushaojha/Desktop/RA work/ALL_GNPS.mgf"

SAMPLE_SIZE = 10000
LABEL_LEVEL = "pathway"
MIN_CLASS_COUNT = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
API_SLEEP = 0.1

MAX_MZ = 1000
BIN_SIZE = 2

RDLogger.DisableLog("rdApp.*")


# logger
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def log_progress(i, total, start_time, prefix=""):
    elapsed = time.time() - start_time
    speed = i / elapsed if elapsed > 0 else 0
    remaining = (total - i) / speed if speed > 0 else 0

    print(f"[{time.strftime('%H:%M:%S')}] {prefix} "
          f"{i}/{total} | {100*i/total:.1f}% | "
          f"{speed:.2f} it/s | ETA {remaining/60:.1f} min")

INVALID_STRINGS = {"", "na", "n/a", "none", "null", "nan"}

def is_invalid_text(x):
    if x is None:
        return True
    return str(x).strip().lower() in INVALID_STRINGS


def get_metadata_value(metadata, keys):
    for k in keys:
        if k in metadata and not is_invalid_text(metadata[k]):
            return str(metadata[k]).strip()
    return None


def canonicalize_smiles(smiles):
    if is_invalid_text(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None


def inchi_to_smiles(inchi):
    if is_invalid_text(inchi):
        return None
    if not str(inchi).startswith("InChI="):
        return None
    try:
        mol = Chem.MolFromInchi(str(inchi))
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None


def extract_smiles(spec):
    meta = spec.metadata or {}
    s = canonicalize_smiles(get_metadata_value(meta, ["smiles", "SMILES"]))
    if s: return s
    return inchi_to_smiles(get_metadata_value(meta, ["inchi", "InChI"]))


# np classifer
def query_npclassifier(smiles):
    try:
        r = requests.get(
            "https://npclassifier.gnps2.org/classify",
            params={"smiles": smiles},
            timeout=20
        )
        return r.json()
    except:
        return None


def get_label(res):
    if res is None:
        return None
    vals = res.get("pathway_results", [])
    return vals[0] if vals else None


# features
def spectrum_to_binned_features(spec):
    bins = np.zeros(MAX_MZ // BIN_SIZE)

    mz = spec.peaks.mz
    inten = spec.peaks.intensities

    if mz is None or len(mz) == 0:
        return bins

    mz = np.array(mz)
    inten = np.array(inten)

    if np.max(inten) > 0:
        inten = inten / np.max(inten)

    for m, i in zip(mz, inten):
        if m < MAX_MZ:
            bins[int(m // BIN_SIZE)] += i

    return bins


# load + sample
def load_and_sample():
    log("Loading MGF...")
    spectra = list(load_from_mgf(MGF_FILE))
    log(f"Total spectra: {len(spectra)}")

    idx = random.sample(range(len(spectra)), min(SAMPLE_SIZE, len(spectra)))
    sampled = [spectra[i] for i in idx]

    return spectra, sampled, idx


# labeling
def label_sample(sampled, indices):
    log("Starting labeling...")
    start = time.time()

    cache = {}
    rows = []

    for i, (idx, spec) in enumerate(zip(indices, sampled), start=1):

        if i % 50 == 0:
            log_progress(i, len(sampled), start, "Label")

        smiles = extract_smiles(spec)

        if not smiles:
            rows.append({"local": i-1, "label": None})
            continue

        if smiles in cache:
            label = cache[smiles]
        else:
            res = query_npclassifier(smiles)
            label = get_label(res)
            cache[smiles] = label
            time.sleep(API_SLEEP)

        rows.append({"local": i-1, "label": label})

    log(f"Labeling done in {time.time()-start:.2f}s")
    return pd.DataFrame(rows)


# train multiple models
def train_models(X, y):
    log("Training models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVM": LinearSVC(),
        "KNN": KNeighborsClassifier(),
        "GB": GradientBoostingClassifier()
    }

    for name, model in models.items():
        log(f"\n=== {name} ===")
        t = time.time()

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        log(f"Time: {time.time()-t:.2f}s")
        log(f"Accuracy: {accuracy_score(y_test, pred):.4f}")

        print(classification_report(y_test, pred))


# main 
def main():
    start = time.time()
    log("START")

    _, sampled, idx = load_and_sample()
    df = label_sample(sampled, idx)

    df = df[df["label"].notna()]

    X, y = [], []
    for _, row in df.iterrows():
        spec = sampled[row["local"]]
        X.append(spectrum_to_binned_features(spec))
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    train_models(X, y)

    log(f"TOTAL TIME: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()