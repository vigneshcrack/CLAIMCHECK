'''
Traditional baseline: TF-IDF + Logistic Regression
Trained and tested using provided claimcheck dataset
Model should perform near chance, as both claims and evidence appear under both labels
'''
import re
from pathlib import Path
import joblib
import pandas as pd
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
 
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "lane1_data_eval" / "data" / "processed" / "claimcheck_dataset.csv"
PREDICTIONS_PATH = HERE / "results/predictions_tfidf.csv"
MODEL_PATH = HERE / "tfidf_baseline.joblib"
C_GRID = (0.1, 1, 10, 100)

# Cleaning data

_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
             "-LCB-": "{", "-RCB-": "}"}
 
 
def clean_fever_text(t):
    if not isinstance(t, str):
        return ""
    for k, v in _BRACKETS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+([.,;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+('s|n't)\b", r"\1", t)
    t = t.replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", t).strip()

# Load data

def load_real_rows(data_path=DATA_PATH):
    df = pd.read_csv(data_path, index_col=0)
    df = df[df.source == "fever_symmetric"].copy()
    df["claim"] = df.claim.map(clean_fever_text)
    df["evidence"] = df.evidence.map(clean_fever_text)
    return tuple(df[df.split == s] for s in ("train", "dev", "test"))   

# Building a model

def _transform(model, claims, evidences):
    #Transforming claim and evidence into numeric matrices for model
    parts = [model["vc"].transform(claims)]
    if model["use_evidence"]:
        parts.append(model["ve"].transform(evidences))
    return hstack(parts).tocsr()

def fit(train, C=1.0, use_evidence=True):
    # Fit on training rows
    vc = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    ve = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    parts = [vc.fit_transform(train.claim)]
    if use_evidence:
        parts.append(ve.fit_transform(train.evidence))
    clf = LogisticRegression(C=C, max_iter=1000).fit(hstack(parts).tocsr(), train.label)
    return {"vc": vc, "ve": ve, "clf": clf, "use_evidence": use_evidence, "C": C}

def _predict_df(model, d):
    proba = model["clf"].predict_proba(_transform(model, d.claim, d.evidence))
    return model["clf"].classes_[proba.argmax(1)], proba.max(1)

# Choosing C using dev split

def choose_C(train, dev, grid=C_GRID, verbose=True):
    best_C, best_acc = None, -1.0
    for C in grid:
        pred, _ = _predict_df(fit(train, C), dev)
        acc = float((pred == dev.label.values).mean())
        if verbose:
            print(f"C={C:<6} dev accuracy = {acc:.3f}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    return best_C


#Final model

_model = None

def _get_model():
    """Lazy-load the saved model; raises a clear error if it hasn't been trained."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python {Path(__file__).name}` first to train it.")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_batch(claims, evidences, model=None):
    """Predict many pairs. Returns [{"label": ..., "confidence": ...}, ...] in input order."""
    model = model or _get_model()
    d = pd.DataFrame({"claim": [clean_fever_text(c) for c in claims],
                      "evidence": [clean_fever_text(e) for e in evidences]})
    pred, conf = _predict_df(model, d)
    return [{"label": str(p), "confidence": float(c)} for p, c in zip(pred, conf)]
 
 
def predict(claim, evidence, model=None):
    return predict_batch([claim], [evidence], model)[0]

def main():
    global _model
    train, dev, test = load_real_rows()
    print("rows  train/dev/test:", len(train), len(dev), len(test))
 
    best_C = choose_C(train, dev)
    print("chosen C:", best_C)
 
    model = fit(train, best_C)
    pred, conf = _predict_df(model, test)
    out = pd.DataFrame({"claim_id": test.index, "predicted_label": pred,
                        "confidence": conf, "gold_label": test.label.values})
    out.to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    _model = model
    print(f"TEST accuracy (claim+evidence TF-IDF): "
          f"{(out.predicted_label == out.gold_label).mean():.3f}")

    claim_only = fit(train, best_C, use_evidence=False)
    pred_c, _ = _predict_df(claim_only, test)
    print(f"TEST accuracy (claim-only, diagnostic): {(pred_c == test.label.values).mean():.3f}")
    print(f"saved {PREDICTIONS_PATH.name} and {MODEL_PATH.name} in {HERE}")
 
 
if __name__ == "__main__":
    main()
 
