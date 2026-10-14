'''
Traditional baseline: TF-IDF + Logistic Regression
Trained and tested using provided claimcheck dataset
Model should perform near chance, as both claims and evidence appear under both labels
'''

import pandas as pd
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
 
DATA = "../lane1_data_eval/data/processed/claimcheck_dataset.csv"

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
 
df = pd.read_csv(DATA, index_col=0)
df = df[df.source == "fever_symmetric"].copy()
df["claim"] = df.claim.map(clean_fever_text)
df["evidence"] = df.evidence.map(clean_fever_text)
train, dev, test = (df[df.split == s] for s in ("train", "dev", "test"))
print("rows  train/dev/test:", len(train), len(dev), len(test))

# Building a model

def build(use_evidence=True):
    vc = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    ve = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
 
    def feats(d, fit=False):
        parts = [vc.fit_transform(d.claim) if fit else vc.transform(d.claim)]
        if use_evidence:
            parts.append(ve.fit_transform(d.evidence) if fit else ve.transform(d.evidence))
        return hstack(parts).tocsr()
    return feats

def fit_eval(C, use_evidence=True):
    feats = build(use_evidence)
    Xtr = feats(train, fit=True)
    clf = LogisticRegression(C=C, max_iter=1000).fit(Xtr, train.label)
    return clf, feats

# Choosing C using dev split

best_C, best_acc = None, -1
for C in (0.1, 1, 10, 100):
    clf, feats = fit_eval(C)
    a = (clf.predict(feats(dev)) == dev.label.values).mean()
    print(f"C={C:<6} dev accuracy = {a:.3f}")
    if a > best_acc:
        best_C, best_acc = C, a
print("chosen C:", best_C)

#Final model

clf, feats = fit_eval(best_C)
proba = clf.predict_proba(feats(test))
pred = clf.classes_[proba.argmax(1)]
out = pd.DataFrame({"claim_id": test.index, "predicted_label": pred,
                    "confidence": proba.max(1), "gold_label": test.label.values})
out.to_csv("predictions_tfidf.csv", index=False)
print(f"TEST accuracy (claim+evidence TF-IDF): {(out.predicted_label == out.gold_label).mean():.3f}")

clf_c, feats_c = fit_eval(best_C, use_evidence=False)
acc_c = (clf_c.predict(feats_c(test)) == test.label.values).mean()
print(f"TEST accuracy (claim-only, diagnostic): {acc_c:.3f}")

real = df[df.source == "fever_symmetric"]
tr, te = real[real.split == "train"], real[real.split == "test"]

for col in ["claim", "evidence", "gold_evidence_id"]:
    shared = set(tr[col]) & set(te[col])
    print(f"{col}: {len(shared)} shared train/test, "
          f"{te[col].isin(shared).sum()} test rows affected")

# For test claims that also appear in train: how often is the label flipped?
m = te[["claim", "label"]].merge(tr[["claim", "label"]].drop_duplicates(),
                                 on="claim", suffixes=("_test", "_train"))
print("rows with a train twin:", len(m),
      "| label differs:", (m.label_test != m.label_train).mean().round(3))