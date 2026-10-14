"""
retrieval.py — Lane 2: Traditional lexical retrieval (BM25 / TF-IDF) for ClaimCheck.

Shared interface contract (from the Lane 2 brief):

    def retrieve(claim: str, k: int) -> list[str]
        Return the top-k evidence_id strings (e.g. 'ev_42'), ranked best-first.

Data (built by Lane 1, not rebuilt here):
    ../lane1_data_eval/data/processed/evidence_corpus.csv   (evidence_id, text)

Usage:
    from retrieval import retrieve
    retrieve("Tiber Oil Field is a landlocked oil field.", k=5)
    # -> ['ev_812', 'ev_813', ...]

    retrieve(claim, k, method="tfidf")      # optional: TF-IDF instead of BM25
    retrieve_with_scores(claim, k)          # [(evidence_id, score, text), ...] for the app / sanity check
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORPUS_CSV = Path(os.environ.get(
    "CLAIMCHECK_CORPUS_CSV",
    HERE.parent / "lane1_data_eval" / "data" / "processed" / "evidence_corpus.csv"))
INDEX_DIR = HERE / "indexes"

# BM25 parameters (standard Robertson / Lucene defaults, not tuned)
K1, B = 1.2, 0.75

# ----------------------------------------------------------------------------
# Preprocessing — identical for documents and claims
# ----------------------------------------------------------------------------
_FEVER_ESCAPES = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
                  "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOP
except Exception:
    _STOP = frozenset()
try:
    from nltk.stem import PorterStemmer
    _STEM = PorterStemmer().stem
except Exception:
    _STEM = None


def clean_text(text: str) -> str:
    """Readable version of a FEVER sentence (for display / NLI): '-LRB-' -> '(' etc."""
    text = str(text).replace("``", '"').replace("''", '"')
    for a, b in _FEVER_ESCAPES.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Undo FEVER escapes -> lowercase -> alphanumeric tokens -> drop stopwords -> Porter stem."""
    text = str(text)
    for a, b in _FEVER_ESCAPES.items():
        text = text.replace(a, b)
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]
    return [_STEM(t) for t in toks] if _STEM else toks


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
def load_corpus(path: str | Path = CORPUS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"evidence_id", "text"} <= set(df.columns):
        raise ValueError(f"{path} must have columns evidence_id, text — found {list(df.columns)}")
    df = df.dropna(subset=["evidence_id", "text"]).drop_duplicates("evidence_id").reset_index(drop=True)
    df["evidence_id"] = df["evidence_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def _doc_text(row) -> str:
    return row["text"]


# ----------------------------------------------------------------------------
# Retrievers
# ----------------------------------------------------------------------------
class BM25Index:
    name = "bm25"

    def __init__(self, k1: float = K1, b: float = B):
        self.k1, self.b = k1, b

    def fit(self, df: pd.DataFrame):
        from rank_bm25 import BM25Okapi
        self.ids = df["evidence_id"].tolist()
        self.texts = df["text"].tolist()
        self.bm25 = BM25Okapi([tokenize(_doc_text(r)) for _, r in df.iterrows()], k1=self.k1, b=self.b)
        return self

    def scores(self, claim: str) -> np.ndarray:
        q = tokenize(claim)
        return np.asarray(self.bm25.get_scores(q)) if q else np.zeros(len(self.ids))


class TfidfIndex:
    name = "tfidf"

    def fit(self, df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = df["evidence_id"].tolist()
        self.texts = df["text"].tolist()
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(_doc_text(r) for _, r in df.iterrows())
        return self

    def scores(self, claim: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([claim]).T).toarray().ravel()


def build_index(method: str = "bm25", corpus_csv: str | Path = CORPUS_CSV, save: bool = True):
    idx = {"bm25": BM25Index, "tfidf": TfidfIndex}[method]().fit(load_corpus(corpus_csv))
    if save:
        INDEX_DIR.mkdir(exist_ok=True)
        with open(INDEX_DIR / f"{method}.pkl", "wb") as f:
            pickle.dump(idx, f)
    return idx


_CACHE: dict = {}


def get_index(method: str = "bm25"):
    if method not in _CACHE:
        p = INDEX_DIR / f"{method}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                _CACHE[method] = pickle.load(f)
        else:                                   # no saved index yet -> build it on the fly
            _CACHE[method] = build_index(method, save=True)
    return _CACHE[method]


# ----------------------------------------------------------------------------
# Shared interface
# ----------------------------------------------------------------------------
def retrieve_with_scores(claim: str, k: int, method: str = "bm25") -> list[tuple[str, float, str]]:
    idx = get_index(method)
    s = idx.scores(claim)
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return [(idx.ids[i], float(s[i]), clean_text(idx.texts[i])) for i in top]


def retrieve(claim: str, k: int, method: str = "bm25") -> list[str]:
    """Return the top-k evidence_id strings (e.g. 'ev_42'), ranked best-first."""
    return [eid for eid, _, _ in retrieve_with_scores(claim, k, method)]
